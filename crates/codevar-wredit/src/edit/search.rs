//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Accelerated whole-document search and replace.
//!
//! A search is a bulk scan, so it goes through [`BulkOps`]: the
//! match flags come from the GPU `find_matches` kernel on large
//! documents and from the SIMD first-word filter on small ones, with
//! no caller-visible difference.
//!
//! [`PackedSource`] is the seam back to the edit engine: it exposes a
//! document's committed packed elements, so a search runs over the
//! writable without knowing the cursor pool or the gap layout. The
//! needle is compared as packed elements too, so a caller builds it
//! with the engine's [`make_packed`](crate::edit::make_packed).

use alloc::vec::Vec;

use crate::dispatch::BulkOps;

use super::writable::StreamWritable;

/// Read-only access to the packed elements of a document.
///
/// Implemented for the writable core so search and replace can read a
/// document without depending on its cursor or gap layout.
pub trait PackedSource {
    /// The committed packed elements, in document order.
    fn packed_elements(&self) -> &[u32];
}

impl<const GAP_SIZE: usize> PackedSource for StreamWritable<u32, u32, GAP_SIZE> {
    fn packed_elements(&self) -> &[u32] {
        self.base.raw_ref().slice()
    }
}

/// Every position of `haystack` that starts a `needle` match.
///
/// Positions are returned in ascending order and include overlapping
/// matches, so `replace_all` can fold them into a non-overlapping edit.
#[must_use]
pub fn find_all(haystack: &[u32], needle: &[u32], ops: &impl BulkOps) -> Vec<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return Vec::new();
    }
    let flags = ops.find_matches(haystack, needle);
    let mut positions = Vec::new();
    for (index, &flag) in flags.iter().enumerate() {
        if flag != 0 {
            positions.push(index);
        }
    }
    positions
}

/// Replaces every non-overlapping occurrence of `needle` in
/// `haystack` with `replacement`.
///
/// Matches are taken left to right; once one is accepted the scan
/// resumes after its whole length, so `aaa` with needle `aa` becomes
/// `replacement` followed by the trailing `a`. A haystack with no match
/// is returned unchanged.
#[must_use]
pub fn replace_all(haystack: &[u32], needle: &[u32], replacement: &[u32], ops: &impl BulkOps) -> Vec<u32> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return haystack.to_vec();
    }
    let flags = ops.find_matches(haystack, needle);
    let mut out = Vec::with_capacity(haystack.len());
    let mut index = 0;
    while index < haystack.len() {
        if index + needle.len() <= haystack.len() && flags.get(index).copied().unwrap_or(0) != 0 {
            out.extend_from_slice(replacement);
            index += needle.len();
        } else {
            out.push(haystack[index]);
            index += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{PackedSource, find_all, replace_all};
    use crate::dispatch::Accelerator;
    use crate::edit::base::{T_COMMON, make_packed};
    use crate::edit::cursor::Cursor;
    use crate::edit::writable::StreamWritable;
    use crate::edit::{TextEditablePut, Writable};

    /// Packs a plain committed character the way the engine stores it.
    fn plain(character: u8) -> u32 {
        make_packed(u32::from(character), T_COMMON)
    }

    /// A flushed document holding `text`.
    fn stream_with(text: &[u8]) -> StreamWritable<u32, u32, 16> {
        let mut stream = StreamWritable::new();
        let Some(id) = stream.append_cursor(Cursor::new()) else {
            return stream;
        };
        stream.set_active_cursor(id);
        for &character in text {
            let _ = stream.put(0, plain(character));
        }
        stream.flush_all();
        stream
    }

    #[test]
    fn find_all_reports_every_match() {
        let ops = Accelerator::with_defaults();
        let haystack = [plain(b'a'), plain(b'b'), 9, plain(b'a'), plain(b'b')];
        let needle = [plain(b'a'), plain(b'b')];
        assert_eq!(find_all(&haystack, &needle, &ops), alloc::vec![0, 3]);
    }

    #[test]
    fn find_all_handles_overlap_and_missing_needles() {
        let ops = Accelerator::with_defaults();
        let haystack = [1u32, 2, 1, 2, 1];
        // `1 2 1` occurs at 0 and 2; the starts overlap at position 2.
        assert_eq!(find_all(&haystack, &[1, 2, 1], &ops), alloc::vec![0, 2]);
        assert_eq!(find_all(&haystack, &[9], &ops), alloc::vec![]);
        assert_eq!(find_all(&haystack, &[], &ops), alloc::vec![]);
        assert_eq!(find_all(&haystack, &[1, 2, 1, 2, 1, 2], &ops), alloc::vec![]);
    }

    #[test]
    fn replace_all_folds_non_overlapping_matches() {
        let ops = Accelerator::with_defaults();
        // `aaa` with needle `aa` consumes the first two characters.
        assert_eq!(
            replace_all(&[1, 1, 1], &[1, 1], &[7, 7, 7], &ops),
            alloc::vec![7, 7, 7, 1]
        );
        assert_eq!(
            replace_all(&[1, 2, 3, 1, 2, 3], &[1, 2], &[9], &ops),
            alloc::vec![9, 3, 9, 3]
        );
        // No match: the haystack survives untouched.
        assert_eq!(replace_all(&[4, 5], &[8], &[9], &ops), alloc::vec![4, 5]);
    }

    #[test]
    fn packed_source_exposes_the_committed_document() {
        let stream = stream_with(b"banana");
        let accelerator = Accelerator::with_defaults();
        let elements = stream.packed_elements();
        assert_eq!(elements.len(), 6);
        assert_eq!(elements[0], plain(b'b'));
        assert_eq!(find_all(elements, &[plain(b'a')], &accelerator).len(), 3);
        assert_eq!(elements, stream.packed_elements());
    }
}
