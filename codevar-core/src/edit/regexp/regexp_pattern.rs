//! Copyright 2026 Codevar
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

use crate::edit::regexp::regexp_parse::Parser;
use crate::edit::regexp::regexp_pattern_ast::{CharClass, Node};
use crate::edit::regexp::regexp_pattern_type::PatternResult;
use std::collections::HashMap;
use std::fmt;

pub const FLAG_UNIX_LINES: u32 = 0x01;
pub const FLAG_CASE_INSENSITIVE: u32 = 0x02;
pub const FLAG_COMMENTS: u32 = 0x04;
pub const FLAG_MULTILINE: u32 = 0x08;
pub const FLAG_LITERAL: u32 = 0x10;
pub const FLAG_DOTALL: u32 = 0x20;
pub const FLAG_UNICODE_CASE: u32 = 0x40;
pub const FLAG_CANON_EQ: u32 = 0x80;
pub const FLAG_UNICODE_CHARACTER_CLASS: u32 = 0x100;

pub struct Pattern {
    pattern: String,
    flags: u32,
    root: Option<Node>,
    match_root: Option<Node>,
    capturing_group_count: usize,
    local_count: usize,
    local_tcn_count: usize,
    named_groups: HashMap<String, usize>,
    group_nodes: Vec<Option<Node>>,
    group_starts: Vec<usize>,
    group_ends: Vec<usize>,
}

impl Pattern {
    pub fn compile(pattern: &str) -> PatternResult<Self> {
        Self::compile_with_flags(pattern, 0)
    }

    pub fn compile_with_flags(pattern: &str, flags: u32) -> PatternResult<Self> {
        let mut p = Pattern {
            pattern: pattern.to_string(),
            flags,
            root: None,
            match_root: None,
            capturing_group_count: 0,
            local_count: 0,
            local_tcn_count: 0,
            named_groups: HashMap::new(),
            group_nodes: vec![None; 10],
            group_starts: vec![0; 10],
            group_ends: vec![0; 10],
        };

        p.compile_internal()?;
        Ok(p)
    }

    fn has(&self, flag: u32) -> bool {
        (self.flags & flag) != 0
    }

    fn compile_internal(&mut self) -> PatternResult<()> {
        let mut parser = Parser::new(&self.pattern, self.flags);
        let match_root = parser.parse()?;

        self.match_root = Some(match_root);
        self.root = Some(Node::Begin);
        self.capturing_group_count = parser.capturing_group_count;
        self.local_count = parser.local_count;
        self.local_tcn_count = 0;

        Ok(())
    }

    pub fn matches(&self, text: &str) -> bool {
        if let Some(ref match_root) = self.match_root {
            let chars: Vec<u32> = text.chars().map(|c| c as u32).collect();
            let mut group_starts = vec![0; self.capturing_group_count];
            let mut group_ends = vec![0; self.capturing_group_count];
            self.match_node(match_root, &chars, 0, &mut group_starts, &mut group_ends)
        } else {
            false
        }
    }

    fn match_node(
        &self,
        node: &Node,
        text: &Vec<u32>,
        pos: usize,
        group_starts: &mut Vec<usize>,
        group_ends: &mut Vec<usize>,
    ) -> bool {
        match node {
            Node::Accept => true,
            Node::Single(ch) => pos < text.len() && text[pos] == *ch,
            Node::SingleI(lower, upper) => {
                if pos < text.len() {
                    let c = text[pos];
                    c == *lower || c == *upper
                } else {
                    false
                }
            }
            Node::CharClass(char_class) => {
                if pos >= text.len() {
                    return false;
                }
                let c = text[pos];
                match char_class {
                    CharClass::Digit => (c as u8).is_ascii_digit(),
                    CharClass::NonDigit => !(c as u8).is_ascii_digit(),
                    CharClass::Whitespace => {
                        c == 9 || c == 10 || c == 13 || c == 32 || c == 0x0C || c == 0x85
                    }
                    CharClass::NonWhitespace => {
                        !(c == 9
                            || c == 10
                            || c == 13
                            || c == 32
                            || c == 0x0C
                            || c == 0x85)
                    }
                    CharClass::Word => (c as u8).is_ascii_alphanumeric() || c == 95,
                    CharClass::NonWord => !((c as u8).is_ascii_alphanumeric() || c == 95),
                    CharClass::Custom(ranges) => {
                        ranges.iter().any(|(start, end)| c >= *start && c <= *end)
                    }
                }
            }
            Node::Begin => pos == 0,
            Node::End => pos == text.len(),
            Node::Caret => pos == 0 || (pos > 0 && text[pos - 1] == 10),
            Node::Dollar => pos == text.len() || (pos < text.len() && text[pos] == 10),
            Node::Slice(slice_chars) => {
                if pos + slice_chars.len() <= text.len() {
                    &text[pos..pos + slice_chars.len()] == slice_chars
                } else {
                    false
                }
            }
            Node::BmpCharPredicate(pred) => {
                if pos < text.len() {
                    pred(text[pos])
                } else {
                    false
                }
            }
            Node::Branch(left, _, right) => {
                if self.match_node(left, text, pos, group_starts, group_ends) {
                    true
                } else {
                    self.match_node(right, text, pos, group_starts, group_ends)
                }
            }
            Node::GroupHead { group_index, .. } => {
                if *group_index < group_starts.len() {
                    group_starts[*group_index] = pos;
                }
                true
            }
            Node::GroupTail { group_index, .. } => {
                if *group_index < group_ends.len() {
                    group_ends[*group_index] = pos;
                }
                true
            }
            Node::Loop {
                body, cmin, cmax, ..
            } => {
                let mut count = 0;
                let mut current_pos = pos;

                // Match minimum required
                while count < *cmin && current_pos < text.len() {
                    if !self.match_node(body, text, current_pos, group_starts, group_ends)
                    {
                        return false;
                    }
                    count += 1;
                    current_pos += 1;
                }

                while count < *cmax && current_pos < text.len() {
                    if self.match_node(body, text, current_pos, group_starts, group_ends)
                    {
                        count += 1;
                        current_pos += 1;
                    } else {
                        break;
                    }
                }

                true
            }
            Node::LazyLoop { body, cmin, .. } => {
                let mut count = 0;
                let mut current_pos = pos;

                // Match minimum required
                while count < *cmin && current_pos < text.len() {
                    if !self.match_node(body, text, current_pos, group_starts, group_ends)
                    {
                        return false;
                    }
                    count += 1;
                    current_pos += 1;
                }
                true
            }
            Node::Prolog(node) => {
                self.match_node(node, text, pos, group_starts, group_ends)
            }
            Node::Cond(_, _, _) => true,
            Node::Ref(group) => {
                if *group >= group_starts.len() || *group >= group_ends.len() {
                    return false;
                }

                let start = group_starts[*group];
                let end = group_ends[*group];

                if end <= start || start >= text.len() || end > text.len() {
                    return false;
                }

                let captured_len = end - start;
                if pos + captured_len > text.len() {
                    return false;
                }

                &text[start..end] == &text[pos..pos + captured_len]
            }
        }
    }

    pub fn group_count(&self) -> usize {
        self.capturing_group_count
    }

    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    pub fn flags(&self) -> u32 {
        self.flags
    }
}

impl fmt::Debug for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pattern")
            .field("pattern", &self.pattern)
            .field("flags", &self.flags)
            .field("capturing_group_count", &self.capturing_group_count)
            .finish()
    }
}
