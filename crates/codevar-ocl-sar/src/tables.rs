//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of
//! the License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Side tables keyed by [`NodeId`], produced alongside diagnostics.
//!
//! Following clang's `Sema`/`ASTContext pattern, the analyzer records
//! what it proved — the type of every expression and what every name
//! resolves to — instead of mutating the tree. Lowering and later
//! passes read these tables to map each node to its type and its
//! definition without re-running name resolution.

use alloc::string::String;
use alloc::vec::Vec;

use codevar_ocl_parse::NodeId;

use crate::types::Ty;

/// The recorded type of every type-checked expression.
///
/// Entries are sparse: nodes the analyzer never types (for example the
/// callee of a call, which is resolved by name rather than checked as a
/// value) simply have no entry.
///
/// # Examples
///
/// ```
/// use codevar_ocl_sar::TypeTable;
/// # use codevar_ocl_sar::{NodeId, Ty};
///
/// let mut types = TypeTable::new();
/// let id = NodeId::from_raw(0);
/// types.insert(id, Ty::bool());
/// assert_eq!(types.get(id), Some(&Ty::bool()));
/// assert!(!types.contains(NodeId::from_raw(1)));
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TypeTable {
    /// Indexed by the raw id; `None` marks an id that was never checked.
    entries: Vec<Option<Ty>>,
}

impl TypeTable {
    /// Creates an empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records `ty` as the type of the node `id`.
    ///
    /// [`NodeId::DUMMY`] nodes are ignored; a repeat call for the same
    /// id overwrites the previous entry.
    pub fn insert(&mut self, id: NodeId, ty: Ty) {
        let Some(index) = id.index() else {
            return;
        };
        if index >= self.entries.len() {
            self.entries.resize(index + 1, None);
        }
        self.entries[index] = Some(ty);
    }

    /// The recorded type of the node `id`, if it was checked.
    #[must_use]
    pub fn get(&self, id: NodeId) -> Option<&Ty> {
        let index = id.index()?;
        self.entries.get(index)?.as_ref()
    }

    /// True when the node `id` has a recorded type.
    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        self.get(id).is_some()
    }

    /// Number of recorded entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.is_some())
            .count()
    }

    /// True when no types have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// What a name resolved to at a use site.
#[derive(Debug, Clone, PartialEq)]
pub enum Res {
    /// A variable, parameter, or pattern binding; `binding` is the id
    /// of the [`Pat`](codevar_ocl_parse::Pat) that introduced it.
    Local {
        /// Identity of the introducing pattern.
        binding: NodeId,
    },
    /// A user-defined function or kernel.
    Function {
        /// Resolved function name.
        name: String,
    },
    /// An intrinsic from the builtin library.
    Builtin {
        /// Resolved builtin name.
        name: String,
    },
}

/// The recorded resolution of every resolved name use.
///
/// Callee paths of calls and variable paths read as values both get an
/// entry; paths that failed resolution have none.
///
/// # Examples
///
/// ```
/// use codevar_ocl_sar::{Res, ResolutionTable};
/// # use codevar_ocl_sar::NodeId;
///
/// let mut resolutions = ResolutionTable::new();
/// resolutions.insert(NodeId::from_raw(3), Res::Function { name: "foo".into() });
/// assert!(matches!(
///     resolutions.get(NodeId::from_raw(3)),
///     Some(Res::Function { name }) if name == "foo"
/// ));
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolutionTable {
    /// Indexed by the raw id; `None` marks a path that resolved to
    /// nothing.
    entries: Vec<Option<Res>>,
}

impl ResolutionTable {
    /// Creates an empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records what `id` resolved to.
    ///
    /// [`NodeId::DUMMY`] nodes are ignored; a repeat call for the same
    /// id overwrites the previous entry.
    pub fn insert(&mut self, id: NodeId, res: Res) {
        let Some(index) = id.index() else {
            return;
        };
        if index >= self.entries.len() {
            self.entries.resize(index + 1, None);
        }
        self.entries[index] = Some(res);
    }

    /// The recorded resolution of the node `id`, if it resolved.
    #[must_use]
    pub fn get(&self, id: NodeId) -> Option<&Res> {
        let index = id.index()?;
        self.entries.get(index)?.as_ref()
    }

    /// True when the node `id` has a recorded resolution.
    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        self.get(id).is_some()
    }

    /// Number of recorded entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.is_some())
            .count()
    }

    /// True when no resolutions have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
