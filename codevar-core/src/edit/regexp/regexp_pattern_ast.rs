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
//! distributed on "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

use std::sync::Arc;

#[derive(Clone)]
pub enum CharClass {
    Digit,
    NonDigit,
    Whitespace,
    NonWhitespace,
    Word,
    NonWord,
    Custom(Vec<(u32, u32)>),
}

#[derive(Clone)]
pub enum Node {
    Single(u32),
    SingleI(u32, u32),
    CharClass(CharClass),
    BmpCharPredicate(Box<Arc<dyn Fn(u32) -> bool + Send + Sync>>),
    Begin,
    End,
    Caret,
    Dollar,
    Slice(Vec<u32>),
    Branch(Box<Node>, Option<Box<Node>>, Box<Node>),
    GroupHead {
        local_index: usize,
        group_index: usize,
    },
    GroupTail {
        local_index: usize,
        group_index: usize,
    },
    Loop {
        body: Box<Node>,
        cmin: u32,
        cmax: u32,
        pos_index: Option<usize>,
        local_index: usize,
    },
    LazyLoop {
        body: Box<Node>,
        cmin: u32,
        cmax: u32,
        local_index: usize,
    },
    Cond(Box<Node>, Box<Node>, Box<Node>),
    Ref(usize),
    Accept,
    Prolog(Box<Node>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qtype {
    GREEDY,
    LAZY,
    POSSESSIVE,
}

#[derive(Default)]
pub struct TreeInfo {
    pub min_length: usize,
    pub max_length: usize,
    pub max_valid: bool,
    pub deterministic: bool,
}

impl TreeInfo {
    pub fn reset(&mut self) {
        self.min_length = 0;
        self.max_length = 0;
        self.max_valid = true;
        self.deterministic = true;
    }
}

pub const MAX_REPS: u32 = 0x7FFFFFFF;
