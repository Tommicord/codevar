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

use std::sync::Arc;

use crate::edit::regexp::regexp_pattern::{
    FLAG_CASE_INSENSITIVE, FLAG_COMMENTS, FLAG_DOTALL, FLAG_LITERAL, FLAG_MULTILINE,
    FLAG_UNIX_LINES,
};
use crate::edit::regexp::regexp_pattern_ast::{CharClass, MAX_REPS, Node};
use crate::edit::regexp::regexp_pattern_type::{PatternError, PatternResult};
use CharClass::*;

pub struct Parser<'a> {
    pub pattern: &'a str,
    pub flags: u32,
    pub temp: Vec<u32>,
    pub cursor: usize,
    pub pattern_length: usize,
    pub capturing_group_count: usize,
    pub local_count: usize,
    pub has_group_ref: bool,
    pub has_supplementary: bool,
    pub top_closure_nodes: Vec<Node>,
}

impl<'a> Parser<'a> {
    pub fn new(pattern: &'a str, flags: u32) -> Self {
        let temp: Vec<u32> = pattern.chars().map(|c| c as u32).collect();
        let pattern_length = temp.len();
        Parser {
            pattern,
            flags,
            temp,
            cursor: 0,
            pattern_length,
            capturing_group_count: 1,
            local_count: 0,
            has_group_ref: false,
            has_supplementary: false,
            top_closure_nodes: Vec::new(),
        }
    }

    fn has(&self, flag: u32) -> bool {
        (self.flags & flag) != 0
    }

    pub fn parse(&mut self) -> PatternResult<Node> {
        let accept = Node::Accept;
        let node = if self.has(FLAG_LITERAL) {
            Node::Slice(self.temp.clone())
        } else {
            self.parse_expression(&accept)?
        };

        if self.cursor < self.pattern_length {
            return Err(PatternError::InvalidPattern(
                "Unexpected character at end of pattern".to_string(),
            ));
        }

        Ok(node)
    }

    fn parse_expression(&mut self, accept: &Node) -> PatternResult<Node> {
        let mut head: Option<Node> = None;
        let mut tail: Option<Node> = None;
        let mut root: Option<Node> = None;

        while self.cursor < self.pattern_length {
            let ch = self.temp[self.cursor];

            match ch {
                40 => {
                    // b'('
                    self.cursor += 1;
                    if self.cursor < self.pattern_length && self.temp[self.cursor] == 63 {
                        self.cursor += 1;
                        let node = self.parse_extended_group(accept)?;
                        self.append_node(&mut head, &mut tail, &mut root, node);
                    } else {
                        let local_idx = self.local_count;
                        self.local_count += 1;
                        let group_idx = self.capturing_group_count;
                        self.capturing_group_count += 1;

                        let group_head = Node::GroupHead {
                            local_index: local_idx,
                            group_index: group_idx,
                        };
                        let group_tail = Node::GroupTail {
                            local_index: local_idx,
                            group_index: group_idx,
                        };

                        self.append_node(&mut head, &mut tail, &mut root, group_head);
                        let group_node = self.parse_expression(accept)?;
                        self.append_node(&mut head, &mut tail, &mut root, group_node);
                        self.append_node(&mut head, &mut tail, &mut root, group_tail);

                        if self.cursor >= self.pattern_length
                            || self.temp[self.cursor] != 41
                        {
                            return Err(PatternError::UnmatchedParenthesis);
                        }
                        self.cursor += 1;
                    }
                }
                41 => {
                    break;
                }
                124 => {
                    self.cursor += 1;
                    let alt = self.parse_expression(accept)?;
                    return Ok(Node::Branch(
                        Box::new(head.unwrap_or(Node::Accept)),
                        None,
                        Box::new(alt),
                    ));
                }
                42 | 43 | 63 | 123 => {
                    let (cmin, cmax, greedy) = self.parse_quantifier()?;
                    if let Some(body) = tail.take() {
                        let local_idx = self.local_count;
                        self.local_count += 1;
                        let loop_node = if greedy {
                            Node::Loop {
                                body: Box::new(body),
                                cmin,
                                cmax,
                                pos_index: None,
                                local_index: local_idx,
                            }
                        } else {
                            Node::LazyLoop {
                                body: Box::new(body),
                                cmin,
                                cmax,
                                local_index: local_idx,
                            }
                        };
                        if let Some(tail_mut) = tail.as_mut() {
                            *tail_mut = loop_node.clone();
                        }
                        if let Some(root_mut) = root.as_mut() {
                            *root_mut = loop_node;
                        }
                    } else {
                        return Err(PatternError::InvalidRepeat);
                    }
                }
                92 => {
                    self.cursor += 1;
                    if self.cursor >= self.pattern_length {
                        return Err(PatternError::InvalidEscape);
                    }
                    let escaped = self.parse_escape()?;
                    let node = self.create_node(escaped);
                    self.append_node(&mut head, &mut tail, &mut root, node);
                }
                94 => {
                    self.cursor += 1;
                    let node = if self.has(FLAG_MULTILINE) {
                        Node::Caret
                    } else {
                        Node::Begin
                    };
                    self.append_node(&mut head, &mut tail, &mut root, node);
                }
                36 => {
                    self.cursor += 1;
                    let node = if self.has(FLAG_MULTILINE) {
                        Node::Dollar
                    } else {
                        Node::End
                    };
                    self.append_node(&mut head, &mut tail, &mut root, node);
                }
                46 => {
                    self.cursor += 1;
                    let node = if self.has(FLAG_DOTALL) {
                        Node::BmpCharPredicate(Box::new(Arc::new(|_ch| true)))
                    } else if self.has(FLAG_UNIX_LINES) {
                        Node::BmpCharPredicate(Box::new(Arc::new(|ch| ch != 10)))
                    } else {
                        Node::BmpCharPredicate(Box::new(Arc::new(|ch| {
                            ch != 10 && ch != 13 && (ch | 1) != 0x2029 && ch != 0x0085
                        })))
                    };
                    self.append_node(&mut head, &mut tail, &mut root, node);
                }
                35 if self.has(FLAG_COMMENTS) => {
                    self.cursor += 1;
                    while self.cursor < self.pattern_length
                        && self.temp[self.cursor] != 10
                    {
                        self.cursor += 1;
                    }
                }
                _ => {
                    self.cursor += 1;
                    let node = if self.has(FLAG_CASE_INSENSITIVE) {
                        let lower = char::from_u32(ch)
                            .ok_or(PatternError::InvalidEscape)?
                            .to_lowercase()
                            .next()
                            .unwrap_or_else(|| 0 as char);
                        let upper = char::from_u32(ch)
                            .ok_or(PatternError::InvalidEscape)?
                            .to_uppercase()
                            .next()
                            .unwrap_or_else(|| 0 as char);
                        Node::SingleI(lower as u32, upper as u32)
                    } else {
                        Node::Single(ch)
                    };
                    self.append_node(&mut head, &mut tail, &mut root, node);
                }
            }
        }

        Ok(head.unwrap_or(Node::Accept))
    }

    fn parse_extended_group(&mut self, accept: &Node) -> PatternResult<Node> {
        if self.cursor >= self.pattern_length {
            return Err(PatternError::InvalidPattern(
                "Incomplete extended group".to_string(),
            ));
        }

        let ch = self.temp[self.cursor];
        match ch {
            58 => {
                self.cursor += 1;
                self.parse_expression(accept)
            }
            61 => {
                self.cursor += 1;
                let node = self.parse_expression(accept)?;
                Ok(Node::Cond(
                    Box::new(node),
                    Box::new(Node::Accept),
                    Box::new(Node::Accept),
                ))
            }
            33 => {
                self.cursor += 1;
                let node = self.parse_expression(accept)?;
                Ok(Node::Cond(
                    Box::new(node),
                    Box::new(Node::Accept),
                    Box::new(Node::Accept),
                ))
            }
            62 => {
                self.cursor += 1;
                let node = self.parse_expression(accept)?;
                Ok(node)
            }
            105 | 109 | 115 | 100 | 117 | 99 | 120 | 45 => {
                self.parse_flags();
                self.parse_expression(accept)
            }
            _ => Err(PatternError::InvalidPattern(format!(
                "Unknown extended group: {}",
                ch as u8 as char
            ))),
        }
    }

    fn parse_flags(&mut self) {
        while self.cursor < self.pattern_length && self.temp[self.cursor] != 41 {
            self.cursor += 1;
        }
        if self.cursor < self.pattern_length {
            self.cursor += 1;
        }
    }

    fn parse_quantifier(&mut self) -> PatternResult<(u32, u32, bool)> {
        if self.cursor >= self.pattern_length {
            return Err(PatternError::InvalidRepeat);
        }

        let ch = self.temp[self.cursor];
        self.cursor += 1;

        let (cmin, cmax) = match ch {
            42 => (0, MAX_REPS),
            43 => (1, MAX_REPS),
            63 => (0, 1),
            123 => {
                let mut cmin = 0u32;
                let mut cmax = MAX_REPS;

                while self.cursor < self.pattern_length
                    && self.temp[self.cursor] >= 48
                    && self.temp[self.cursor] <= 57
                {
                    cmin = cmin * 10 + (self.temp[self.cursor] as u8 - 48) as u32;
                    self.cursor += 1;
                }

                if self.cursor < self.pattern_length && self.temp[self.cursor] == 44 {
                    self.cursor += 1;
                    if self.cursor < self.pattern_length
                        && self.temp[self.cursor] >= 48
                        && self.temp[self.cursor] <= 57
                    {
                        cmax = 0;
                        while self.cursor < self.pattern_length
                            && self.temp[self.cursor] >= 48
                            && self.temp[self.cursor] <= 57
                        {
                            cmax = cmax * 10 + (self.temp[self.cursor] as u8 - 48) as u32;
                            self.cursor += 1;
                        }
                    } else {
                        cmax = MAX_REPS;
                    }
                } else {
                    cmax = cmin;
                }

                if self.cursor >= self.pattern_length || self.temp[self.cursor] != 125 {
                    return Err(PatternError::InvalidRepeat);
                }
                self.cursor += 1;
                (cmin, cmax)
            }
            _ => return Err(PatternError::InvalidRepeat),
        };

        let greedy = if self.cursor < self.pattern_length && self.temp[self.cursor] == 63
        {
            self.cursor += 1;
            false
        } else {
            true
        };

        Ok((cmin, cmax, greedy))
    }

    fn parse_escape(&mut self) -> PatternResult<u32> {
        let ch = self.temp[self.cursor];
        self.cursor += 1;

        match ch {
            48..=55 => {
                let mut val = (ch as u8 - 48) as u32;
                if self.cursor < self.pattern_length
                    && self.temp[self.cursor] >= 48
                    && self.temp[self.cursor] <= 55
                {
                    val = val * 8 + (self.temp[self.cursor] as u8 - 48) as u32;
                    self.cursor += 1;
                    if self.cursor < self.pattern_length
                        && self.temp[self.cursor] >= 48
                        && self.temp[self.cursor] <= 55
                    {
                        val = val * 8 + (self.temp[self.cursor] as u8 - 48) as u32;
                        self.cursor += 1;
                    }
                }
                Ok(val)
            }
            120 => {
                if self.cursor + 1 >= self.pattern_length {
                    return Err(PatternError::InvalidEscape);
                }
                let hi = self.hex_value(self.temp[self.cursor]);
                let lo = self.hex_value(self.temp[self.cursor + 1]);
                self.cursor += 2;
                Ok(hi * 16 + lo)
            }
            117 => {
                if self.cursor + 3 >= self.pattern_length {
                    return Err(PatternError::InvalidEscape);
                }
                let mut val = 0u32;
                for _ in 0..4 {
                    val = val * 16 + self.hex_value(self.temp[self.cursor]);
                    self.cursor += 1;
                }
                Ok(val)
            }
            116 => Ok(9),
            110 => Ok(10),
            114 => Ok(13),
            102 => Ok(12),
            97 => Ok(7),
            101 => Ok(27),
            100 => Ok(0xFFFFFFFE),
            68 => Ok(0xFFFFFFFD),
            115 => Ok(0xFFFFFFFC),
            83 => Ok(0xFFFFFFFB),
            119 => Ok(0xFFFFFFFA),
            87 => Ok(0xFFFFFFF9),
            49..=57 => {
                self.has_group_ref = true;
                Ok((ch as u8 - 48) as u32)
            }
            _ => Ok(ch),
        }
    }

    fn hex_value(&self, ch: u32) -> u32 {
        match ch {
            48..=57 => (ch as u8 - 48) as u32,
            97..=102 => (ch as u8 - 97) as u32 + 10,
            65..=70 => (ch as u8 - 65) as u32 + 10,
            _ => 0,
        }
    }

    fn create_node(&mut self, ch: u32) -> Node {
        if ch > 0xFFFF {
            self.has_supplementary = true;
        }

        match ch {
            0xFFFFFFFE => Node::CharClass(CharClass::Digit),
            0xFFFFFFFD => Node::CharClass(CharClass::NonDigit),
            0xFFFFFFFC => Node::CharClass(CharClass::Whitespace),
            0xFFFFFFFB => Node::CharClass(CharClass::NonWhitespace),
            0xFFFFFFFA => Node::CharClass(CharClass::Word),
            0xFFFFFFF9 => Node::CharClass(CharClass::NonWord),
            _ => {
                if self.has(FLAG_CASE_INSENSITIVE) {
                    let lower = (ch as u8 as char).to_ascii_lowercase() as u32;
                    let upper = (ch as u8 as char).to_ascii_uppercase() as u32;
                    Node::SingleI(lower, upper)
                } else {
                    Node::Single(ch)
                }
            }
        }
    }

    fn append_node(
        &self,
        head: &mut Option<Node>,
        tail: &mut Option<Node>,
        root: &mut Option<Node>,
        node: Node,
    ) {
        if head.is_none() {
            *head = Some(node.clone());
        }
        // For sequences we just update tail and root
        // The actual sequence matching is handled by the match_node function
        *tail = Some(node.clone());
        *root = tail.clone();
    }
}
