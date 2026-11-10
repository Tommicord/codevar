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

//! The planar boundary graph: junction nodes and boundary edges.
//!
//! Every edge is shared by exactly two faces; nodes are points where four
//! (or more) region corners meet. This is the structure the mosaic fitter
//! walks to assign one fitted geometry per shared segment.

use crate::types::PointI32;

/// A node in the boundary graph: a pixel-space point where segments meet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Node {
    pub point: PointI32,
    /// Whether this is a junction (degree > 2) vs. a simple boundary point.
    pub junction: bool,
}

/// An edge in the boundary graph: connects two nodes and is shared by two faces.
#[derive(Debug, Clone)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    /// Pixel-space points along the edge (excluding `from`, including `to`).
    pub points: Vec<PointI32>,
    /// The face index on one side of the edge.
    pub face_a: usize,
    /// The face index on the other side of the edge.
    pub face_b: usize,
}

/// The planar boundary graph of a set of region masks.
#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.edges.is_empty()
    }

    /// Add a node, returning its index (deduplicating by point).
    pub fn add_node(&mut self, point: PointI32, junction: bool) -> usize {
        if let Some(idx) = self.nodes.iter().position(|n| n.point == point) {
            // Upgrade a plain node to a junction if requested.
            if junction && !self.nodes[idx].junction {
                self.nodes[idx].junction = true;
            }
            return idx;
        }
        let idx = self.nodes.len();
        self.nodes.push(Node { point, junction });
        idx
    }

    /// Add an edge between two node indices.
    pub fn add_edge(&mut self, from: usize, to: usize, points: Vec<PointI32>, face_a: usize, face_b: usize) {
        self.edges.push(Edge {
            from,
            to,
            points,
            face_a,
            face_b,
        });
    }

    /// All edges incident on a node.
    pub fn incident_edges(&self, node: usize) -> Vec<usize> {
        self.edges
            .iter()
            .enumerate()
            .filter(|(_, e)| e.from == node || e.to == node)
            .map(|(i, _)| i)
            .collect()
    }

    /// The edge connecting `node` to `neighbor`, if any.
    pub fn edge_between(&self, node: usize, neighbor: usize) -> Option<usize> {
        self.edges
            .iter()
            .position(|e| (e.from == node && e.to == neighbor) || (e.from == neighbor && e.to == node))
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}
