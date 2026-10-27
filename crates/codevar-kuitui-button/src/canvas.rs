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

//! The `concrete` decoration: custom drawing with the [`Canvas`] widget.
//!
//! [`codevar_kuitui::Modifier::concrete`] takes a closure, which is the
//! most general hook the framework offers — the application paints
//! whatever it likes around the node it decorates. A closure is awkward
//! to keep around though: the shapes it draws usually change between two
//! frames, and a `Fn` cannot capture the state it mutates.
//!
//! [`Concrete`] is the shared-handle answer. It owns a [`Canvas`] behind
//! an `Rc<RefCell<_>>`, which is the widget the application already knows
//! how to fill — `fill_rounded_rect`, `stroke_rect`, `line`, `polygon`,
//! `gradient` — and turns it into a decoration whose drawing is replayed
//! on every frame from the current shape list. The application keeps its
//! own clone of the handle and edits the canvas as often as it likes; the
//! decoration never has to be rebuilt, so the tree stays intact.
//!
//! The shapes are in the coordinate space of the decorated node, with the
//! same origin the [`Canvas`] widget uses when it is the node's own
//! widget, so a shape list written for one paints identically in the
//! other.

use alloc::rc::Rc;
use core::cell::RefCell;

use codevar_kuitui::{DrawScope, Modifier, Offset, PaintStyle, Rect, Size};
use codevar_kuitui_widget::canvas::{gradient_bands, polygon_edges, polygon_spans};
use codevar_kuitui_widget::{Canvas, Shape};

/// Expands every shape of `canvas` into operations of the display list.
///
/// The expansion is the one the [`Canvas`] widget applies to itself: a
/// rounded or sharp rectangle, a stroked outline, a scanline-filled
/// polygon and a banded gradient, each in the coordinate space of the
/// node being drawn.
pub fn paint(scope: &mut DrawScope<'_>, canvas: &Canvas) {
    for shape in canvas.shapes() {
        paint_shape(scope, shape);
    }
}

/// Expands one shape into the operations of the display list.
fn paint_shape(scope: &mut DrawScope<'_>, shape: &Shape) {
    match shape {
        Shape::Rect { rect, radii, paint } => match paint.style {
            PaintStyle::Fill => scope.fill_rounded_rect(*rect, *radii, paint.color),
            PaintStyle::Stroke(width) => scope.stroke_rect(*rect, *radii, paint.color, width),
        },
        Shape::Line {
            from,
            to,
            color,
            width,
        } => scope.draw_line(*from, *to, *color, *width),
        Shape::Polygon { points, paint } => match paint.style {
            PaintStyle::Fill => {
                for span in polygon_spans(points) {
                    let rect =
                        Rect::from_min_size(Offset::new(span.left, span.y), span.right - span.left, 1.0);
                    scope.fill_rect(rect, paint.color);
                }
            }
            PaintStyle::Stroke(width) => {
                for (from, to) in polygon_edges(points) {
                    scope.draw_line(from, to, paint.color, width);
                }
            }
        },
        Shape::Gradient {
            rect,
            gradient,
            steps,
        } => {
            for (band, color) in gradient_bands(*rect, gradient, *steps) {
                scope.fill_rect(band, color);
            }
        }
    }
}

/// The decoration that paints a [`Canvas`] around the node it decorates.
///
/// The canvas is shared: what it holds when a frame is drawn is what that
/// frame shows. A decoration built from another decoration, or from a
/// clone of the handle, draws the same shapes, because there is only one
/// canvas behind them.
#[derive(Debug, Clone)]
pub struct Concrete {
    /// The shapes to paint, shared with the application.
    canvas: Rc<RefCell<Canvas>>,
}

impl Concrete {
    /// The decoration that paints an empty canvas of `size`.
    #[must_use]
    pub fn with_size(size: Size) -> Self {
        Self {
            canvas: Rc::new(RefCell::new(Canvas::new(size))),
        }
    }

    /// The decoration that paints `canvas`.
    #[must_use]
    pub fn with_canvas(canvas: Canvas) -> Self {
        Self {
            canvas: Rc::new(RefCell::new(canvas)),
        }
    }

    /// The handle of the canvas the decoration paints.
    ///
    /// The application keeps this, records into it, and clears it; the
    /// decoration picks the change up on the next frame without the tree
    /// being rebuilt.
    #[must_use]
    pub fn canvas(&self) -> Rc<RefCell<Canvas>> {
        Rc::clone(&self.canvas)
    }

    /// The decoration whose canvas has `build` run over it once.
    #[must_use]
    pub fn drawn(self, build: impl FnOnce(&mut Canvas)) -> Self {
        build(&mut self.canvas.borrow_mut());
        self
    }

    /// The modifier operation this decoration adds to a node.
    ///
    /// The framework paints a concrete operation after the container of
    /// the modifier chain and the background pass of the widget and
    /// before the content of the node, so the shapes sit over every
    /// container — the one of the chain and the opaque one a button's
    /// own widget paints in its background pass — and under the label.
    #[must_use]
    pub fn modifier(&self) -> Modifier {
        let canvas = Rc::clone(&self.canvas);
        Modifier::none().concrete(move |scope, _bounds| {
            let shared = canvas.borrow();
            paint(scope, &shared);
        })
    }
}
