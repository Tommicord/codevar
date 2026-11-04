//! Copyright 2026 Codevar Project
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

//! Display-list expansion, the immediate block and the clip scissor.

use super::*;
use alloc::rc::Rc;
use codevar_kuitui::effect::{EffectKinds, Gradient};
use codevar_kuitui::text::TextStyle;

/// A black fill, the color most tests do not care about.
fn fill() -> Paint {
    Paint::fill(Color::BLACK)
}

#[test]
fn a_filled_rectangle_expands_to_two_triangles() {
    let mut list = DrawList::new();
    list.rect(
        Rect::from_min_size(Offset::new(4.0, 8.0), 10.0, 20.0),
        CornerRadii::ZERO,
        Paint::fill(Color::RED),
    );
    let mut geometry = UiGeometry::new();
    geometry.expand(&mut list);

    assert_eq!(geometry.vertices().len(), 6);
    assert_eq!(geometry.spans().len(), 1);
    let span = geometry.spans()[0];
    assert_eq!(span.first_vertex, 0);
    assert_eq!(span.vertex_count, 6);
    assert_eq!(span.clip, None);
    assert_eq!(span.effect, None, "a plain record needs no effect program");

    let first = geometry.vertices()[0];
    assert_eq!(first.position, [4.0, 8.0]);
    assert_eq!(first.color, channels(Color::RED));
    assert!(list.is_empty(), "expand drains the FIFO queue");
}

#[test]
fn spans_follow_the_paint_order_and_tile_the_buffer() {
    let mut list = DrawList::new();
    list.rect(
        Rect::from_min_size(Offset::ZERO, 4.0, 4.0),
        CornerRadii::ZERO,
        fill(),
    );
    list.line(
        Offset::ZERO,
        Offset::new(4.0, 0.0),
        Paint::stroke(Color::WHITE, 2.0),
    );
    let mut geometry = UiGeometry::new();
    geometry.expand(&mut list);

    let spans = geometry.spans();
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].first_vertex, 0);
    assert_eq!(spans[0].vertex_count, 6);
    assert_eq!(spans[1].first_vertex, 6);
    assert_eq!(spans[1].vertex_count, 6);
    assert_eq!(geometry.vertices().len(), 12);
}

#[test]
fn a_span_carries_the_command_clip() {
    let mut list = DrawList::new();
    let clip = Rect::from_min_size(Offset::new(5.0, 5.0), 20.0, 20.0);
    list.push_clip(clip);
    list.rect(
        Rect::from_min_size(Offset::ZERO, 100.0, 100.0),
        CornerRadii::ZERO,
        fill(),
    );
    list.pop_clip();
    list.rect(
        Rect::from_min_size(Offset::ZERO, 1.0, 1.0),
        CornerRadii::ZERO,
        fill(),
    );

    let mut geometry = UiGeometry::new();
    geometry.expand(&mut list);
    assert_eq!(geometry.spans()[0].clip, Some(clip));
    assert_eq!(geometry.spans()[1].clip, None);
    assert_eq!(
        geometry.spans()[0].effect,
        None,
        "a rectangular clip is a scissor, not a layer"
    );
}

#[test]
fn a_rounded_clip_becomes_the_layer_that_cuts_the_span() {
    let mut list = DrawList::new();
    let clip = Rect::from_min_size(Offset::ZERO, 40.0, 20.0);
    let radii = CornerRadii::uniform(8.0);
    list.push_rounded_clip(clip, radii);
    // A canvas that paints the whole node box is the shape a square scissor
    // leaves behind: the rounding has to reach the program.
    list.rect(clip, CornerRadii::ZERO, fill());
    list.pop_clip();

    let mut geometry = UiGeometry::new();
    geometry.expand(&mut list);

    let span = geometry.spans()[0];
    assert_eq!(span.clip, Some(clip));
    assert_eq!(span.effect, Some(Effect::new(clip).shape(radii)));
}

#[test]
fn a_layer_painting_the_whole_clip_takes_its_corners() {
    let mut list = DrawList::new();
    let clip = Rect::from_min_size(Offset::ZERO, 40.0, 20.0);
    let radii = CornerRadii::uniform(8.0);
    let ramp = Gradient::horizontal(clip, Color::BLACK, Color::WHITE);
    list.push_rounded_clip(clip, radii);
    list.push_effect(Effect::new(clip).gradient(ramp, CornerRadii::ZERO));
    list.rect(clip, CornerRadii::ZERO, fill());
    list.pop_effect();
    list.pop_clip();

    let mut geometry = UiGeometry::new();
    geometry.expand(&mut list);

    // The ramp of a canvas covers the whole box it decorates, so the
    // corners of the clip are its own corners and nothing moves.
    let span = geometry.spans()[0];
    assert_eq!(span.effect, Some(Effect::new(clip).gradient(ramp, radii)));
}

#[test]
fn the_layer_of_an_operation_is_the_box_that_cuts_the_span() {
    let mut list = DrawList::new();
    let clip = Rect::from_min_size(Offset::ZERO, 40.0, 20.0);
    let bounds = Rect::from_min_size(Offset::new(10.0, 10.0), 20.0, 10.0);
    list.push_rounded_clip(clip, CornerRadii::uniform(8.0));
    list.push_effect(Effect::new(bounds).blur(4.0, CornerRadii::ZERO));
    list.rect(bounds, CornerRadii::ZERO, fill());
    list.pop_effect();
    list.pop_clip();

    let mut geometry = UiGeometry::new();
    geometry.expand(&mut list);

    // One span is cut by one box: the halo of the operation keeps the
    // program and its clip is honored as the rectangle it is inside.
    let span = geometry.spans()[0];
    assert_eq!(span.clip, Some(clip));
    assert!(
        span.effect
            .is_some_and(|effect| effect.kinds.contains(EffectKinds::BLUR)),
        "the layer of the operation survives the rounding of the clip"
    );
}

#[test]
fn an_effect_span_keeps_the_layer_and_widens_its_quad() {
    let bounds = Rect::from_min_size(Offset::new(10.0, 10.0), 40.0, 20.0);
    let effect = Effect::new(bounds).blur(6.0, CornerRadii::uniform(3.0));
    let mut list = DrawList::new();
    list.push_effect(effect);
    list.rect(bounds, CornerRadii::uniform(3.0), fill());
    list.pop_effect();

    let mut geometry = UiGeometry::new();
    geometry.expand(&mut list);

    let span = geometry.spans()[0];
    assert_eq!(span.effect, Some(effect));
    // Three sigmas of a six-pixel blur reach 18 pixels past every edge, and
    // the shader needs those fragments to draw the halo.
    assert_eq!(geometry.vertices()[0].position, [-8.0, -8.0]);
    assert_eq!(geometry.vertices()[1].position, [68.0, -8.0]);
}

#[test]
fn a_rounded_rectangle_is_cut_by_the_effect_program() {
    let bounds = Rect::from_min_size(Offset::new(10.0, 10.0), 40.0, 20.0);
    let radii = CornerRadii::uniform(8.0);
    let mut list = DrawList::new();
    list.rect(bounds, radii, fill());

    let mut geometry = UiGeometry::new();
    geometry.expand(&mut list);

    // The corners are a property of the rasterization, so the operation
    // is one quad and a layer that carries nothing but its rounding.
    let span = geometry.spans()[0];
    assert_eq!(span.vertex_count, 6);
    assert_eq!(span.effect, Some(Effect::new(bounds).shape(radii)));
    assert_eq!(geometry.vertices()[0].position, [10.0, 10.0]);
}

#[test]
fn a_rounded_outline_is_a_ring_instead_of_four_edges() {
    let bounds = Rect::from_min_size(Offset::ZERO, 30.0, 12.0);
    let radii = CornerRadii::uniform(6.0);
    let mut list = DrawList::new();
    list.rect(bounds, radii, Paint::stroke(Color::WHITE, 2.0));

    let mut geometry = UiGeometry::new();
    geometry.expand(&mut list);

    assert_eq!(geometry.spans().len(), 1);
    assert_eq!(geometry.spans()[0].vertex_count, 6, "one quad");
    assert_eq!(
        geometry.spans()[0].effect,
        Some(Effect::new(bounds).shape(radii).stroke(2.0))
    );
}

#[test]
fn an_outlined_rectangle_expands_to_four_edges() {
    let mut list = DrawList::new();
    list.rect(
        Rect::from_min_size(Offset::ZERO, 10.0, 10.0),
        CornerRadii::ZERO,
        Paint::stroke(Color::WHITE, 2.0),
    );
    let mut geometry = UiGeometry::new();
    geometry.expand(&mut list);

    assert_eq!(geometry.spans().len(), 1);
    assert_eq!(geometry.spans()[0].vertex_count, 24, "four quads");
}

#[test]
fn an_outline_wider_than_the_rectangle_is_clamped() {
    let mut list = DrawList::new();
    list.rect(
        Rect::from_min_size(Offset::ZERO, 4.0, 4.0),
        CornerRadii::ZERO,
        Paint::stroke(Color::WHITE, 100.0),
    );
    let mut geometry = UiGeometry::new();
    geometry.expand(&mut list);
    // The four edges still get emitted, none of them inverted.
    assert_eq!(geometry.spans()[0].vertex_count, 24);
    assert!(
        geometry
            .vertices()
            .iter()
            .all(|vertex| vertex.position[0] >= 0.0 && vertex.position[0] <= 4.0)
    );
}

#[test]
fn a_line_expands_to_a_quad_around_its_axis() {
    let mut list = DrawList::new();
    list.line(
        Offset::new(0.0, 0.0),
        Offset::new(10.0, 0.0),
        Paint::stroke(Color::WHITE, 2.0),
    );
    let mut geometry = UiGeometry::new();
    geometry.expand(&mut list);

    assert_eq!(geometry.spans()[0].vertex_count, 6);
    // A horizontal line of width 2 is offset by ±1 on the y axis.
    assert!(
        geometry
            .vertices()
            .iter()
            .all(|vertex| (vertex.position[1] - -1.0).abs() < 1e-6 || (vertex.position[1] - 1.0).abs() < 1e-6)
    );
}

#[test]
fn a_degenerate_line_draws_nothing() {
    let mut list = DrawList::new();
    list.line(Offset::ZERO, Offset::ZERO, Paint::stroke(Color::WHITE, 2.0));
    let mut geometry = UiGeometry::new();
    geometry.expand(&mut list);
    assert!(geometry.is_empty());
    assert_eq!(geometry.vertices().len(), 0);
}

#[test]
fn a_text_run_is_left_to_the_text_pass() {
    let mut list = DrawList::new();
    list.text(Offset::ZERO, Rc::from("hi"), TextStyle::default());
    let mut geometry = UiGeometry::new();
    geometry.expand(&mut list);
    assert!(geometry.is_empty());
    assert_eq!(geometry.vertices().len(), 0);
}

#[test]
fn expanding_clears_the_previous_frame() {
    let mut geometry = UiGeometry::new();
    let mut first = DrawList::new();
    first.rect(
        Rect::from_min_size(Offset::ZERO, 1.0, 1.0),
        CornerRadii::ZERO,
        fill(),
    );
    geometry.expand(&mut first);
    assert_eq!(geometry.vertices().len(), 6);

    let mut second = DrawList::new();
    second.line(
        Offset::ZERO,
        Offset::new(1.0, 0.0),
        Paint::stroke(Color::WHITE, 1.0),
    );
    geometry.expand(&mut second);
    assert_eq!(geometry.vertices().len(), 6);
    assert_eq!(geometry.spans().len(), 1);
}

#[test]
fn the_effect_block_keeps_the_shader_offsets() {
    // The same block `ui_effect.vert.glsl` declares and the Vulkan backend
    // pushes; the words the WebGPU pipe sends are cut from these offsets.
    assert_eq!(EFFECT_PUSH_BYTES, 108);
    assert_eq!(EFFECT_PUSH_WORDS, 27);
    assert_eq!(core::mem::offset_of!(EffectParams, color_from), 0);
    assert_eq!(core::mem::offset_of!(EffectParams, color_to), 16);
    assert_eq!(core::mem::offset_of!(EffectParams, viewport), 32);
    assert_eq!(core::mem::offset_of!(EffectParams, origin), 40);
    assert_eq!(core::mem::offset_of!(EffectParams, size), 48);
    assert_eq!(core::mem::offset_of!(EffectParams, axis_start), 56);
    assert_eq!(core::mem::offset_of!(EffectParams, axis_end), 64);
    assert_eq!(core::mem::offset_of!(EffectParams, blur_pixel), 72);
    assert_eq!(core::mem::offset_of!(EffectParams, radii), 80);
    assert_eq!(core::mem::offset_of!(EffectParams, rotation), 96);
    assert_eq!(core::mem::offset_of!(EffectParams, kinds), 100);
    assert_eq!(core::mem::offset_of!(EffectParams, stroke), 104);
}

#[test]
fn the_effect_words_are_the_field_bit_patterns() {
    let bounds = Rect::from_min_size(Offset::new(4.0, 6.0), 40.0, 20.0);
    let effect = Effect::new(bounds)
        .blur(6.0, CornerRadii::uniform(2.0))
        .rotate(0.5);
    let params = effect_params(&effect, [800.0, 600.0]);

    let words = params.words();
    assert_eq!(words.len(), EFFECT_PUSH_WORDS);
    // Every `vec4` is four words and every `vec2` two, in declaration order,
    // so the block the pipe sends starts where the shader's block starts.
    assert!(
        words[..8]
            .iter()
            .all(|word| *word == 0.0f32.to_bits()),
        "a layer without a ramp pushes two transparent colors"
    );
    assert_eq!(words[8], 800.0f32.to_bits(), "the viewport");
    assert_eq!(words[9], 600.0f32.to_bits());
    assert_eq!(words[10], 4.0f32.to_bits(), "the layer origin");
    assert_eq!(words[11], 6.0f32.to_bits());
    assert_eq!(words[18], 6.0f32.to_bits(), "the blur sigma");
    assert_eq!(words[19], 0.0f32.to_bits(), "no pixel grid");
    for (corner, radius) in [2.0f32.to_bits(); 4].iter().enumerate() {
        assert_eq!(words[20 + corner], *radius, "the four corner radii");
    }
    assert_eq!(words[24], 0.5f32.to_bits(), "the rotation");
    assert_eq!(words[25], 1 | 4, "the blur and rotate kind bits");
    assert_eq!(
        words[26],
        0.0f32.to_bits(),
        "a layer without an outline pushes no width"
    );
}

#[test]
fn the_kind_bits_are_the_ones_the_shader_tests() {
    // `ui_effect.frag.glsl` names these KIND_BLUR, KIND_PIXELATE,
    // KIND_ROTATE, KIND_GRADIENT and KIND_STROKE; the two sides meet in
    // the immediate block and nothing else checks them.
    assert_eq!(EffectKinds::BLUR.bits(), 1);
    assert_eq!(EffectKinds::PIXELATE.bits(), 2);
    assert_eq!(EffectKinds::ROTATE.bits(), 4);
    assert_eq!(EffectKinds::GRADIENT.bits(), 8);
    assert_eq!(EffectKinds::STROKE.bits(), 16);
}

#[test]
fn an_outline_sends_its_width_as_the_last_word() {
    let bounds = Rect::from_min_size(Offset::ZERO, 30.0, 12.0);
    let effect = Effect::new(bounds).stroke(2.5);

    let params = effect_params(&effect, [64.0, 64.0]);
    assert_eq!(params.stroke, 2.5);
    assert_eq!(params.kinds, 16);
    assert_eq!(params.words()[26], 2.5f32.to_bits());
}

#[test]
fn the_effect_parameters_carry_the_ramp_and_the_grid() {
    let bounds = Rect::from_min_size(Offset::new(4.0, 6.0), 40.0, 20.0);
    let gradient = Gradient::new(
        Offset::new(4.0, 16.0),
        Offset::new(44.0, 16.0),
        Color::RED,
        Color::BLUE,
    );
    let effect = Effect::new(bounds)
        .pixelate(5.0)
        .rotate(0.5)
        .gradient(gradient, CornerRadii::uniform(2.0));

    let params = effect_params(&effect, [800.0, 600.0]);
    assert_eq!(params.viewport, [800.0, 600.0]);
    assert_eq!(params.origin, [4.0, 6.0]);
    assert_eq!(params.size, [40.0, 20.0]);
    assert_eq!(params.axis_start, [4.0, 16.0]);
    assert_eq!(params.axis_end, [44.0, 16.0]);
    assert_eq!(params.color_from, channels(Color::RED));
    assert_eq!(params.color_to, channels(Color::BLUE));
    assert_eq!(params.blur_pixel, [0.0, 5.0]);
    assert_eq!(params.rotation, 0.5);
    assert_eq!(params.radii, [2.0, 2.0, 2.0, 2.0]);
    assert_eq!(params.kinds, 2 | 4 | 8);
}

#[test]
fn a_layer_without_a_ramp_collapses_its_axis() {
    let bounds = Rect::from_min_size(Offset::new(2.0, 2.0), 20.0, 10.0);
    let effect = Effect::new(bounds).blur(3.0, CornerRadii::ZERO);

    let params = effect_params(&effect, [100.0, 100.0]);
    let center = [12.0, 7.0];
    assert_eq!(params.axis_start, center);
    assert_eq!(params.axis_end, center);
    assert_eq!(params.blur_pixel, [3.0, 0.0]);
    assert_eq!(params.kinds, 1);
}

#[test]
fn the_scissor_of_an_unclipped_command_covers_the_target() {
    let scissor = scissor_for(None, 800, 600);
    assert_eq!(scissor.x, 0);
    assert_eq!(scissor.y, 0);
    assert_eq!(scissor.width, 800);
    assert_eq!(scissor.height, 600);
}

#[test]
fn the_scissor_of_a_clip_is_snapped_outward_and_clamped() {
    let clip = Rect::from_min_size(Offset::new(10.4, 20.6), 100.0, 50.0);
    let scissor = scissor_for(Some(clip), 800, 600);
    assert_eq!(scissor.x, 10);
    assert_eq!(scissor.y, 20);
    assert_eq!(scissor.width, 101);
    assert_eq!(scissor.height, 51);

    let outside = Rect::from_min_size(Offset::new(-50.0, -50.0), 20.0, 20.0);
    let clamped = scissor_for(Some(outside), 800, 600);
    assert_eq!(clamped.x, 0);
    assert_eq!(clamped.y, 0);
    assert_eq!(clamped.width, 0);
    assert_eq!(clamped.height, 0);
}

#[test]
fn a_recorded_frame_expands_and_drains() {
    let frame = UiFrame::new();
    frame.with(|list| {
        list.rect(
            Rect::from_min_size(Offset::ZERO, 4.0, 4.0),
            CornerRadii::ZERO,
            Paint::fill(Color::BLACK),
        );
    });
    assert!(!frame.is_empty());

    let mut geometry = UiGeometry::new();
    frame.with(|list| geometry.expand(list));
    assert_eq!(geometry.spans().len(), 1);
    assert!(frame.is_empty(), "the pipe consumed the queue");
}
