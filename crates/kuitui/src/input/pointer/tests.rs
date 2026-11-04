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

//! Behavior of the pointer state, the update, and the hit test.

use codevar_kuitui_control::{KeyState, Modifiers, Point, PointerButton, PointerEvent, Timestamp};

use crate::geometry::{Offset, Rect};
use crate::input::mouse::{BoundId, GeometryCache, VertexBounds};
use crate::input::{
    Input, KeyboardModifiers, MouseHandler, PointerButtons, PointerPhase, PointerState, PointerUpdate,
};

use super::MouseBinding;

fn when(micros: u64) -> Timestamp {
    Timestamp::from_micros(micros)
}

fn motion(position: Option<Point>, delta: Option<Point>) -> PointerEvent {
    PointerEvent::Motion { position, delta }
}

fn press(button: PointerButton, state: KeyState, modifiers: Modifiers) -> PointerEvent {
    PointerEvent::Button {
        button,
        state,
        position: None,
        modifiers,
    }
}

fn scroll(delta: Point) -> PointerEvent {
    PointerEvent::Scroll {
        delta,
        position: None,
        modifiers: Modifiers::NONE,
    }
}

#[test]
fn a_position_delta_follows_the_movement_between_reports() {
    let state = PointerState::new();
    state.apply(&motion(Some(Point::new(10.0, 5.0)), None), when(1));
    assert_eq!(state.position(), Some(Offset::new(10.0, 5.0)));
    assert_eq!(state.delta(), Offset::ZERO);

    let update = PointerUpdate::new(&state, &motion(Some(Point::new(14.0, 2.0)), None), when(2));
    assert_eq!(update.phase, PointerPhase::Motion);
    assert_eq!(update.delta, Offset::new(4.0, -3.0));
    assert_eq!(update.position, Some(Offset::new(14.0, 2.0)));
    assert_eq!(state.delta(), Offset::new(4.0, -3.0));
}

#[test]
fn an_explicit_delta_is_used_and_accumulated_without_a_position() {
    let state = PointerState::new();
    state.apply(&motion(None, Some(Point::new(3.0, 4.0))), when(1));
    state.apply(&motion(None, Some(Point::new(1.0, 1.0))), when(2));
    assert_eq!(state.position(), None);
    assert_eq!(state.delta(), Offset::new(4.0, 5.0));

    let snapshot = state.finish_frame();
    assert_eq!(snapshot.delta, Offset::new(4.0, 5.0));
    assert_eq!(state.delta(), Offset::ZERO);
}

#[test]
fn a_button_press_and_release_update_the_set_and_the_counters() {
    let state = PointerState::new();
    state.apply(
        &press(PointerButton::Primary, KeyState::Pressed, Modifiers::SHIFT),
        when(1),
    );
    assert!(state.is_pressed(PointerButton::Primary));
    assert_eq!(state.buttons().count(), 1);
    assert_eq!(state.last_pressed(), Some(PointerButton::Primary));
    assert_eq!(state.press_count(), 1);
    assert!(
        state
            .modifiers()
            .contains(KeyboardModifiers::SHIFT)
    );

    state.apply(
        &press(PointerButton::Primary, KeyState::Released, Modifiers::NONE),
        when(2),
    );
    assert!(!state.is_pressed(PointerButton::Primary));
    assert!(state.buttons().is_empty());
    assert_eq!(state.last_released(), Some(PointerButton::Primary));
    assert_eq!(state.release_count(), 1);
    assert!(state.modifiers().is_empty());
}

#[test]
fn a_update_reports_the_button_and_the_held_set() {
    let state = PointerState::new();
    let update = PointerUpdate::new(
        &state,
        &press(PointerButton::Secondary, KeyState::Pressed, Modifiers::CONTROL),
        when(1),
    );
    assert_eq!(update.phase, PointerPhase::Button);
    assert_eq!(update.button, Some(PointerButton::Secondary));
    assert!(update.buttons.contains(PointerButtons::SECONDARY));
    assert!(
        update
            .modifiers
            .contains(KeyboardModifiers::CONTROL)
    );
}

#[test]
fn a_cancel_releases_every_button_and_marks_the_pointer_outside() {
    let state = PointerState::new();
    state.apply(
        &PointerEvent::Enter {
            position: Some(Point::new(4.0, 4.0)),
        },
        when(1),
    );
    state.apply(
        &press(PointerButton::Primary, KeyState::Pressed, Modifiers::NONE),
        when(2),
    );
    state.apply(
        &press(PointerButton::Forward, KeyState::Pressed, Modifiers::NONE),
        when(3),
    );
    assert!(state.is_inside());
    assert_eq!(state.buttons().count(), 2);

    state.apply(&PointerEvent::Cancel, when(4));
    assert!(!state.is_inside());
    assert!(state.buttons().is_empty());
}

#[test]
fn other_buttons_get_their_own_bits_up_to_the_set_limit() {
    let first = PointerButtons::from_button(PointerButton::Other(0));
    let last = PointerButtons::from_button(PointerButton::Other(26));
    assert_eq!(first, Some(PointerButtons::from_bits_truncate(1 << 5)));
    assert!(last.is_some());
    assert_eq!(PointerButtons::from_button(PointerButton::Other(27)), None);

    let held = PointerButtons::PRIMARY | PointerButtons::SECONDARY;
    assert!(held.contains(PointerButtons::PRIMARY));
    assert!(!held.contains(PointerButtons::MIDDLE));
    assert_eq!(held.count(), 2);
}

#[test]
fn scroll_accumulates_for_the_frame_and_restarts_on_the_next_one() {
    let state = PointerState::new();
    state.apply(&scroll(Point::new(0.0, 3.0)), when(1));
    state.apply(&scroll(Point::new(0.0, -1.0)), when(2));
    assert_eq!(state.scroll(), Offset::new(0.0, 2.0));

    let snapshot = state.finish_frame();
    assert_eq!(snapshot.scroll, Offset::new(0.0, 2.0));
    assert_eq!(snapshot.sequence % 2, 0);
    assert_eq!(state.scroll(), Offset::ZERO);
}

#[test]
fn a_snapshot_carries_every_field_of_one_instant() {
    let state = PointerState::new();
    state.apply(
        &PointerEvent::Enter {
            position: Some(Point::new(9.0, 7.0)),
        },
        when(1),
    );
    state.apply(
        &press(PointerButton::Middle, KeyState::Pressed, Modifiers::ALT),
        when(2),
    );
    let snapshot = state.snapshot();
    assert!(snapshot.inside);
    assert_eq!(snapshot.position, Some(Offset::new(9.0, 7.0)));
    assert!(snapshot.is_pressed(PointerButton::Middle));
    assert!(
        snapshot
            .modifiers
            .contains(KeyboardModifiers::ALT)
    );
    assert_eq!(snapshot.last_pressed, Some(PointerButton::Middle));
    assert_eq!(snapshot.sequence, state.sequence());
}

static TARGET: PointerState = PointerState::new();

struct Target {
    bounds: Vec<VertexBounds>,
    area: Rect,
}

impl Input for Target {}

impl MouseHandler for Target {
    fn pointer(&self) -> &'static PointerState {
        &TARGET
    }

    fn geometry(&self) -> &[VertexBounds] {
        &self.bounds
    }

    fn interactive_area(&self) -> Rect {
        self.area
    }
}

#[test]
fn a_handler_hits_the_shape_under_the_pointer() {
    let area = Rect::from_min_size(Offset::ZERO, 80.0, 80.0);
    let target = Target {
        bounds: vec![VertexBounds::rect(
            BoundId::new(7),
            Rect::from_min_size(Offset::new(20.0, 20.0), 40.0, 40.0),
        )],
        area,
    };
    let mut cache = GeometryCache::new();

    TARGET.apply(&motion(Some(Point::new(30.0, 30.0)), None), when(1));
    let inside = target.hit(&mut cache);
    assert_eq!(inside.map(|hit| hit.distance), Some(0.0));
    assert_eq!(inside.map(|hit| hit.id), Some(BoundId::new(7)));

    TARGET.apply(&motion(Some(Point::new(70.0, 70.0)), None), when(2));
    let outside = target.hit(&mut cache);
    assert!(outside.is_some_and(|hit| hit.distance > 0.0));
}

static BOUND: PointerState = PointerState::new();

#[test]
fn a_binding_rebuilds_its_geometry_and_area() {
    let mut binding = MouseBinding::new(&BOUND, Rect::from_min_size(Offset::ZERO, 80.0, 80.0));
    binding.set_geometry(vec![VertexBounds::rect(
        BoundId::new(3),
        Rect::from_min_size(Offset::new(20.0, 20.0), 40.0, 40.0),
    )]);
    BOUND.apply(&motion(Some(Point::new(30.0, 30.0)), None), when(1));
    assert_eq!(binding.hit().map(|hit| hit.id), Some(BoundId::new(3)));

    binding.set_geometry(Vec::new());
    assert_eq!(binding.hit(), None);

    binding.set_geometry(vec![VertexBounds::rect(
        BoundId::new(4),
        Rect::from_min_size(Offset::new(20.0, 20.0), 40.0, 40.0),
    )]);
    binding.set_area(Rect::from_min_size(Offset::new(0.0, 0.0), 40.0, 40.0));
    BOUND.apply(&motion(Some(Point::new(30.0, 30.0)), None), when(2));
    assert_eq!(binding.hit().map(|hit| hit.id), Some(BoundId::new(4)));
}
