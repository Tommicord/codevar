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

//! Tests for the keyboard state and its packed key encoding.

use codevar_kuitui_control::{DeviceId, KeyCode, KeyEvent, KeyState, Modifiers, Timestamp};

use super::keys::{KEY_SLOTS, code_at, decode_key, encode_key, slot};
use super::modifiers::{KeyboardModifiers, modifier_of};
use super::state::KeyboardState;

fn press(state: &KeyboardState, code: KeyCode, modifiers: Modifiers) {
    state.apply(
        &KeyEvent {
            device: DeviceId::new(1),
            code,
            state: KeyState::Pressed,
            modifiers,
        },
        Timestamp::from_micros(10),
    );
}

fn release(state: &KeyboardState, code: KeyCode, modifiers: Modifiers) {
    state.apply(
        &KeyEvent {
            device: DeviceId::new(1),
            code,
            state: KeyState::Released,
            modifiers,
        },
        Timestamp::from_micros(20),
    );
}

#[test]
fn the_slot_table_is_sorted_and_round_trips_every_normalized_key() {
    assert!(
        KEY_SLOTS.windows(2).all(|pair| pair[0] < pair[1]),
        "the slot table must stay sorted for the binary search"
    );
    for (index, code) in KEY_SLOTS.iter().enumerate() {
        let slot = slot(*code);
        assert_eq!(slot, u8::try_from(index).ok());
        if let Some(slot) = slot {
            assert_eq!(code_at(slot), *code);
            assert_eq!(decode_key(encode_key(*code)), Some(*code));
        }
    }
}

#[test]
fn a_press_records_the_key_the_modifier_and_the_held_set() {
    let state = KeyboardState::new();
    press(&state, KeyCode::KeyA, Modifiers::NONE);

    assert_eq!(state.last_pressed(), Some(KeyCode::KeyA));
    assert!(state.is_pressed(KeyCode::KeyA));
    assert_eq!(state.held_count(), 1);
    assert_eq!(state.press_count(), 1);
    assert_eq!(state.release_count(), 0);
    assert_eq!(state.device(), Some(DeviceId::new(1)));
    assert_eq!(state.sequence() % 2, 0);
}

#[test]
fn a_release_clears_the_held_key_but_remembers_it() {
    let state = KeyboardState::new();
    press(&state, KeyCode::KeyA, Modifiers::NONE);
    release(&state, KeyCode::KeyA, Modifiers::NONE);

    assert_eq!(state.last_pressed(), Some(KeyCode::KeyA));
    assert_eq!(state.last_released(), Some(KeyCode::KeyA));
    assert!(!state.is_pressed(KeyCode::KeyA));
    assert_eq!(state.held_count(), 0);
    assert_eq!(state.release_count(), 1);
}

#[test]
fn releasing_one_side_of_a_modifier_keeps_the_flag_while_the_other_is_held() {
    let state = KeyboardState::new();
    press(&state, KeyCode::ShiftLeft, Modifiers::SHIFT);
    press(&state, KeyCode::ShiftRight, Modifiers::SHIFT);
    assert!(
        state
            .modifiers()
            .contains(KeyboardModifiers::SHIFT)
    );

    release(&state, KeyCode::ShiftLeft, Modifiers::SHIFT);
    assert!(
        state
            .modifiers()
            .contains(KeyboardModifiers::SHIFT),
        "the right shift is still down"
    );

    release(&state, KeyCode::ShiftRight, Modifiers::NONE);
    assert!(
        !state
            .modifiers()
            .contains(KeyboardModifiers::SHIFT)
    );
}

#[test]
fn locks_toggle_on_press_and_survive_the_release() {
    let state = KeyboardState::new();
    press(&state, KeyCode::CapsLock, Modifiers::NONE);
    release(&state, KeyCode::CapsLock, Modifiers::CAPS_LOCK);
    assert!(
        state
            .modifiers()
            .contains(KeyboardModifiers::CAPS_LOCK)
    );

    press(&state, KeyCode::CapsLock, Modifiers::CAPS_LOCK);
    assert!(
        !state
            .modifiers()
            .contains(KeyboardModifiers::CAPS_LOCK)
    );
}

#[test]
fn a_reported_modifier_is_adopted_when_the_press_arrives() {
    let state = KeyboardState::new();
    press(&state, KeyCode::KeyC, Modifiers::CONTROL);
    assert!(
        state
            .modifiers()
            .contains(KeyboardModifiers::CONTROL)
    );
}

#[test]
fn native_keys_are_tracked_in_their_own_set() {
    let state = KeyboardState::new();
    press(&state, KeyCode::Native(300), Modifiers::NONE);
    assert_eq!(state.last_pressed(), Some(KeyCode::Native(300)));
    assert!(state.is_pressed(KeyCode::Native(300)));
    assert_eq!(state.held_count(), 1);

    release(&state, KeyCode::Native(300), Modifiers::NONE);
    assert!(!state.is_pressed(KeyCode::Native(300)));
    assert_eq!(state.last_released(), Some(KeyCode::Native(300)));
    assert_eq!(state.held_count(), 0);
}

#[test]
fn a_snapshot_reports_the_same_state_as_the_accessors() {
    let state = KeyboardState::new();
    press(&state, KeyCode::ControlLeft, Modifiers::CONTROL);
    press(&state, KeyCode::KeyV, Modifiers::CONTROL);

    let snapshot = state.snapshot();
    assert!(snapshot.is_pressed(KeyCode::KeyV));
    assert!(snapshot.is_pressed(KeyCode::ControlLeft));
    assert_eq!(snapshot.last_pressed, Some(KeyCode::KeyV));
    assert_eq!(snapshot.held_count(), 2);
    assert!(snapshot.is_shortcut(KeyCode::KeyV, KeyboardModifiers::CONTROL));
    assert!(!snapshot.is_shortcut(KeyCode::KeyV, KeyboardModifiers::ALT));
    assert_eq!(snapshot.device, Some(DeviceId::new(1)));
}

#[test]
fn repeat_events_do_not_change_the_held_set() {
    let state = KeyboardState::new();
    press(&state, KeyCode::KeyA, Modifiers::NONE);
    state.apply(
        &KeyEvent {
            device: DeviceId::new(1),
            code: KeyCode::KeyA,
            state: KeyState::Repeat,
            modifiers: Modifiers::NONE,
        },
        Timestamp::from_micros(30),
    );

    assert_eq!(state.last_repeat(), Some(KeyCode::KeyA));
    assert!(state.is_pressed(KeyCode::KeyA));
    assert_eq!(state.held_count(), 1);
}

#[test]
fn only_modifier_keys_are_recorded_as_the_last_modifier() {
    let state = KeyboardState::new();
    press(&state, KeyCode::KeyA, Modifiers::NONE);
    assert_eq!(state.last_modifier(), None);

    press(&state, KeyCode::AltRight, Modifiers::ALT);
    assert_eq!(state.last_modifier(), Some(KeyCode::AltRight));
    assert_eq!(modifier_of(KeyCode::AltRight), Some(KeyboardModifiers::ALT));
}
