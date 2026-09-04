package net.codevar.bluetooth;

// Copyright 2026 Codevar
// Licensed under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the
// License at
//
//   https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in
// writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
// OR CONDITIONS OF ANY KIND, either express or implied. See
// the License for the specific language governing
// permissions and limitations under the License.

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

/**
 * Represents an input event from a Bluetooth HID device.
 *
 * <p>This class encapsulates keyboard and mouse input events, including key presses,
 * key releases, mouse movements, button clicks, and scroll events.</p>
 *
 * <p><b>Usage Example:</b></p>
 * <pre>{@code
 * if (event.getType() == InputType.KEYBOARD) {
 *     KeyEvent keyEvent = event.getKeyEvent();
 *     Log.d("Input", "Key: " + keyEvent.getKeyCode() + " Action: " + keyEvent.getAction());
 * } else if (event.getType() == InputType.MOUSE) {
 *     MouseEvent mouseEvent = event.getMouseEvent();
 *     Log.d("Input", "Mouse: x=" + mouseEvent.getX() + " y=" + mouseEvent.getY());
 * }
 * }</pre>
 */
public class NetBluetoothInputEvent {
    private final InputType type;
    private final KeyEvent keyEvent;
    private final MouseEvent mouseEvent;
    private final long timestamp;

    /**
     * Constructs a new keyboard input event.
     *
     * @param keyEvent The keyboard event data. Must not be null.
     */
    public NetBluetoothInputEvent(@NonNull KeyEvent keyEvent) {
        this.type = InputType.KEYBOARD;
        this.keyEvent = keyEvent;
        this.mouseEvent = null;
        this.timestamp = System.currentTimeMillis();
    }

    /**
     * Constructs a new mouse input event.
     *
     * @param mouseEvent The mouse event data. Must not be null.
     */
    public NetBluetoothInputEvent(@NonNull MouseEvent mouseEvent) {
        this.type = InputType.MOUSE;
        this.keyEvent = null;
        this.mouseEvent = mouseEvent;
        this.timestamp = System.currentTimeMillis();
    }

    /**
     * Returns the input event type.
     *
     * @return The event type.
     */
    @NonNull
    public InputType getType() {
        return type;
    }

    /**
     * Returns the keyboard event data.
     *
     * @return The keyboard event, or null if this is not a keyboard event.
     */
    @Nullable
    public KeyEvent getKeyEvent() {
        return keyEvent;
    }

    /**
     * Returns the mouse event data.
     *
     * @return The mouse event, or null if this is not a mouse event.
     */
    @Nullable
    public MouseEvent getMouseEvent() {
        return mouseEvent;
    }

    /**
     * Returns the timestamp when this event was created.
     *
     * @return The timestamp in milliseconds.
     */
    public long getTimestamp() {
        return timestamp;
    }

    /**
     * Returns a string representation of this event.
     *
     * @return A string describing the event.
     */
    @Override
    public String toString() {
        if (type == InputType.KEYBOARD && keyEvent != null) {
            return "InputEvent{type=KEYBOARD, keyCode=" + keyEvent.keyCode() +
                    ", action=" + keyEvent.action() + ", modifiers=" + keyEvent.modifiers() + "}";
        } else if (type == InputType.MOUSE && mouseEvent != null) {
            return "InputEvent{type=MOUSE, x=" + mouseEvent.x() +
                    ", y=" + mouseEvent.y() + ", buttons=" + mouseEvent.buttons() +
                    ", eventType=" + mouseEvent.eventType() + "}";
        }
        return "InputEvent{type=" + type + "}";
    }

    /**
     * Enumeration of input event types.
     */
    public enum InputType {
        /**
         * Keyboard input event
         */
        KEYBOARD,
        /**
         * Mouse input event
         */
        MOUSE
    }

    /**
         * Represents a keyboard input event.
         */
        public record KeyEvent(int keyCode, KeyAction action, int modifiers) {
        /**
         * Constructs a new KeyEvent.
         *
         * @param keyCode   The key code (e.g., Android KeyEvent.KEYCODE_A).
         * @param action    The key action (PRESS or RELEASE).
         * @param modifiers Modifier flags (SHIFT, ALT, CTRL, etc.).
         */
        public KeyEvent {
        }

            /**
             * Returns the key code.
             *
             * @return The key code.
             */
            @Override
            public int keyCode() {
                return keyCode;
            }

            /**
             * Returns the key action.
             *
             * @return The key action.
             */
            @Override
            public KeyAction action() {
                return action;
            }

            /**
             * Returns the modifier flags.
             *
             * @return The modifier flags.
             */
            @Override
            public int modifiers() {
                return modifiers;
            }

            /**
             * Enumeration of key actions.
             */
            public enum KeyAction {
                /**
                 * Key was pressed
                 */
                PRESS,
                /**
                 * Key was released
                 */
                RELEASE
            }
        }

    /**
         * Represents a mouse input event.
         */
        public record MouseEvent(int x, int y, int buttons, int scrollDelta, MouseEventType eventType) {
        /**
         * Constructs a new MouseEvent.
         *
         * @param x           The X coordinate.
         * @param y           The Y coordinate.
         * @param buttons     Button state (bitmask of button states).
         * @param scrollDelta Scroll delta (for scroll events).
         * @param eventType   The mouse event type.
         */
        public MouseEvent {
        }

            /**
             * Returns the X coordinate.
             *
             * @return The X coordinate.
             */
            @Override
            public int x() {
                return x;
            }

            /**
             * Returns the Y coordinate.
             *
             * @return The Y coordinate.
             */
            @Override
            public int y() {
                return y;
            }

            /**
             * Returns the button state bitmask.
             *
             * @return The button state.
             */
            @Override
            public int buttons() {
                return buttons;
            }

            /**
             * Returns the scroll delta.
             *
             * @return The scroll delta.
             */
            @Override
            public int scrollDelta() {
                return scrollDelta;
            }

            /**
             * Returns the mouse event type.
             *
             * @return The event type.
             */
            @Override
            public MouseEventType eventType() {
                return eventType;
            }

            /**
             * Enumeration of mouse event types.
             */
            public enum MouseEventType {
                /**
                 * Mouse moved
                 */
                MOVE,
                /**
                 * Mouse button pressed
                 */
                BUTTON_PRESS,
                /**
                 * Mouse button released
                 */
                BUTTON_RELEASE,
                /**
                 * Mouse wheel scrolled
                 */
                SCROLL
            }
        }
}
