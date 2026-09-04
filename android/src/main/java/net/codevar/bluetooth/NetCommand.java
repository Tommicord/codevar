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

/**
 * Command interface for receiving input events from Bluetooth HID devices.
 * <p>
 * Implement this interface to receive keyboard and mouse events from
 * the NetBluetoothHandler. All methods are called from the handler's
 * background thread, so implementations should be thread-safe.
 * </p>
 */
public interface NetCommand {

    /**
     * Called when a key is pressed.
     *
     * @param keyCode   The Android key code (e.g., KeyEvent.KEYCODE_A).
     * @param modifiers Modifier flags (SHIFT, ALT, CTRL, etc.).
     */
    void onKeyPressed(int keyCode, int modifiers);

    /**
     * Called when a key is released.
     *
     * @param keyCode   The Android key code (e.g., KeyEvent.KEYCODE_A).
     * @param modifiers Modifier flags (SHIFT, ALT, CTRL, etc.).
     */
    void onKeyReleased(int keyCode, int modifiers);

    /**
     * Called when the mouse moves.
     *
     * @param x The X coordinate.
     * @param y The Y coordinate.
     */
    void onMouseMove(int x, int y);

    /**
     * Called when a mouse button is pressed.
     *
     * @param button The button identifier (1=left, 2=middle, 3=right, etc.).
     * @param x      The X coordinate at the time of press.
     * @param y      The Y coordinate at the time of press.
     */
    void onButtonPress(int button, int x, int y);

    /**
     * Called when a mouse button is released.
     *
     * @param button The button identifier (1=left, 2=middle, 3=right, etc.).
     * @param x      The X coordinate at the time of release.
     * @param y      The Y coordinate at the time of release.
     */
    void onButtonRelease(int button, int x, int y);

    /**
     * Called when the mouse wheel scrolls.
     *
     * @param delta The scroll delta (positive for scroll up, negative for scroll down).
     * @param x     The X coordinate at the time of scroll.
     * @param y     The Y coordinate at the time of scroll.
     */
    void onMouseScroll(int delta, int x, int y);
}
