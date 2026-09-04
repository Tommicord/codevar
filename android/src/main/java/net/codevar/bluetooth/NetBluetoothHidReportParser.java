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

import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

/**
 * Parser for Bluetooth HID reports from keyboards and mice.
 *
 * <p>This class parses raw HID report data into {@link NetBluetoothInputEvent} objects.
 * It supports standard HID boot protocol for keyboards and mice.</p>
 *
 * <p><b>Keyboard Report Format (Boot Protocol):</b>
 * <pre>
 * Byte 0: Modifier keys (bitmask)
 * Byte 1: Reserved
 * Byte 2-6: Key codes (up to 6 simultaneous keys)
 * </pre></p>
 *
 * <p><b>Mouse Report Format (Boot Protocol):</b>
 * <pre>
 * Byte 0: Buttons (bitmask)
 * Byte 1: X displacement (signed)
 * Byte 2: Y displacement (signed)
 * </pre></p>
 *
 * <p><b>Usage Example:</b></p>
 * <pre>{@code
 * NetBluetoothHidReportParser parser = new NetBluetoothHidReportParser();
 * byte[] report = device.readReport();
 * NetBluetoothInputEvent event = parser.parseReport(report, HidDeviceType.KEYBOARD);
 * if (event != null) {
 *     handleEvent(event);
 * }
 * }</pre>
 */
public class NetBluetoothHidReportParser {
    private static final String TAG = "Net:HidParser";

    // Keyboard modifier bitmasks
    private static final byte MODIFIER_LEFT_CTRL = (byte) 0x01;
    private static final byte MODIFIER_LEFT_SHIFT = (byte) 0x02;
    private static final byte MODIFIER_LEFT_ALT = (byte) 0x04;
    private static final byte MODIFIER_LEFT_META = (byte) 0x08;
    private static final byte MODIFIER_RIGHT_CTRL = (byte) 0x10;
    private static final byte MODIFIER_RIGHT_SHIFT = (byte) 0x20;
    private static final byte MODIFIER_RIGHT_ALT = (byte) 0x40;
    private static final byte MODIFIER_RIGHT_META = (byte) 0x80;

    // Mouse button bitmasks
    private static final byte BUTTON_LEFT = (byte) 0x01;
    private static final byte BUTTON_RIGHT = (byte) 0x02;
    private static final byte BUTTON_MIDDLE = (byte) 0x04;

    // Previous keyboard state for tracking key releases
    private byte[] previousKeyboardReport = new byte[8];
    private byte previousMouseButtons = 0;
    private int previousMouseX = 0;
    private int previousMouseY = 0;

    /**
     * Parses a HID report and returns the corresponding input event.
     *
     * @param report     The raw HID report data. Must not be null.
     * @param deviceType The type of HID device. Must not be null.
     * @return The parsed input event, or null if no event could be parsed.
     */
    @Nullable
    public NetBluetoothInputEvent parseReport(@NonNull byte[] report,
                                              @NonNull NetBluetoothHidDevice.HidDeviceType deviceType) {
        if (report == null || report.length == 0) {
            return null;
        }

        switch (deviceType) {
            case KEYBOARD:
                return parseKeyboardReport(report);
            case MOUSE:
                return parseMouseReport(report);
            case COMBO:
                // Try to parse as keyboard first, then mouse
                NetBluetoothInputEvent event = parseKeyboardReport(report);
                if (event == null) {
                    event = parseMouseReport(report);
                }
                return event;
            case UNKNOWN:
                Log.w(TAG, "Cannot parse report for unknown device type");
                return null;
            default:
                return null;
        }
    }

    /**
     * Parses a keyboard HID report.
     *
     * @param report The keyboard report data. Must not be null.
     * @return The keyboard input event, or null if no key change detected.
     */
    @Nullable
    private NetBluetoothInputEvent parseKeyboardReport(@NonNull byte[] report) {
        if (report.length < 8) {
            Log.w(TAG, "Invalid keyboard report length: " + report.length);
            return null;
        }

        byte modifiers = report[0];
        byte[] keyCodes = new byte[6];
        System.arraycopy(report, 2, keyCodes, 0, 6);

        // Check for key changes
        for (int i = 0; i < 6; i++) {
            byte currentKey = keyCodes[i];
            byte previousKey = previousKeyboardReport[i + 2];

            if (currentKey != previousKey) {
                if (currentKey != 0) {
                    // Key pressed
                    NetBluetoothInputEvent.KeyEvent keyEvent = new NetBluetoothInputEvent.KeyEvent(
                            currentKey & 0xFF,
                            NetBluetoothInputEvent.KeyEvent.KeyAction.PRESS,
                            modifiers & 0xFF
                    );
                    System.arraycopy(report, 0, previousKeyboardReport, 0, Math.min(report.length, previousKeyboardReport.length));
                    return new NetBluetoothInputEvent(keyEvent);
                } else if (previousKey != 0) {
                    // Key released
                    NetBluetoothInputEvent.KeyEvent keyEvent = new NetBluetoothInputEvent.KeyEvent(
                            previousKey & 0xFF,
                            NetBluetoothInputEvent.KeyEvent.KeyAction.RELEASE,
                            modifiers & 0xFF
                    );
                    System.arraycopy(report, 0, previousKeyboardReport, 0, Math.min(report.length, previousKeyboardReport.length));
                    return new NetBluetoothInputEvent(keyEvent);
                }
            }
        }

        // Check for modifier changes
        if (modifiers != previousKeyboardReport[0]) {
            // Find which modifier changed
            byte changed = (byte) (modifiers ^ previousKeyboardReport[0]);
            int modifierKeyCode = getModifierKeyCode(changed);
            if (modifierKeyCode != 0) {
                boolean pressed = (modifiers & changed) != 0;
                NetBluetoothInputEvent.KeyEvent keyEvent = new NetBluetoothInputEvent.KeyEvent(
                        modifierKeyCode,
                        pressed ? NetBluetoothInputEvent.KeyEvent.KeyAction.PRESS : NetBluetoothInputEvent.KeyEvent.KeyAction.RELEASE,
                        modifiers & 0xFF
                );
                System.arraycopy(report, 0, previousKeyboardReport, 0, Math.min(report.length, previousKeyboardReport.length));
                return new NetBluetoothInputEvent(keyEvent);
            }
        }

        // Update previous state
        System.arraycopy(report, 0, previousKeyboardReport, 0, Math.min(report.length, previousKeyboardReport.length));
        return null;
    }

    /**
     * Parses a mouse HID report.
     *
     * @param report The mouse report data. Must not be null.
     * @return The mouse input event, or null if no change detected.
     */
    @Nullable
    private NetBluetoothInputEvent parseMouseReport(@NonNull byte[] report) {
        if (report.length < 3) {
            Log.w(TAG, "Invalid mouse report length: " + report.length);
            return null;
        }

        byte buttons = report[0];
        int x = report[1];
        int y = report[2];

        // Convert signed bytes to integers
        if (x > 127) x -= 256;
        if (y > 127) y -= 256;

        // Check for button changes
        if (buttons != previousMouseButtons) {
            byte changed = (byte) (buttons ^ previousMouseButtons);
            int buttonIndex = getButtonIndex(changed);
            if (buttonIndex != -1) {
                boolean pressed = (buttons & changed) != 0;
                NetBluetoothInputEvent.MouseEvent mouseEvent = new NetBluetoothInputEvent.MouseEvent(
                        0, 0,
                        buttons & 0xFF,
                        0,
                        pressed ? NetBluetoothInputEvent.MouseEvent.MouseEventType.BUTTON_PRESS :
                                NetBluetoothInputEvent.MouseEvent.MouseEventType.BUTTON_RELEASE
                );
                previousMouseButtons = buttons;
                return new NetBluetoothInputEvent(mouseEvent);
            }
        }

        // Check for movement
        if (x != 0 || y != 0) {
            NetBluetoothInputEvent.MouseEvent mouseEvent = new NetBluetoothInputEvent.MouseEvent(
                    x, y,
                    buttons & 0xFF,
                    0,
                    NetBluetoothInputEvent.MouseEvent.MouseEventType.MOVE
            );
            previousMouseButtons = buttons;
            return new NetBluetoothInputEvent(mouseEvent);
        }

        previousMouseButtons = buttons;
        return null;
    }

    /**
     * Converts a modifier bitmask to a key code.
     *
     * @param modifier The modifier bitmask.
     * @return The corresponding key code, or 0 if unknown.
     */
    private int getModifierKeyCode(byte modifier) {
        if ((modifier & MODIFIER_LEFT_CTRL) != 0) return 0x01; // Left Control
        if ((modifier & MODIFIER_LEFT_SHIFT) != 0) return 0x02; // Left Shift
        if ((modifier & MODIFIER_LEFT_ALT) != 0) return 0x04; // Left Alt
        if ((modifier & MODIFIER_LEFT_META) != 0) return 0x08; // Left Meta
        if ((modifier & MODIFIER_RIGHT_CTRL) != 0) return 0x10; // Right Control
        if ((modifier & MODIFIER_RIGHT_SHIFT) != 0) return 0x20; // Right Shift
        if ((modifier & MODIFIER_RIGHT_ALT) != 0) return 0x40; // Right Alt
        if ((modifier & MODIFIER_RIGHT_META) != 0) return 0x80; // Right Meta
        return 0;
    }

    /**
     * Converts a button bitmask to a button index.
     *
     * @param button The button bitmask.
     * @return The button index (0=left, 1=right, 2=middle), or -1 if unknown.
     */
    private int getButtonIndex(byte button) {
        if ((button & BUTTON_LEFT) != 0) return 0;
        if ((button & BUTTON_RIGHT) != 0) return 1;
        if ((button & BUTTON_MIDDLE) != 0) return 2;
        return -1;
    }

    /**
     * Resets the parser state.
     *
     * <p>Call this method when connecting to a new device to clear previous state.</p>
     */
    public void reset() {
        previousKeyboardReport = new byte[8];
        previousMouseButtons = 0;
        previousMouseX = 0;
        previousMouseY = 0;
    }
}
