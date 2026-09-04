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

import org.junit.Test;
import org.junit.Before;

import static org.junit.Assert.*;

/**
 * Unit tests for NetBluetoothHidReportParser.
 */
public class NetBluetoothHidReportParserTest {
    private NetBluetoothHidReportParser parser;

    @Before
    public void setUp() {
        parser = new NetBluetoothHidReportParser();
    }

    @Test
    public void testConstructor() {
        assertNotNull(parser);
    }

    @Test
    public void testReset() {
        parser.reset();
        assertNotNull(parser);
    }

    @Test
    public void testParseKeyboardReportKeyPress() {
        byte[] report = new byte[8];
        report[0] = 0x02; // Left shift modifier
        report[2] = 0x04; // 'a' key
        report[3] = 0x05; // 'b' key

        NetBluetoothInputEvent event = parser.parseReport(report, NetBluetoothHidDevice.HidDeviceType.KEYBOARD);
        assertNotNull(event);
        assertEquals(NetBluetoothInputEvent.InputType.KEYBOARD, event.getType());
        assertEquals(0x04, event.getKeyEvent().keyCode());
        assertEquals(NetBluetoothInputEvent.KeyEvent.KeyAction.PRESS, event.getKeyEvent().action());
        assertEquals(0x02, event.getKeyEvent().modifiers());
    }

    @Test
    public void testParseKeyboardReportKeyRelease() {
        byte[] report1 = new byte[8];
        report1[2] = 0x04; // 'a' key pressed

        parser.parseReport(report1, NetBluetoothHidDevice.HidDeviceType.KEYBOARD);

        byte[] report2 = new byte[8];
        report2[2] = 0x00; // Key released

        NetBluetoothInputEvent event = parser.parseReport(report2, NetBluetoothHidDevice.HidDeviceType.KEYBOARD);
        assertNotNull(event);
        assertEquals(NetBluetoothInputEvent.InputType.KEYBOARD, event.getType());
        assertEquals(0x04, event.getKeyEvent().keyCode());
        assertEquals(NetBluetoothInputEvent.KeyEvent.KeyAction.RELEASE, event.getKeyEvent().action());
    }

    @Test
    public void testParseKeyboardReportModifierPress() {
        byte[] report = new byte[8];
        report[0] = 0x02; // Left shift pressed

        NetBluetoothInputEvent event = parser.parseReport(report, NetBluetoothHidDevice.HidDeviceType.KEYBOARD);
        assertNotNull(event);
        assertEquals(NetBluetoothInputEvent.InputType.KEYBOARD, event.getType());
        assertEquals(0x02, event.getKeyEvent().keyCode());
        assertEquals(NetBluetoothInputEvent.KeyEvent.KeyAction.PRESS, event.getKeyEvent().action());
    }

    @Test
    public void testParseKeyboardReportModifierRelease() {
        byte[] report1 = new byte[8];
        report1[0] = 0x02; // Left shift pressed

        parser.parseReport(report1, NetBluetoothHidDevice.HidDeviceType.KEYBOARD);

        byte[] report2 = new byte[8];
        report2[0] = 0x00; // Shift released

        NetBluetoothInputEvent event = parser.parseReport(report2, NetBluetoothHidDevice.HidDeviceType.KEYBOARD);
        assertNotNull(event);
        assertEquals(NetBluetoothInputEvent.InputType.KEYBOARD, event.getType());
        assertEquals(0x02, event.getKeyEvent().keyCode());
        assertEquals(NetBluetoothInputEvent.KeyEvent.KeyAction.RELEASE, event.getKeyEvent().action());
    }

    @Test
    public void testParseKeyboardReportNoChange() {
        byte[] report = new byte[8];
        report[2] = 0x04; // 'a' key

        NetBluetoothInputEvent event1 = parser.parseReport(report, NetBluetoothHidDevice.HidDeviceType.KEYBOARD);
        assertNotNull(event1);

        NetBluetoothInputEvent event2 = parser.parseReport(report, NetBluetoothHidDevice.HidDeviceType.KEYBOARD);
        assertNull(event2); // No change, should return null
    }

    @Test
    public void testParseKeyboardReportInvalidLength() {
        byte[] report = new byte[4]; // Too short

        NetBluetoothInputEvent event = parser.parseReport(report, NetBluetoothHidDevice.HidDeviceType.KEYBOARD);
        assertNull(event);
    }

    @Test
    public void testParseKeyboardReportNull() {
        NetBluetoothInputEvent event = parser.parseReport(null, NetBluetoothHidDevice.HidDeviceType.KEYBOARD);
        assertNull(event);
    }

    @Test
    public void testParseKeyboardReportEmpty() {
        byte[] report = new byte[0];

        NetBluetoothInputEvent event = parser.parseReport(report, NetBluetoothHidDevice.HidDeviceType.KEYBOARD);
        assertNull(event);
    }

    @Test
    public void testParseMouseReportMove() {
        byte[] report = new byte[3];
        report[0] = 0x00; // No buttons
        report[1] = 0x05; // Move right 5
        report[2] = (byte) 0xFB; // Move up 5 (signed)

        NetBluetoothInputEvent event = parser.parseReport(report, NetBluetoothHidDevice.HidDeviceType.MOUSE);
        assertNotNull(event);
        assertEquals(NetBluetoothInputEvent.InputType.MOUSE, event.getType());
        assertEquals(5, event.getMouseEvent().x());
        assertEquals(-5, event.getMouseEvent().y());
        assertEquals(NetBluetoothInputEvent.MouseEvent.MouseEventType.MOVE, event.getMouseEvent().eventType());
    }

    @Test
    public void testParseMouseReportButtonPress() {
        byte[] report = new byte[3];
        report[0] = 0x01; // Left button pressed
        report[1] = 0x00;
        report[2] = 0x00;

        NetBluetoothInputEvent event = parser.parseReport(report, NetBluetoothHidDevice.HidDeviceType.MOUSE);
        assertNotNull(event);
        assertEquals(NetBluetoothInputEvent.InputType.MOUSE, event.getType());
        assertEquals(0x01, event.getMouseEvent().buttons());
        assertEquals(NetBluetoothInputEvent.MouseEvent.MouseEventType.BUTTON_PRESS, event.getMouseEvent().eventType());
    }

    @Test
    public void testParseMouseReportButtonRelease() {
        byte[] report1 = new byte[3];
        report1[0] = 0x01; // Left button pressed

        parser.parseReport(report1, NetBluetoothHidDevice.HidDeviceType.MOUSE);

        byte[] report2 = new byte[3];
        report2[0] = 0x00; // Button released

        NetBluetoothInputEvent event = parser.parseReport(report2, NetBluetoothHidDevice.HidDeviceType.MOUSE);
        assertNotNull(event);
        assertEquals(NetBluetoothInputEvent.InputType.MOUSE, event.getType());
        assertEquals(NetBluetoothInputEvent.MouseEvent.MouseEventType.BUTTON_RELEASE, event.getMouseEvent().eventType());
    }

    @Test
    public void testParseMouseReportRightButton() {
        byte[] report = new byte[3];
        report[0] = 0x02; // Right button pressed
        report[1] = 0x00;
        report[2]  = 0x00;

        NetBluetoothInputEvent event = parser.parseReport(report, NetBluetoothHidDevice.HidDeviceType.MOUSE);
        assertNotNull(event);
        assertEquals(0x02, event.getMouseEvent().buttons());
    }

    @Test
    public void testParseMouseReportMiddleButton() {
        byte[] report = new byte[3];
        report[0] = 0x04; // Middle button pressed
        report[1] = 0x00;
        report[2] = 0x00;

        NetBluetoothInputEvent event = parser.parseReport(report, NetBluetoothHidDevice.HidDeviceType.MOUSE);
        assertNotNull(event);
        assertEquals(0x04, event.getMouseEvent().buttons());
    }

    @Test
    public void testParseMouseReportNoChange() {
        byte[] report = new byte[3];
        report[0] = 0x01;
        report[1] = 0x00;
        report[2] = 0x00;

        NetBluetoothInputEvent event1 = parser.parseReport(report, NetBluetoothHidDevice.HidDeviceType.MOUSE);
        assertNotNull(event1);

        NetBluetoothInputEvent event2 = parser.parseReport(report, NetBluetoothHidDevice.HidDeviceType.MOUSE);
        assertNull(event2); // No change
    }

    @Test
    public void testParseMouseReportInvalidLength() {
        byte[] report = new byte[2]; // Too short

        NetBluetoothInputEvent event = parser.parseReport(report, NetBluetoothHidDevice.HidDeviceType.MOUSE);
        assertNull(event);
    }

    @Test
    public void testParseMouseReportNull() {
        NetBluetoothInputEvent event = parser.parseReport(null, NetBluetoothHidDevice.HidDeviceType.MOUSE);
        assertNull(event);
    }

    @Test
    public void testParseComboDeviceKeyboard() {
        byte[] report = new byte[8];
        report[2] = 0x04; // 'a' key

        NetBluetoothInputEvent event = parser.parseReport(report, NetBluetoothHidDevice.HidDeviceType.COMBO);
        assertNotNull(event);
        assertEquals(NetBluetoothInputEvent.InputType.KEYBOARD, event.getType());
    }

    @Test
    public void testParseComboDeviceMouse() {
        byte[] report = new byte[3];
        report[0] = 0x01;
        report[1] = 0x05;
        report[2] = 0x00;

        NetBluetoothInputEvent event = parser.parseReport(report, NetBluetoothHidDevice.HidDeviceType.COMBO);
        assertNotNull(event);
        assertEquals(NetBluetoothInputEvent.InputType.MOUSE, event.getType());
    }

    @Test
    public void testParseUnknownDeviceType() {
        byte[] report = new byte[8];

        NetBluetoothInputEvent event = parser.parseReport(report, NetBluetoothHidDevice.HidDeviceType.UNKNOWN);
        assertNull(event);
    }

    @Test
    public void testMultipleKeyChanges() {
        byte[] report1 = new byte[8];
        report1[2] = 0x04; // 'a' key

        NetBluetoothInputEvent event1 = parser.parseReport(report1, NetBluetoothHidDevice.HidDeviceType.KEYBOARD);
        assertNotNull(event1);
        assertEquals(0x04, event1.getKeyEvent().keyCode());

        byte[] report2 = new byte[8];
        report2[2] = 0x05; // 'b' key

        NetBluetoothInputEvent event2 = parser.parseReport(report2, NetBluetoothHidDevice.HidDeviceType.KEYBOARD);
        assertNotNull(event2);
        assertEquals(0x05, event2.getKeyEvent().keyCode());
    }

    @Test
    public void testSignedMouseMovement() {
        byte[] report = new byte[3];
        report[0] = 0x00;
        report[1] = (byte) 0xFF; // Move left 1
        report[2] = (byte) 0xFF; // Move up 1

        NetBluetoothInputEvent event = parser.parseReport(report, NetBluetoothHidDevice.HidDeviceType.MOUSE);
        assertNotNull(event);
        assertEquals(-1, event.getMouseEvent().x());
        assertEquals(-1, event.getMouseEvent().y());
    }

    @Test
    public void testLargeMouseMovement() {
        byte[] report = new byte[3];
        report[0] = 0x00;
        report[1] = 0x50; // Move right 80
        report[2] = 0x30; // Move down 48

        NetBluetoothInputEvent event = parser.parseReport(report, NetBluetoothHidDevice.HidDeviceType.MOUSE);
        assertNotNull(event);
        assertEquals(80, event.getMouseEvent().x());
        assertEquals(48, event.getMouseEvent().y());
    }
}
