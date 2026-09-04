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

import static org.junit.Assert.*;

/**
 * Unit tests for NetBluetoothInputEvent.
 */
public class NetBluetoothInputEventTest {

    @Test
    public void testKeyboardEventConstructor() {
        NetBluetoothInputEvent.KeyEvent keyEvent = new NetBluetoothInputEvent.KeyEvent(
                0x04,
                NetBluetoothInputEvent.KeyEvent.KeyAction.PRESS,
                0x02
        );
        assertEquals(0x04, keyEvent.keyCode());
        assertEquals(NetBluetoothInputEvent.KeyEvent.KeyAction.PRESS, keyEvent.action());
        assertEquals(0x02, keyEvent.modifiers());
    }

    @Test
    public void testKeyboardEventRelease() {
        NetBluetoothInputEvent.KeyEvent keyEvent = new NetBluetoothInputEvent.KeyEvent(
                0x04,
                NetBluetoothInputEvent.KeyEvent.KeyAction.RELEASE,
                0x02
        );
        assertEquals(NetBluetoothInputEvent.KeyEvent.KeyAction.RELEASE, keyEvent.action());
    }

    @Test
    public void testKeyboardEventNoModifiers() {
        NetBluetoothInputEvent.KeyEvent keyEvent = new NetBluetoothInputEvent.KeyEvent(
                0x04,
                NetBluetoothInputEvent.KeyEvent.KeyAction.PRESS,
                0x00
        );
        assertEquals(0x00, keyEvent.modifiers());
    }

    @Test
    public void testMouseEventConstructor() {
        NetBluetoothInputEvent.MouseEvent mouseEvent = new NetBluetoothInputEvent.MouseEvent(
                100, 200, 0x01, 0, NetBluetoothInputEvent.MouseEvent.MouseEventType.MOVE
        );
        assertEquals(100, mouseEvent.x());
        assertEquals(200, mouseEvent.y());
        assertEquals(0x01, mouseEvent.buttons());
        assertEquals(0, mouseEvent.scrollDelta());
        assertEquals(NetBluetoothInputEvent.MouseEvent.MouseEventType.MOVE, mouseEvent.eventType());
    }

    @Test
    public void testMouseEventButtonPress() {
        NetBluetoothInputEvent.MouseEvent mouseEvent = new NetBluetoothInputEvent.MouseEvent(
                0, 0, 0x01, 0, NetBluetoothInputEvent.MouseEvent.MouseEventType.BUTTON_PRESS
        );
        assertEquals(NetBluetoothInputEvent.MouseEvent.MouseEventType.BUTTON_PRESS, mouseEvent.eventType());
    }

    @Test
    public void testMouseEventButtonRelease() {
        NetBluetoothInputEvent.MouseEvent mouseEvent = new NetBluetoothInputEvent.MouseEvent(
                0, 0, 0x00, 0, NetBluetoothInputEvent.MouseEvent.MouseEventType.BUTTON_RELEASE
        );
        assertEquals(NetBluetoothInputEvent.MouseEvent.MouseEventType.BUTTON_RELEASE, mouseEvent.eventType());
    }

    @Test
    public void testMouseEventScroll() {
        NetBluetoothInputEvent.MouseEvent mouseEvent = new NetBluetoothInputEvent.MouseEvent(
                0, 0, 0x00, 5, NetBluetoothInputEvent.MouseEvent.MouseEventType.SCROLL
        );
        assertEquals(5, mouseEvent.scrollDelta());
        assertEquals(NetBluetoothInputEvent.MouseEvent.MouseEventType.SCROLL, mouseEvent.eventType());
    }

    @Test
    public void testInputEventKeyboardConstructor() {
        NetBluetoothInputEvent.KeyEvent keyEvent = new NetBluetoothInputEvent.KeyEvent(
                0x04,
                NetBluetoothInputEvent.KeyEvent.KeyAction.PRESS,
                0x02
        );
        NetBluetoothInputEvent event = new NetBluetoothInputEvent(keyEvent);
        assertEquals(NetBluetoothInputEvent.InputType.KEYBOARD, event.getType());
        assertEquals(keyEvent, event.getKeyEvent());
        assertNull(event.getMouseEvent());
        assertTrue(event.getTimestamp() > 0);
    }

    @Test
    public void testInputEventMouseConstructor() {
        NetBluetoothInputEvent.MouseEvent mouseEvent = new NetBluetoothInputEvent.MouseEvent(
                100, 200, 0x01, 0, NetBluetoothInputEvent.MouseEvent.MouseEventType.MOVE
        );
        NetBluetoothInputEvent event = new NetBluetoothInputEvent(mouseEvent);
        assertEquals(NetBluetoothInputEvent.InputType.MOUSE, event.getType());
        assertEquals(mouseEvent, event.getMouseEvent());
        assertNull(event.getKeyEvent());
        assertTrue(event.getTimestamp() > 0);
    }

    @Test
    public void testInputEventGetType() {
        NetBluetoothInputEvent.KeyEvent keyEvent = new NetBluetoothInputEvent.KeyEvent(
                0x04,
                NetBluetoothInputEvent.KeyEvent.KeyAction.PRESS,
                0x02
        );
        NetBluetoothInputEvent event = new NetBluetoothInputEvent(keyEvent);
        assertEquals(NetBluetoothInputEvent.InputType.KEYBOARD, event.getType());
    }

    @Test
    public void testInputEventGetKeyEvent() {
        NetBluetoothInputEvent.KeyEvent keyEvent = new NetBluetoothInputEvent.KeyEvent(
                0x04,
                NetBluetoothInputEvent.KeyEvent.KeyAction.PRESS,
                0x02
        );
        NetBluetoothInputEvent event = new NetBluetoothInputEvent(keyEvent);
        assertEquals(keyEvent, event.getKeyEvent());
    }

    @Test
    public void testInputEventGetMouseEvent() {
        NetBluetoothInputEvent.MouseEvent mouseEvent = new NetBluetoothInputEvent.MouseEvent(
                100, 200, 0x01, 0, NetBluetoothInputEvent.MouseEvent.MouseEventType.MOVE
        );
        NetBluetoothInputEvent event = new NetBluetoothInputEvent(mouseEvent);
        assertEquals(mouseEvent, event.getMouseEvent());
    }

    @Test
    public void testInputEventGetTimestamp() {
        NetBluetoothInputEvent.KeyEvent keyEvent = new NetBluetoothInputEvent.KeyEvent(
                0x04,
                NetBluetoothInputEvent.KeyEvent.KeyAction.PRESS,
                0x02
        );
        NetBluetoothInputEvent event = new NetBluetoothInputEvent(keyEvent);
        long timestamp = event.getTimestamp();
        assertTrue(timestamp > 0);
        assertTrue(System.currentTimeMillis() - timestamp < 1000);
    }

    @Test
    public void testInputEventToStringKeyboard() {
        NetBluetoothInputEvent.KeyEvent keyEvent = new NetBluetoothInputEvent.KeyEvent(
                0x04,
                NetBluetoothInputEvent.KeyEvent.KeyAction.PRESS,
                0x02
        );
        NetBluetoothInputEvent event = new NetBluetoothInputEvent(keyEvent);
        String str = event.toString();
        assertTrue(str.contains("KEYBOARD"));
        assertTrue(str.contains("4"));
        assertTrue(str.contains("PRESS"));
    }

    @Test
    public void testInputEventToStringMouse() {
        NetBluetoothInputEvent.MouseEvent mouseEvent = new NetBluetoothInputEvent.MouseEvent(
                100, 200, 0x01, 0, NetBluetoothInputEvent.MouseEvent.MouseEventType.MOVE
        );
        NetBluetoothInputEvent event = new NetBluetoothInputEvent(mouseEvent);
        String str = event.toString();
        assertTrue(str.contains("MOUSE"));
        assertTrue(str.contains("100"));
        assertTrue(str.contains("200"));
        assertTrue(str.contains("MOVE"));
    }

    @Test
    public void testKeyActionEnum() {
        assertEquals(2, NetBluetoothInputEvent.KeyEvent.KeyAction.values().length);
        assertEquals("PRESS", NetBluetoothInputEvent.KeyEvent.KeyAction.PRESS.name());
        assertEquals("RELEASE", NetBluetoothInputEvent.KeyEvent.KeyAction.RELEASE.name());
    }

    @Test
    public void testMouseEventTypeEnum() {
        assertEquals(4, NetBluetoothInputEvent.MouseEvent.MouseEventType.values().length);
        assertEquals("MOVE", NetBluetoothInputEvent.MouseEvent.MouseEventType.MOVE.name());
        assertEquals("BUTTON_PRESS", NetBluetoothInputEvent.MouseEvent.MouseEventType.BUTTON_PRESS.name());
        assertEquals("BUTTON_RELEASE", NetBluetoothInputEvent.MouseEvent.MouseEventType.BUTTON_RELEASE.name());
        assertEquals("SCROLL", NetBluetoothInputEvent.MouseEvent.MouseEventType.SCROLL.name());
    }

    @Test
    public void testInputTypeEnum() {
        assertEquals(2, NetBluetoothInputEvent.InputType.values().length);
        assertEquals("KEYBOARD", NetBluetoothInputEvent.InputType.KEYBOARD.name());
        assertEquals("MOUSE", NetBluetoothInputEvent.InputType.MOUSE.name());
    }
}
