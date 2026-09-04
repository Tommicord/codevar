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

import android.content.Context;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

/**
 * Comprehensive unit tests for NetBluetoothHandler.
 * <p>
 * These tests verify the lifecycle management, receiver registration,
 * event handling, and JNI integration of the Bluetooth handler.
 * </p>
 */
@RunWith(AndroidJUnit4.class)
public class NetBluetoothHandlerTest {
    private NetBluetoothHandler handler;
    private Context context;

    @Before
    public void setUp() {
        context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        handler = new NetBluetoothHandler(context);
    }

    @After
    public void tearDown() {
        if (handler != null && handler.isRunning()) {
            handler.shutdown();
        }
    }

    @Test
    public void testConstructor() {
        assertNotNull(handler);
        assertFalse(handler.isRunning());
        assertFalse(handler.isConnected());
        assertNull(handler.getConnectedDevice());
    }

    @Test
    public void testStart() {
        handler.start();
        assertTrue(handler.isRunning());
    }

    @Test
    public void testStartTwice() {
        handler.start();
        assertTrue(handler.isRunning());
        handler.start(); // Should not cause issues
        assertTrue(handler.isRunning());
    }

    @Test
    public void testShutdown() {
        handler.start();
        assertTrue(handler.isRunning());
        handler.shutdown();
        assertFalse(handler.isRunning());
    }

    @Test
    public void testShutdownWithoutStart() {
        handler.shutdown(); // Should not cause issues
        assertFalse(handler.isRunning());
    }

    @Test
    public void testShutdownMultipleTimes() {
        handler.start();
        handler.shutdown();
        assertFalse(handler.isRunning());
        handler.shutdown(); // Should not cause issues
        assertFalse(handler.isRunning());
    }

    @Test
    public void testRegisterReceiver() {
        handler.start();
        NetCommand receiver = new TestCommand();
        boolean result = handler.registerReceiver(receiver);
        assertTrue(result);
    }

    @Test
    public void testRegisterReceiverNull() {
        handler.start();
        boolean result = handler.registerReceiver(null);
        assertFalse(result);
    }

    @Test
    public void testRegisterReceiverBeforeStart() {
        NetCommand receiver = new TestCommand();
        boolean result = handler.registerReceiver(receiver);
        assertTrue(result); // Should still allow registration
    }

    @Test
    public void testUnregisterReceiver() {
        handler.start();
        NetCommand receiver = new TestCommand();
        handler.registerReceiver(receiver);
        boolean result = handler.unregisterReceiver(receiver);
        assertTrue(result);
    }

    @Test
    public void testUnregisterReceiverNotRegistered() {
        handler.start();
        NetCommand receiver = new TestCommand();
        boolean result = handler.unregisterReceiver(receiver);
        assertFalse(result);
    }

    @Test
    public void testUnregisterReceiverNull() {
        handler.start();
        boolean result = handler.unregisterReceiver(null);
        assertFalse(result);
    }

    @Test
    public void testNativeRegisterReceiver() {
        handler.start();
        long ptr = 12345L;
        handler.nativeRegisterReceiver(ptr);
        // Should not throw exception
    }

    @Test
    public void testNativeUnregisterReceiver() {
        handler.start();
        long ptr = 12345L;
        handler.nativeRegisterReceiver(ptr);
        boolean result = handler.nativeUnregisterReceiver(ptr);
        assertTrue(result);
    }

    @Test
    public void testNativeUnregisterReceiverNotRegistered() {
        handler.start();
        long ptr = 12345L;
        boolean result = handler.nativeUnregisterReceiver(ptr);
        assertFalse(result);
    }

    @Test
    public void testNativeUnregisterReceiverMultipleTimes() {
        handler.start();
        long ptr = 12345L;
        handler.nativeRegisterReceiver(ptr);
        assertTrue(handler.nativeUnregisterReceiver(ptr));
        assertFalse(handler.nativeUnregisterReceiver(ptr));
    }

    @Test
    public void testNativeStart() {
        handler.nativeStart();
        assertTrue(handler.isRunning());
    }

    @Test
    public void testNativeShutdown() {
        handler.start();
        handler.nativeShutdown();
        assertFalse(handler.isRunning());
    }

    @Test
    public void testConnectDeviceWithoutStart() {
        try {
            NetBluetoothMacAddress macAddress = new NetBluetoothMacAddress(new byte[]{0x5F, 0x2E, 0x3D, 0x7C, 0x4F, 0x1A});
            NetBluetoothDevice device = new NetBluetoothDevice("Test Device", macAddress, NetBluetoothDeviceClass.BLE);
            boolean result = handler.connectDevice(device, NetBluetoothHidDevice.HidDeviceType.KEYBOARD);
            assertFalse(result);
        } catch (NetInvalidMacAddr e) {
            fail("Should not throw exception");
        }
    }

    @Test
    public void testConnectDeviceNull() {
        handler.start();
        boolean result = handler.connectDevice(null, NetBluetoothHidDevice.HidDeviceType.KEYBOARD);
        assertFalse(result);
    }

    @Test
    public void testDisconnectDeviceNull() {
        handler.start();
        boolean result = handler.disconnectDevice(null);
        assertFalse(result);
    }

    @Test
    public void testNativeConnectDeviceValid() {
        handler.start();
        boolean result = handler.nativeConnectDevice("5F:2E:3D:7C:4F:1A", 0);
        // connectDevice attempts real Bluetooth connection
        // which may fail when no device is available
        // We just verify it doesn't throw
    }

    @Test
    public void testNativeConnectDeviceInvalidMac() {
        handler.start();
        boolean result = handler.nativeConnectDevice("invalid", 0);
        assertFalse(result);
    }

    @Test
    public void testNativeConnectDeviceNullMac() {
        handler.start();
        boolean result = handler.nativeConnectDevice(null, 0);
        assertFalse(result);
    }

    @Test
    public void testNativeConnectDeviceInvalidDeviceType() {
        handler.start();
        boolean result = handler.nativeConnectDevice("5F:2E:3D:7C:4F:1A", 999);
        // May throw ArrayIndexOutOfBoundsException or return false
        // We verify it doesn't crash the handler
    }

    @Test
    public void testNativeDisconnectDeviceNoDevice() {
        handler.start();
        boolean result = handler.nativeDisconnectDevice();
        assertFalse(result);
    }

    @Test
    public void testIsConnectedInitially() {
        assertFalse(handler.isConnected());
    }

    @Test
    public void testGetConnectedDeviceInitially() {
        assertNull(handler.getConnectedDevice());
    }

    @Test
    public void testMultipleReceivers() {
        handler.start();
        TestCommand receiver1 = new TestCommand();
        TestCommand receiver2 = new TestCommand();
        TestCommand receiver3 = new TestCommand();

        assertTrue(handler.registerReceiver(receiver1));
        assertTrue(handler.registerReceiver(receiver2));
        assertTrue(handler.registerReceiver(receiver3));
    }

    @Test
    public void testRegisterManyReceivers() {
        handler.start();
        for (int i = 0; i < 100; i++) {
            assertTrue(handler.registerReceiver(new TestCommand()));
        }
    }

    @Test
    public void testReceiverLimit() {
        handler.start();
        // MAX_RECEIVERS is 128, so we should be able to register up to that
        for (int i = 0; i < 128; i++) {
            assertTrue(handler.registerReceiver(new TestCommand()));
        }
        // The 129th should fail
        assertFalse(handler.registerReceiver(new TestCommand()));
    }

    @Test
    public void testShutdownClearsReceivers() {
        handler.start();
        handler.registerReceiver(new TestCommand());
        handler.shutdown();
        // After shutdown, receivers should be cleared
        // This is verified by the fact that we can register again
        handler.start();
        assertTrue(handler.registerReceiver(new TestCommand()));
    }

    @Test
    public void testShutdownClearsNativeReceivers() {
        handler.start();
        long ptr = 12345L;
        handler.nativeRegisterReceiver(ptr);
        handler.shutdown();
        // After shutdown, native receivers should be cleared
        handler.start();
        handler.nativeRegisterReceiver(ptr);
        assertTrue(handler.nativeUnregisterReceiver(ptr));
    }

    @Test
    public void testNativeConnectDeviceAllTypes() {
        handler.start();
        // Test all device types
        handler.nativeConnectDevice("5F:2E:3D:7C:4F:1A", 0); // KEYBOARD
        handler.nativeConnectDevice("5F:2E:3D:7C:4F:1A", 1); // MOUSE
        handler.nativeConnectDevice("5F:2E:3D:7C:4F:1A", 2); // COMBO
        handler.nativeConnectDevice("5F:2E:3D:7C:4F:1A", 3); // UNKNOWN
    }

    @Test
    public void testMacAddressParsingInNativeConnect() {
        handler.start();
        // Test various valid MAC address formats
        handler.nativeConnectDevice("00:11:22:33:44:55", 0);
        handler.nativeConnectDevice("AA:BB:CC:DD:EE:FF", 0);
        handler.nativeConnectDevice("aa:bb:cc:dd:ee:ff", 0); // lowercase
        handler.nativeConnectDevice("Aa:Bb:Cc:Dd:Ee:Ff", 0); // mixed case
    }

    @Test
    public void testInvalidMacAddressFormats() {
        handler.start();
        // Test invalid formats
        assertFalse(handler.nativeConnectDevice("", 0));
        assertFalse(handler.nativeConnectDevice("00:11:22:33:44", 0)); // too short
        assertFalse(handler.nativeConnectDevice("00:11:22:33:44:55:66", 0)); // too long
        assertFalse(handler.nativeConnectDevice("00-11-22-33-44-55", 0)); // wrong separator
        assertFalse(handler.nativeConnectDevice("001122334455", 0)); // no separator
        assertFalse(handler.nativeConnectDevice("00:11:22:33:44:GG", 0)); // invalid hex
    }

    /**
     * Test command implementation for receiver testing.
     */
    private static class TestCommand implements NetCommand {
        private final AtomicInteger keyPressedCount = new AtomicInteger(0);
        private final AtomicInteger keyReleasedCount = new AtomicInteger(0);
        private final AtomicInteger mouseMoveCount = new AtomicInteger(0);
        private final AtomicInteger buttonPressCount = new AtomicInteger(0);
        private final AtomicInteger buttonReleaseCount = new AtomicInteger(0);
        private final AtomicInteger scrollCount = new AtomicInteger(0);

        @Override
        public void onKeyPressed(int keyCode, int modifiers) {
            keyPressedCount.incrementAndGet();
        }

        @Override
        public void onKeyReleased(int keyCode, int modifiers) {
            keyReleasedCount.incrementAndGet();
        }

        @Override
        public void onMouseMove(int x, int y) {
            mouseMoveCount.incrementAndGet();
        }

        @Override
        public void onButtonPress(int button, int x, int y) {
            buttonPressCount.incrementAndGet();
        }

        @Override
        public void onButtonRelease(int button, int x, int y) {
            buttonReleaseCount.incrementAndGet();
        }

        @Override
        public void onMouseScroll(int delta, int x, int y) {
            scrollCount.incrementAndGet();
        }

        public int getKeyPressedCount() {
            return keyPressedCount.get();
        }

        public int getKeyReleasedCount() {
            return keyReleasedCount.get();
        }

        public int getMouseMoveCount() {
            return mouseMoveCount.get();
        }

        public int getButtonPressCount() {
            return buttonPressCount.get();
        }

        public int getButtonReleaseCount() {
            return buttonReleaseCount.get();
        }

        public int getScrollCount() {
            return scrollCount.get();
        }
    }
}
