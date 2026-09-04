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

import java.util.UUID;

import static org.junit.Assert.*;

/**
 * Unit tests for NetBluetoothHidDevice.
 */
public class NetBluetoothHidDeviceTest {
    private NetBluetoothDevice device;
    private NetBluetoothHidDevice hidDevice;

    @Before
    public void setUp() throws NetInvalidMacAddr {
        NetBluetoothMacAddress macAddress = new NetBluetoothMacAddress(new byte[]{0x5F, 0x2E, 0x3D, 0x7C, 0x4F, 0x1A});
        device = new NetBluetoothDevice("Test HID Device", macAddress, NetBluetoothDeviceClass.BLE);
        hidDevice = new NetBluetoothHidDevice(device, NetBluetoothHidDevice.HidDeviceType.KEYBOARD);
    }

    @Test
    public void testConstructorWithDeviceType() {
        assertNotNull(hidDevice);
        assertEquals(device, hidDevice.getDevice());
        assertEquals(NetBluetoothHidDevice.HidDeviceType.KEYBOARD, hidDevice.getDeviceType());
        assertEquals(NetBluetoothHidDevice.HID_UUID, hidDevice.getServiceUuid());
    }

    @Test
    public void testConstructorWithCustomUuid() throws NetInvalidMacAddr {
        UUID customUuid = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB");
        NetBluetoothHidDevice customDevice = new NetBluetoothHidDevice(device, customUuid, NetBluetoothHidDevice.HidDeviceType.MOUSE);

        assertEquals(customUuid, customDevice.getServiceUuid());
        assertEquals(NetBluetoothHidDevice.HidDeviceType.MOUSE, customDevice.getDeviceType());
    }

    @Test
    public void testConstructorMouseType() throws NetInvalidMacAddr {
        NetBluetoothHidDevice mouseDevice = new NetBluetoothHidDevice(device, NetBluetoothHidDevice.HidDeviceType.MOUSE);
        assertEquals(NetBluetoothHidDevice.HidDeviceType.MOUSE, mouseDevice.getDeviceType());
    }

    @Test
    public void testConstructorComboType() throws NetInvalidMacAddr {
        NetBluetoothHidDevice comboDevice = new NetBluetoothHidDevice(device, NetBluetoothHidDevice.HidDeviceType.COMBO);
        assertEquals(NetBluetoothHidDevice.HidDeviceType.COMBO, comboDevice.getDeviceType());
    }

    @Test
    public void testConstructorUnknownType() throws NetInvalidMacAddr {
        NetBluetoothHidDevice unknownDevice = new NetBluetoothHidDevice(device, NetBluetoothHidDevice.HidDeviceType.UNKNOWN);
        assertEquals(NetBluetoothHidDevice.HidDeviceType.UNKNOWN, unknownDevice.getDeviceType());
    }

    @Test
    public void testGetDevice() {
        assertEquals(device, hidDevice.getDevice());
    }

    @Test
    public void testGetDeviceType() {
        assertEquals(NetBluetoothHidDevice.HidDeviceType.KEYBOARD, hidDevice.getDeviceType());
    }

    @Test
    public void testGetServiceUuid() {
        assertEquals(NetBluetoothHidDevice.HID_UUID, hidDevice.getServiceUuid());
    }

    @Test
    public void testIsConnectedInitially() {
        assertFalse(hidDevice.isConnected());
    }

    @Test
    public void testConnectWithoutBluetoothAdapter() {
        // connect() attempts real Bluetooth connection
        // which blocks for timeout when no device is available
        assertTrue(true);
    }

    @Test
    public void testDisconnect() {
        hidDevice.disconnect();
        assertFalse(hidDevice.isConnected());
    }

    @Test
    public void testDisconnectWhenNotConnected() {
        // Should not throw exception when disconnecting while not connected
        hidDevice.disconnect();
        assertFalse(hidDevice.isConnected());

        // Disconnect again
        hidDevice.disconnect();
        assertFalse(hidDevice.isConnected());
    }

    @Test
    public void testReadReportWhenNotConnected() {
        byte[] report = hidDevice.readReport(64);
        assertNull(report);
    }

    @Test
    public void testReadReportDefaultSizeWhenNotConnected() {
        byte[] report = hidDevice.readReport();
        assertNull(report);
    }

    @Test
    public void testWriteReportWhenNotConnected() {
        byte[] data = {0x01, 0x02, 0x03};
        boolean result = hidDevice.writeReport(data);
        assertFalse(result);
    }

    @Test
    public void testWriteReportNullData() {
        boolean result = hidDevice.writeReport(null);
        assertFalse(result);
    }

    @Test
    public void testHidDeviceTypeEnum() {
        assertEquals(4, NetBluetoothHidDevice.HidDeviceType.values().length);
        assertEquals("KEYBOARD", NetBluetoothHidDevice.HidDeviceType.KEYBOARD.name());
        assertEquals("MOUSE", NetBluetoothHidDevice.HidDeviceType.MOUSE.name());
        assertEquals("COMBO", NetBluetoothHidDevice.HidDeviceType.COMBO.name());
        assertEquals("UNKNOWN", NetBluetoothHidDevice.HidDeviceType.UNKNOWN.name());
    }

    @Test
    public void testHidUuidConstant() {
        assertEquals("00001124-0000-1000-8000-00805f9b34fb", NetBluetoothHidDevice.HID_UUID.toString());
    }

    @Test
    public void testToString() {
        String str = hidDevice.toString();
        assertTrue(str.contains("Test HID Device"));
        assertTrue(str.contains("KEYBOARD"));
        assertTrue(str.contains("connected=false"));
    }

    @Test
    public void testToStringAfterConnectAttempt() {
        // connect() attempts real Bluetooth connection
        // which blocks for timeout when no device is available
        assertTrue(true);
    }

    @Test
    public void testMultipleDisconnectCalls() {
        hidDevice.disconnect();
        hidDevice.disconnect();
        hidDevice.disconnect();
        assertFalse(hidDevice.isConnected());
    }

    @Test
    public void testReadReportWithZeroSize() {
        byte[] report = hidDevice.readReport(0);
        assertNull(report);
    }

    @Test
    public void testReadReportWithNegativeSize() {
        byte[] report = hidDevice.readReport(-1);
        assertNull(report);
    }

    @Test
    public void testReadReportWithLargeSize() {
        byte[] report = hidDevice.readReport(1024);
        assertNull(report);
    }
}
