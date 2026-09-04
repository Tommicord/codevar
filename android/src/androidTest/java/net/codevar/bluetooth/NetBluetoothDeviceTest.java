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
 * Unit tests for NetBluetoothDevice.
 */
public class NetBluetoothDeviceTest {
    private NetBluetoothMacAddress macAddress;
    private NetBluetoothDevice device;

    @Before
    public void setUp() throws NetInvalidMacAddr {
        macAddress = new NetBluetoothMacAddress(new byte[]{0x5F, 0x2E, 0x3D, 0x7C, 0x4F, 0x1A});
        device = new NetBluetoothDevice("Test Device", macAddress, NetBluetoothDeviceClass.BLE);
    }

    @Test
    public void testConstructor() {
        assertNotNull(device);
        assertEquals("Test Device", device.getName());
        assertEquals(macAddress, device.getAddress());
        assertEquals(NetBluetoothDeviceClass.BLE, device.getClazz());
    }

    @Test
    public void testConstructorWithNullName() throws NetInvalidMacAddr {
        NetBluetoothDevice nullNameDevice = new NetBluetoothDevice(null, macAddress, NetBluetoothDeviceClass.CLASSIC);
        assertNull(nullNameDevice.getName());
        assertEquals(macAddress, nullNameDevice.getAddress());
        assertEquals(NetBluetoothDeviceClass.CLASSIC, nullNameDevice.getClazz());
    }

    @Test
    public void testGetAddress() {
        assertEquals(macAddress, device.getAddress());
    }

    @Test
    public void testSetAddress() throws NetInvalidMacAddr {
        NetBluetoothMacAddress newMac = new NetBluetoothMacAddress(new byte[]{0x01, 0x02, 0x03, 0x04, 0x05, 0x06});
        device.setAddress(newMac);
        assertEquals(newMac, device.getAddress());
    }

    @Test
    public void testGetName() {
        assertEquals("Test Device", device.getName());
    }

    @Test
    public void testSetName() {
        device.setName("New Name");
        assertEquals("New Name", device.getName());
    }

    @Test
    public void testSetNullName() {
        device.setName(null);
        assertNull(device.getName());
    }

    @Test
    public void testGetClazz() {
        assertEquals(NetBluetoothDeviceClass.BLE, device.getClazz());
    }

    @Test
    public void testSetClazz() {
        device.setClazz(NetBluetoothDeviceClass.CLASSIC);
        assertEquals(NetBluetoothDeviceClass.CLASSIC, device.getClazz());
    }

    @Test
    public void testGetRssi() {
        assertEquals(Integer.MIN_VALUE, device.getRssi());
    }

    @Test
    public void testSetRssi() {
        device.setRssi(-65);
        assertEquals(-65, device.getRssi());
    }

    @Test
    public void testIsConnected() {
        assertFalse(device.isConnected());
    }

    @Test
    public void testSetConnected() {
        device.setConnected(true);
        assertTrue(device.isConnected());
    }

    @Test
    public void testGetLastSeenTimestamp() {
        long timestamp = device.getLastSeenTimestamp();
        assertTrue(timestamp > 0);
        assertTrue(System.currentTimeMillis() - timestamp < 1000);
    }

    @Test
    public void testUpdateLastSeen() throws InterruptedException {
        Thread.sleep(10);
        long oldTimestamp = device.getLastSeenTimestamp();
        device.updateLastSeen();
        long newTimestamp = device.getLastSeenTimestamp();
        assertTrue(newTimestamp > oldTimestamp);
    }

    @Test
    public void testGetTimeSinceLastSeen() throws InterruptedException {
        Thread.sleep(10);
        long timeSince = device.getTimeSinceLastSeen();
        assertTrue(timeSince >= 10);
    }

    @Test
    public void testMatches() throws NetInvalidMacAddr {
        NetBluetoothMacAddress sameMac = new NetBluetoothMacAddress(new byte[]{0x5F, 0x2E, 0x3D, 0x7C, 0x4F, 0x1A});
        NetBluetoothDevice sameDevice = new NetBluetoothDevice("Different Name", sameMac, NetBluetoothDeviceClass.CLASSIC);
        assertTrue(device.matches(sameDevice));
    }

    @Test
    public void testMatchesDifferentMac() throws NetInvalidMacAddr {
        NetBluetoothMacAddress differentMac = new NetBluetoothMacAddress(new byte[]{0x01, 0x02, 0x03, 0x04, 0x05, 0x06});
        NetBluetoothDevice differentDevice = new NetBluetoothDevice("Test Device", differentMac, NetBluetoothDeviceClass.BLE);
        assertFalse(device.matches(differentDevice));
    }

    @Test
    public void testMatchesNull() {
        assertFalse(device.matches(null));
    }

    @Test
    public void testHasMacAddress() {
        assertTrue(device.hasMacAddress("5F:2E:3D:7C:4F:1A"));
        assertTrue(device.hasMacAddress("5f:2e:3d:7c:4f:1a"));
    }

    @Test
    public void testHasMacAddressDifferent() {
        assertFalse(device.hasMacAddress("01:02:03:04:05:06"));
    }

    @Test
    public void testHasMacAddressNull() {
        assertFalse(device.hasMacAddress(null));
    }

    @Test
    public void testToString() {
        String str = device.toString();
        assertTrue(str.contains("Test Device"));
        assertTrue(str.contains("5F:2E:3D:7C:4F:1A"));
        assertTrue(str.contains("BLE"));
    }

    @Test
    public void testEquals() throws NetInvalidMacAddr {
        NetBluetoothMacAddress sameMac = new NetBluetoothMacAddress(new byte[]{0x5F, 0x2E, 0x3D, 0x7C, 0x4F, 0x1A});
        NetBluetoothDevice sameDevice = new NetBluetoothDevice("Different Name", sameMac, NetBluetoothDeviceClass.CLASSIC);
        assertEquals(device, sameDevice);
    }

    @Test
    public void testEqualsDifferent() throws NetInvalidMacAddr {
        NetBluetoothMacAddress differentMac = new NetBluetoothMacAddress(new byte[]{0x01, 0x02, 0x03, 0x04, 0x05, 0x06});
        NetBluetoothDevice differentDevice = new NetBluetoothDevice("Test Device Different", differentMac, NetBluetoothDeviceClass.BLE);
        assertNotEquals(device, differentDevice);
    }

    @Test
    public void testEqualsNull() {
        assertNotEquals(null, device);
    }

    @Test
    public void testEqualsSame() {
        assertEquals(device, device);
    }

    @Test
    public void testHashCode() throws NetInvalidMacAddr {
        NetBluetoothMacAddress sameMac = new NetBluetoothMacAddress(new byte[]{0x5F, 0x2E, 0x3D, 0x7C, 0x4F, 0x1A});
        NetBluetoothDevice sameDevice = new NetBluetoothDevice("Different Name", sameMac, NetBluetoothDeviceClass.CLASSIC);
        assertEquals(device.hashCode(), sameDevice.hashCode());
    }

    @Test
    public void testHashCodeDifferent() throws NetInvalidMacAddr {
        NetBluetoothMacAddress differentMac = new NetBluetoothMacAddress(new byte[]{0x01, 0x02, 0x03, 0x04, 0x05, 0x06});
        NetBluetoothDevice differentDevice = new NetBluetoothDevice("Test Device", differentMac, NetBluetoothDeviceClass.BLE);
        assertNotEquals(device.hashCode(), differentDevice.hashCode());
    }
}
