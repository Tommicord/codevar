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

import java.util.Arrays;

/**
 * Represents a Bluetooth device with its identifying information and runtime state.
 *
 * <p>This class encapsulates the essential properties of a Bluetooth device including
 * its MAC address, device class, human-readable name, and runtime state such as
 * connection status, signal strength, and last seen timestamp. It serves as a comprehensive
 * data container for Bluetooth device information discovered or used within the application.</p>
 *
 * <p><b>Usage Example:</b></p>
 * <pre>{@code
 * NetBluetoothMacAddress macAddress = new NetBluetoothMacAddress(new byte[]{0x5F, 0x2E, 0x3D, 0x7C, 0x4F, 0x1A});
 * NetBluetoothDevice device = new NetBluetoothDevice("My Device", macAddress, NetBluetoothDeviceClass.BLE);
 *
 * // Update runtime state
 * device.setRssi(-65);
 * device.setConnected(true);
 * device.updateLastSeen();
 *
 * // Check device properties
 * String deviceName = device.getName();
 * String macString = device.getAddress().toString();
 * boolean isConnected = device.isConnected();
 * int signalStrength = device.getRssi();
 * }</pre>
 */
public class NetBluetoothDevice {
    /**
     * The MAC address of the Bluetooth device
     */
    protected NetBluetoothMacAddress address;

    /**
     * The device class (BLE, Classic, Dual, or Unknown)
     */
    protected NetBluetoothDeviceClass clazz;

    /**
     * The human-readable name of the Bluetooth device
     */
    protected String name;

    /**
     * The signal strength (RSSI) in dBm. Values typically range from -100 to 0.
     */
    protected int rssi;

    /**
     * Whether the device is currently connected
     */
    protected boolean connected;

    /**
     * The timestamp (in milliseconds) when this device was last seen
     */
    protected long lastSeenTimestamp;

    /**
     * Constructs a new NetBluetoothDevice with the specified properties.
     *
     * @param name    The human-readable name of the Bluetooth device. Can be null or empty.
     * @param address The MAC address of the device. Must not be null.
     * @param clazz   The device class indicating the type of Bluetooth technology (BLE, Classic, Dual, or Unknown).
     *                Must not be null.
     */
    public NetBluetoothDevice(String name, NetBluetoothMacAddress address, NetBluetoothDeviceClass clazz) {
        this.address = address;
        this.clazz = clazz;
        this.name = name;
        this.rssi = Integer.MIN_VALUE;
        this.connected = false;
        this.lastSeenTimestamp = System.currentTimeMillis();
    }

    /**
     * Returns the MAC address of this Bluetooth device.
     *
     * @return The {@link NetBluetoothMacAddress} object containing the device's MAC address.
     * The returned object is a reference to the internal instance, not a copy.
     */
    public NetBluetoothMacAddress getAddress() {
        return address;
    }

    /**
     * Sets the MAC address of this Bluetooth device.
     *
     * @param address The new MAC address. Must not be null.
     */
    public void setAddress(NetBluetoothMacAddress address) {
        this.address = address;
    }

    /**
     * Returns the human-readable name of this Bluetooth device.
     *
     * @return The device name as a String. May be null or empty if the device name is not available.
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the human-readable name of this Bluetooth device.
     *
     * @param name The new device name. Can be null or empty.
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Returns the device class of this Bluetooth device.
     *
     * @return The {@link NetBluetoothDeviceClass} enum value indicating the type of Bluetooth technology
     * used by this device (BLE, CLASSIC, DUAL, or UNKNOWN).
     */
    public NetBluetoothDeviceClass getClazz() {
        return clazz;
    }

    /**
     * Sets the device class of this Bluetooth device.
     *
     * @param clazz The new device class. Must not be null.
     */
    public void setClazz(NetBluetoothDeviceClass clazz) {
        this.clazz = clazz;
    }

    /**
     * Returns the signal strength (RSSI) of this Bluetooth device.
     *
     * @return The RSSI value in dBm. Returns {@link Integer#MIN_VALUE} if RSSI has not been set.
     * Values typically range from -100 (weak signal) to 0 (strong signal).
     */
    public int getRssi() {
        return rssi;
    }

    /**
     * Sets the signal strength (RSSI) of this Bluetooth device.
     *
     * @param rssi The RSSI value in dBm. Values typically range from -100 to 0.
     */
    public void setRssi(int rssi) {
        this.rssi = rssi;
    }

    /**
     * Returns whether this Bluetooth device is currently connected.
     *
     * @return true if the device is connected, false otherwise.
     */
    public boolean isConnected() {
        return connected;
    }

    /**
     * Sets the connection state of this Bluetooth device.
     *
     * @param connected true if the device is connected, false otherwise.
     */
    public void setConnected(boolean connected) {
        this.connected = connected;
    }

    /**
     * Returns the timestamp when this device was last seen.
     *
     * @return The last seen timestamp in milliseconds since epoch.
     */
    public long getLastSeenTimestamp() {
        return lastSeenTimestamp;
    }

    /**
     * Updates the last seen timestamp to the current time.
     */
    public void updateLastSeen() {
        this.lastSeenTimestamp = System.currentTimeMillis();
    }

    /**
     * Returns the time elapsed since this device was last seen.
     *
     * @return The elapsed time in milliseconds.
     */
    public long getTimeSinceLastSeen() {
        return System.currentTimeMillis() - lastSeenTimestamp;
    }

    /**
     * Checks if this device matches another device based on MAC address.
     *
     * @param other The other device to compare with. Can be null.
     * @return true if both devices have the same MAC address, false otherwise.
     */
    public boolean matches(NetBluetoothDevice other) {
        if (other == null || other.address == null || this.address == null) {
            return false;
        }
        return Arrays.equals(this.address.getBuffer(), other.address.getBuffer());
    }

    /**
     * Checks if this device has the specified MAC address.
     *
     * @param macAddress The MAC address string to check. Can be null.
     * @return true if this device's MAC address matches the specified address, false otherwise.
     */
    public boolean hasMacAddress(String macAddress) {
        if (macAddress == null || this.address == null) {
            return false;
        }
        return this.address.toString().equalsIgnoreCase(macAddress);
    }

    /**
     * Returns a string representation of this Bluetooth device.
     *
     * @return A string containing the device name, MAC address, and connection state.
     */
    @Override
    public String toString() {
        return "NetBluetoothDevice{" +
                "name='" + name + '\'' +
                ", address=" + (address != null ? address.toString() : "null") +
                ", clazz=" + clazz +
                ", rssi=" + rssi +
                ", connected=" + connected +
                ", lastSeen=" + lastSeenTimestamp +
                '}';
    }

    /**
     * Checks if this device is equal to another object.
     *
     * @param obj The object to compare with.
     * @return true if the objects are equal (same MAC address), false otherwise.
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        NetBluetoothDevice that = (NetBluetoothDevice) obj;
        return address != null && Arrays.equals(address.getBuffer(), that.address.getBuffer());
    }

    /**
     * Returns the hash code of this device based on its MAC address or Last seen timestamp.
     *
     * @return The hash code.
     */
    @Override
    public int hashCode() {
        return address != null ? Arrays.hashCode(address.getBuffer()) : (int) this.lastSeenTimestamp;
    }
}
