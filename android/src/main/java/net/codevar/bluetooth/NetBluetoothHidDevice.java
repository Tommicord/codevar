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

import java.util.UUID;

/**
 * Represents a Bluetooth HID device with communication capabilities.
 *
 * <p>This class manages the connection to a Bluetooth HID device (keyboard or mouse),
 * handles data communication through the socket, and provides methods for sending
 * and receiving HID reports.</p>
 *
 * <p><b>Usage Example:</b></p>
 * <pre>{@code
 * NetBluetoothHidDevice hidDevice = new NetBluetoothHidDevice(device, uuid);
 * if (hidDevice.connect()) {
 *     byte[] report = hidDevice.readReport();
 *     hidDevice.writeReport(responseData);
 *     hidDevice.disconnect();
 * }
 * }</pre>
 */
public class NetBluetoothHidDevice {
    /**
     * Standard HID UUID for Bluetooth HID devices
     */
    public static final UUID HID_UUID = UUID.fromString("00001124-0000-1000-8000-00805F9B34FB");
    private static final String TAG = "Net:HidDevice";
    private final NetBluetoothDevice device;
    private final UUID serviceUuid;
    private final HidDeviceType deviceType;
    private NetBluetoothSocket socket;
    private NetBluetoothByteStream byteStream;
    private boolean connected;

    /**
     * Constructs a new NetBluetoothHidDevice.
     *
     * @param device      The Bluetooth device. Must not be null.
     * @param serviceUuid The service UUID to connect to. Must not be null.
     * @param deviceType  The type of HID device. Must not be null.
     */
    public NetBluetoothHidDevice(@NonNull NetBluetoothDevice device, @NonNull UUID serviceUuid,
                                 @NonNull HidDeviceType deviceType) {
        this.device = device;
        this.serviceUuid = serviceUuid;
        this.deviceType = deviceType;
        this.socket = null;
        this.byteStream = null;
        this.connected = false;
    }

    /**
     * Constructs a new NetBluetoothHidDevice with the standard HID UUID.
     *
     * @param device     The Bluetooth device. Must not be null.
     * @param deviceType The type of HID device. Must not be null.
     */
    public NetBluetoothHidDevice(@NonNull NetBluetoothDevice device, @NonNull HidDeviceType deviceType) {
        this(device, HID_UUID, deviceType);
    }

    /**
     * Connects to the HID device.
     *
     * <p>This method establishes a Bluetooth connection to the device and initializes
     * the byte stream for communication.</p>
     *
     * @return true if the connection was successful, false otherwise.
     */
    public boolean connect() {
        if (connected) {
            Log.w(TAG, "Already connected to device: " + device.getAddress());
            return true;
        }

        try {
            android.bluetooth.BluetoothDevice btDevice = getAndroidBluetoothDevice();
            if (btDevice == null) {
                Log.e(TAG, "Cannot get Android Bluetooth device");
                return false;
            }

            socket = new NetBluetoothSocket(btDevice, serviceUuid);
            if (!socket.connect()) {
                Log.e(TAG, "Failed to connect socket");
                return false;
            }

            byteStream = new NetBluetoothByteStream(socket);
            connected = true;
            Log.i(TAG, "Connected to HID device: " + device.getName());
            return true;
        } catch (Exception e) {
            Log.e(TAG, "Error connecting to HID device", e);
            disconnect();
            return false;
        }
    }

    /**
     * Disconnects from the HID device.
     *
     * <p>This method closes the socket and releases all resources.</p>
     */
    public void disconnect() {
        if (byteStream != null) {
            byteStream.close();
            byteStream = null;
        }

        if (socket != null) {
            socket.disconnect();
            socket = null;
        }

        connected = false;
        Log.i(TAG, "Disconnected from HID device: " + device.getName());
    }

    /**
     * Reads a HID report from the device.
     *
     * <p>This method reads a standard HID report (typically 8-64 bytes) from the device.</p>
     *
     * @param reportSize The expected size of the report in bytes.
     * @return The report data, or null if an error occurs.
     */
    @Nullable
    public byte[] readReport(int reportSize) {
        if (!connected || byteStream == null) {
            Log.e(TAG, "Not connected to device");
            return null;
        }

        return byteStream.readBytes(reportSize);
    }

    /**
     * Reads a HID report with default size (64 bytes).
     *
     * @return The report data, or null if an error occurs.
     */
    @Nullable
    public byte[] readReport() {
        return readReport(64);
    }

    /**
     * Writes a HID report to the device.
     *
     * @param report The report data to write. Must not be null.
     * @return true if the write was successful, false otherwise.
     */
    public boolean writeReport(@NonNull byte[] report) {
        if (!connected || byteStream == null) {
            Log.e(TAG, "Not connected to device");
            return false;
        }

        return byteStream.writeBytes(report);
    }

    /**
     * Returns whether the device is connected.
     *
     * @return true if connected, false otherwise.
     */
    public boolean isConnected() {
        return connected && (socket != null && socket.isConnected());
    }

    /**
     * Returns the NetBluetoothDevice associated with this HID device.
     *
     * @return The device.
     */
    @NonNull
    public NetBluetoothDevice getDevice() {
        return device;
    }

    /**
     * Returns the HID device type.
     *
     * @return The device type.
     */
    @NonNull
    public HidDeviceType getDeviceType() {
        return deviceType;
    }

    /**
     * Returns the service UUID used for this connection.
     *
     * @return The service UUID.
     */
    @NonNull
    public UUID getServiceUuid() {
        return serviceUuid;
    }

    /**
     * Returns the Android BluetoothDevice object.
     *
     * @return The Android BluetoothDevice, or null if not available.
     */
    @Nullable
    private android.bluetooth.BluetoothDevice getAndroidBluetoothDevice() {
        try {
            android.bluetooth.BluetoothAdapter adapter =
                    android.bluetooth.BluetoothAdapter.getDefaultAdapter();
            if (adapter == null) {
                return null;
            }

            String macAddress = device.getAddress().toString();
            return adapter.getRemoteDevice(macAddress);
        } catch (Exception e) {
            Log.e(TAG, "Error getting Android Bluetooth device", e);
            return null;
        }
    }

    /**
     * Returns a string representation of this HID device.
     *
     * @return A string describing the device.
     */
    @Override
    public String toString() {
        return "NetBluetoothHidDevice{" +
                "device=" + device.getName() +
                ", type=" + deviceType +
                ", connected=" + connected +
                '}';
    }

    /**
     * Enumeration of HID device types.
     */
    public enum HidDeviceType {
        /**
         * Keyboard device
         */
        KEYBOARD,
        /**
         * Mouse device
         */
        MOUSE,
        /**
         * Combined keyboard and mouse device
         */
        COMBO,
        /**
         * Unknown HID device
         */
        UNKNOWN
    }
}
