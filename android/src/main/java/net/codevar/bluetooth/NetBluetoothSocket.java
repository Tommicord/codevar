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

import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothSocket;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.UUID;

/**
 * A wrapper class for managing Bluetooth socket connections.
 *
 * <p>This class provides a simplified interface for connecting to Bluetooth devices
 * and managing the underlying socket lifecycle. It handles connection establishment,
 * stream access, and proper resource cleanup.</p>
 *
 * <p><b>Usage Example:</b></p>
 * <pre>{@code
 * NetBluetoothSocket socket = new NetBluetoothSocket(device, uuid);
 * if (socket.connect()) {
 *     InputStream input = socket.getInputStream();
 *     OutputStream output = socket.getOutputStream();
 *     // Use streams for communication
 *     socket.disconnect();
 * }
 * }</pre>
 */
public class NetBluetoothSocket {
    private static final String TAG = "Net:BluetoothSocket";

    private final BluetoothDevice device;
    private final UUID uuid;
    private BluetoothSocket socket;
    private boolean connected;

    /**
     * Constructs a new NetBluetoothSocket for the specified device and UUID.
     *
     * @param device The Bluetooth device to connect to. Must not be null.
     * @param uuid   The UUID of the service to connect to. Must not be null.
     */
    public NetBluetoothSocket(@NonNull BluetoothDevice device, @NonNull UUID uuid) {
        this.device = device;
        this.uuid = uuid;
        this.socket = null;
        this.connected = false;
    }

    /**
     * Establishes a connection to the Bluetooth device.
     *
     * <p>This method creates a socket and attempts to connect to the device.
     * The connection is blocking and may take several seconds.</p>
     *
     * @return true if the connection was successful, false otherwise.
     */
    public boolean connect() {
        if (connected) {
            Log.w(TAG, "Already connected to device: " + device.getAddress());
            return true;
        }

        try {
            socket = device.createRfcommSocketToServiceRecord(uuid);
            socket.connect();
            connected = true;
            Log.i(TAG, "Connected to device: " + device.getAddress());
            return true;
        } catch (IOException e) {
            Log.e(TAG, "Failed to connect to device: " + device.getAddress(), e);
            disconnect();
            return false;
        }
    }

    /**
     * Disconnects from the Bluetooth device and closes the socket.
     *
     * <p>This method safely closes the socket and releases all resources.
     * It can be called multiple times without side effects.</p>
     */
    public void disconnect() {
        if (socket != null) {
            try {
                socket.close();
            } catch (IOException e) {
                Log.e(TAG, "Error closing socket", e);
            } finally {
                socket = null;
                connected = false;
            }
        }
    }

    /**
     * Returns the input stream for reading data from the device.
     *
     * @return The input stream, or null if not connected.
     */
    @Nullable
    public InputStream getInputStream() {
        if (!connected || socket == null) {
            return null;
        }
        try {
            return socket.getInputStream();
        } catch (IOException e) {
            Log.e(TAG, "Error getting input stream", e);
            return null;
        }
    }

    /**
     * Returns the output stream for writing data to the device.
     *
     * @return The output stream, or null if not connected.
     */
    @Nullable
    public OutputStream getOutputStream() {
        if (!connected || socket == null) {
            return null;
        }
        try {
            return socket.getOutputStream();
        } catch (IOException e) {
            Log.e(TAG, "Error getting output stream", e);
            return null;
        }
    }

    /**
     * Returns whether the socket is currently connected.
     *
     * @return true if connected, false otherwise.
     */
    public boolean isConnected() {
        return connected && socket != null && socket.isConnected();
    }

    /**
     * Returns the Bluetooth device associated with this socket.
     *
     * @return The Bluetooth device.
     */
    @NonNull
    public BluetoothDevice getDevice() {
        return device;
    }

    /**
     * Returns the UUID used for this connection.
     *
     * @return The UUID.
     */
    @NonNull
    public UUID getUuid() {
        return uuid;
    }
}
