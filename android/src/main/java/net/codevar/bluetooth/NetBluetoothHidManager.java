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
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Manager class for Bluetooth HID devices (keyboards and mice).
 *
 * <p>This class manages the lifecycle of Bluetooth HID devices, handles device connections,
 * reads input reports, parses them into input events, and forwards events to registered listeners.
 * It runs input reading on a background thread and delivers events on the main thread.</p>
 *
 * <p><b>Usage Example:</b></p>
 * <pre>{@code
 * NetBluetoothHidManager manager = new NetBluetoothHidManager(context);
 * manager.setInputListener(new NetBluetoothInputListener() {
 *     @Override
 *     public void onInputEvent(NetBluetoothInputEvent event) {
 *         // Handle input event
 *     }
 *
 *     @Override
 *     public void onDeviceConnected(NetBluetoothDevice device) {
 *         // Handle device connection
 *     }
 *
 *     @Override
 *     public void onDeviceDisconnected(NetBluetoothDevice device) {
 *         // Handle device disconnection
 *     }
 *
 *     @Override
 *     public void onError(String error) {
 *         // Handle errors
 *     }
 * });
 *
 * // Connect to a device
 * manager.connectDevice(device, NetBluetoothHidDevice.HidDeviceType.KEYBOARD);
 * }</pre>
 */
public class NetBluetoothHidManager {
    private static final String TAG = "Net:HidManager";

    private final Context context;
    private final Handler mainHandler;
    private final ExecutorService executorService;
    private final Map<String, NetBluetoothHidDevice> connectedDevices;
    private NetBluetoothInputListener inputListener;
    private final NetBluetoothHidReportParser reportParser;
    private volatile boolean running;

    /**
     * Constructs a new NetBluetoothHidManager.
     *
     * @param context The Android context. Must not be null.
     */
    public NetBluetoothHidManager(@NonNull Context context) {
        this.context = context.getApplicationContext();
        this.mainHandler = new Handler(Looper.getMainLooper());
        this.executorService = Executors.newCachedThreadPool();
        this.connectedDevices = new HashMap<>();
        this.reportParser = new NetBluetoothHidReportParser();
        this.running = true;
    }

    /**
     * Sets the input listener for receiving input events.
     *
     * @param listener The listener to set. Can be null to remove the listener.
     */
    public void setInputListener(@Nullable NetBluetoothInputListener listener) {
        this.inputListener = listener;
    }

    /**
     * Connects to a Bluetooth HID device.
     *
     * @param device     The device to connect to. Must not be null.
     * @param deviceType The type of HID device. Must not be null.
     * @return true if the connection was initiated successfully, false otherwise.
     */
    public boolean connectDevice(@NonNull NetBluetoothDevice device,
                                 @NonNull NetBluetoothHidDevice.HidDeviceType deviceType) {
        if (device == null) {
            Log.e(TAG, "Device cannot be null");
            return false;
        }

        String macAddress = device.getAddress().toString();
        if (connectedDevices.containsKey(macAddress)) {
            Log.w(TAG, "Device already connected: " + macAddress);
            return false;
        }

        NetBluetoothHidDevice hidDevice = new NetBluetoothHidDevice(device, deviceType);

        executorService.submit(() -> {
            if (hidDevice.connect()) {
                synchronized (connectedDevices) {
                    connectedDevices.put(macAddress, hidDevice);
                }
                notifyDeviceConnected(device);
                startReadingFromDevice(hidDevice);
            } else {
                notifyError("Failed to connect to device: " + device.getName());
            }
        });

        return true;
    }

    /**
     * Disconnects from a Bluetooth HID device.
     *
     * @param device The device to disconnect. Must not be null.
     * @return true if the device was connected and disconnected, false otherwise.
     */
    public boolean disconnectDevice(@NonNull NetBluetoothDevice device) {
        if (device == null) {
            return false;
        }

        String macAddress = device.getAddress().toString();
        NetBluetoothHidDevice hidDevice;

        synchronized (connectedDevices) {
            hidDevice = connectedDevices.remove(macAddress);
        }

        if (hidDevice != null) {
            hidDevice.disconnect();
            notifyDeviceDisconnected(device);
            return true;
        }

        return false;
    }

    /**
     * Disconnects all connected devices.
     */
    public void disconnectAll() {
        List<NetBluetoothHidDevice> devices;
        synchronized (connectedDevices) {
            devices = new ArrayList<>(connectedDevices.values());
            connectedDevices.clear();
        }

        for (NetBluetoothHidDevice device : devices) {
            device.disconnect();
            notifyDeviceDisconnected(device.getDevice());
        }
    }

    /**
     * Returns a list of all connected devices.
     *
     * @return A list of connected devices.
     */
    @NonNull
    public List<NetBluetoothDevice> getConnectedDevices() {
        List<NetBluetoothDevice> devices = new ArrayList<>();
        synchronized (connectedDevices) {
            for (NetBluetoothHidDevice hidDevice : connectedDevices.values()) {
                devices.add(hidDevice.getDevice());
            }
        }
        return devices;
    }

    /**
     * Checks if a device is currently connected.
     *
     * @param device The device to check. Must not be null.
     * @return true if the device is connected, false otherwise.
     */
    public boolean isDeviceConnected(@NonNull NetBluetoothDevice device) {
        if (device != null) {
            String macAddress = device.getAddress().toString();
            synchronized (connectedDevices) {
                return connectedDevices.containsKey(macAddress);
            }
        }
        return false;
    }

    /**
     * Starts reading input reports from a device.
     *
     * @param hidDevice The HID device to read from. Must not be null.
     */
    private void startReadingFromDevice(@NonNull NetBluetoothHidDevice hidDevice) {
        executorService.submit(() -> {
            while (running && hidDevice.isConnected()) {
                try {
                    byte[] report = hidDevice.readReport();
                    if (report != null) {
                        NetBluetoothInputEvent event = reportParser.parseReport(report, hidDevice.getDeviceType());
                        if (event != null) {
                            notifyInputEvent(event);
                        }
                    } else {
                        // Read failed, device may be disconnected
                        break;
                    }
                } catch (Exception e) {
                    Log.e(TAG, "Error reading from device", e);
                    break;
                }
            }

            // Device disconnected or error occurred
            if (hidDevice.isConnected()) {
                disconnectDevice(hidDevice.getDevice());
            }
        });
    }

    /**
     * Notifies the listener of an input event.
     *
     * @param event The input event. Must not be null.
     */
    private void notifyInputEvent(@NonNull NetBluetoothInputEvent event) {
        if (inputListener != null) {
            mainHandler.post(() -> inputListener.onInputEvent(event));
        }
    }

    /**
     * Notifies the listener of a device connection.
     *
     * @param device The connected device. Must not be null.
     */
    private void notifyDeviceConnected(@NonNull NetBluetoothDevice device) {
        if (inputListener != null) {
            mainHandler.post(() -> inputListener.onDeviceConnected(device));
        }
    }

    /**
     * Notifies the listener of a device disconnection.
     *
     * @param device The disconnected device. Must not be null.
     */
    private void notifyDeviceDisconnected(@NonNull NetBluetoothDevice device) {
        if (inputListener != null) {
            mainHandler.post(() -> inputListener.onDeviceDisconnected(device));
        }
    }

    /**
     * Notifies the listener of an error.
     *
     * @param error The error message. Must not be null.
     */
    private void notifyError(@NonNull String error) {
        if (inputListener != null) {
            mainHandler.post(() -> inputListener.onError(error));
        }
    }

    /**
     * Shuts down the manager and releases all resources.
     */
    public void shutdown() {
        running = false;
        disconnectAll();
        executorService.shutdown();
    }

    /**
     * Returns whether the manager is currently running.
     *
     * @return true if running, false otherwise.
     */
    public boolean isRunning() {
        return running;
    }
}
