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
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Async controller for Bluetooth HID input events.
 * <p>
 * This class manages Bluetooth connections, receives input events from HID devices,
 * and distributes them to registered receivers using the NetCommand interface.
 * It runs in a separate background thread and supports multiple concurrent receivers.
 * </p>
 * <p>
 * Key features:
 * <ul>
 *   <li>Key deduplication: Removes key press events from queue when corresponding release arrives</li>
 *   <li>Thread-safe receiver registration</li>
 *   <li>Event queue with configurable capacity</li>
 *   <li>Lifecycle management (start/shutdown)</li>
 *   <li>JNI methods for Rust integration</li>
 * </ul>
 * </p>
 */
public class NetBluetoothHandler {
    private static final String TAG = "Net:BluetoothHandler";

    /**
     * Default queue capacity for input events
     */
    private static final int DEFAULT_QUEUE_CAPACITY = 256;

    /**
     * Maximum number of receivers
     */
    private static final int MAX_RECEIVERS = 128;

    private final NetBluetoothHidManager hidManager;
    private final ExecutorService executorService;
    private final BlockingQueue<NetBluetoothInputEvent> eventQueue;
    private final Set<NetCommand> receivers;
    private final Map<Long, NetCommand> nativeReceivers;
    private final Set<Integer> pressedKeys;

    private final AtomicBoolean isRunning;
    private final AtomicBoolean isProcessing;

    private NetBluetoothHidDevice connectedDevice;
    private Thread processingThread;

    /**
     * Constructs a new NetBluetoothHandler.
     *
     * @param context The Android context.
     */
    public NetBluetoothHandler(@NonNull Context context) {
        this.hidManager = new NetBluetoothHidManager(context);
        this.executorService = Executors.newSingleThreadExecutor();
        this.eventQueue = new ArrayBlockingQueue<>(DEFAULT_QUEUE_CAPACITY);
        this.receivers = ConcurrentHashMap.newKeySet();
        this.nativeReceivers = new ConcurrentHashMap<>();
        this.pressedKeys = ConcurrentHashMap.newKeySet();
        this.isRunning = new AtomicBoolean(false);
        this.isProcessing = new AtomicBoolean(false);
    }

    /**
     * Starts the Bluetooth handler.
     * <p>
     * This initializes the HID manager and begins processing input events.
     * Must be called before connecting to devices or receiving events.
     * </p>
     */
    public void start() {
        if (isRunning.compareAndSet(false, true)) {
            hidManager.setInputListener(new NetBluetoothInputListener() {
                @Override
                public void onInputEvent(NetBluetoothInputEvent event) {
                    if (event != null) {
                        eventQueue.offer(event);
                    }
                }

                @Override
                public void onDeviceConnected(NetBluetoothDevice device) {
                    Log.i(TAG, "Device connected: " + device.getName());
                    connectedDevice = null;
                    if (!hidManager.getConnectedDevices().isEmpty()) {
                        connectedDevice = new NetBluetoothHidDevice(
                                hidManager
                                        .getConnectedDevices()
                                        .iterator()
                                        .next(),
                                NetBluetoothHidDevice.HidDeviceType.UNKNOWN
                        );
                    }
                    startProcessingThread();
                }

                @Override
                public void onDeviceDisconnected(NetBluetoothDevice device) {
                    Log.i(TAG, "Device disconnected: " + device.getName());
                    connectedDevice = null;
                    stopProcessingThread();
                }

                @Override
                public void onError(String error) {
                    Log.e(TAG, "Error: " + error);
                }
            });
            Log.i(TAG, "Bluetooth handler started");
        }
    }

    /**
     * Shuts down the Bluetooth handler.
     * <p>
     * This stops processing events, disconnects all devices, and releases resources.
     * After shutdown, the handler cannot be restarted.
     * </p>
     */
    public void shutdown() {
        if (isRunning.compareAndSet(true, false)) {
            stopProcessingThread();
            hidManager.shutdown();
            executorService.shutdown();
            receivers.clear();
            nativeReceivers.clear();
            pressedKeys.clear();
            eventQueue.clear();
            try {
                if (!executorService.awaitTermination(5, TimeUnit.SECONDS)) {
                    executorService.shutdownNow();
                }
            } catch (InterruptedException e) {
                executorService.shutdownNow();
                Thread.currentThread().interrupt();
            }
            Log.i(TAG, "Bluetooth handler shut down");
        }
    }

    /**
     * Connects to a Bluetooth HID device.
     *
     * @param device     The device to connect to.
     * @param deviceType The type of HID device.
     * @return true if the connection was initiated successfully.
     */
    public boolean connectDevice(@NonNull NetBluetoothDevice device,
                                 @NonNull NetBluetoothHidDevice.HidDeviceType deviceType) {
        if (!isRunning.get()) {
            Log.w(TAG, "Cannot connect: handler not started");
            return false;
        }
        return hidManager.connectDevice(device, deviceType);
    }

    /**
     * Disconnects from a Bluetooth HID device.
     *
     * @param device The device to disconnect.
     * @return true if the disconnection was successful.
     */
    public boolean disconnectDevice(@NonNull NetBluetoothDevice device) {
        return hidManager.disconnectDevice(device);
    }

    /**
     * Registers a command receiver.
     * <p>
     * The receiver will be notified of all input events from connected devices.
     * </p>
     *
     * @param receiver The receiver to register.
     * @return true if the receiver was registered successfully.
     */
    public boolean registerReceiver(@NonNull NetCommand receiver) {
        if (receiver == null) {
            return false;
        }
        if (receivers.size() >= MAX_RECEIVERS) {
            return false;
        }
        return receivers.add(receiver);
    }

    /**
     * Unregisters a command receiver.
     *
     * @param receiver The receiver to unregister.
     * @return true if the receiver was unregistered successfully.
     */
    public boolean unregisterReceiver(@NonNull NetCommand receiver) {
        return receiver != null && receivers.remove(receiver);
    }

    /**
     * Checks if the handler is running.
     *
     * @return true if the handler is running.
     */
    public boolean isRunning() {
        return isRunning.get();
    }

    /**
     * Checks if a device is connected.
     *
     * @return true if a device is connected.
     */
    public boolean isConnected() {
        return connectedDevice != null && connectedDevice.isConnected();
    }

    /**
     * Gets the connected device.
     *
     * @return The connected device, or null if no device is connected.
     */
    @Nullable
    public NetBluetoothDevice getConnectedDevice() {
        return (connectedDevice != null) ? connectedDevice.getDevice() : null;
    }

    /**
     * JNI method: Connects to a device by MAC address.
     * <p>
     * This method is called from Rust code via JNI.
     * </p>
     *
     * @param macAddress The MAC address of the device (hex string, e.g., "5F:2E:3D:7C:4F:1A").
     * @param deviceType The device type (0=KEYBOARD, 1=MOUSE, 2=COMBO, 3=UNKNOWN).
     * @return true if the connection was initiated successfully.
     */
    public boolean nativeConnectDevice(String macAddress, int deviceType) {
        try {
            NetBluetoothMacAddress address = new NetBluetoothMacAddress(macAddress);
            NetBluetoothDevice device = new NetBluetoothDevice("Native Device", address, NetBluetoothDeviceClass.BLE);
            NetBluetoothHidDevice.HidDeviceType type = NetBluetoothHidDevice.HidDeviceType.values()[deviceType];
            return connectDevice(device, type);
        } catch (Exception e) {
            Log.e(TAG, "Failed to connect from native: " + e.getMessage());
            return false;
        }
    }

    /**
     * JNI method: Disconnects from the current device.
     * <p>
     * This method is called from Rust code via JNI.
     * </p>
     *
     * @return true if the disconnection was successful.
     */
    public boolean nativeDisconnectDevice() {
        if (connectedDevice != null) {
            return disconnectDevice(connectedDevice.getDevice());
        }
        return false;
    }

    /**
     * JNI method: Starts the handler.
     * <p>
     * This method is called from Rust code via JNI.
     * </p>
     */
    public void nativeStart() {
        start();
    }

    /**
     * JNI method: Shuts down the handler.
     * <p>
     * This method is called from Rust code via JNI.
     * </p>
     */
    public void nativeShutdown() {
        shutdown();
    }

    /**
     * JNI method: Registers a native receiver callback.
     * <p>
     * This method is called from Rust code via JNI to register a callback
     * that will receive input events. The callback is implemented in Rust.
     * </p>
     * <h3>About the ptr parameter:</h3>
     * <p>
     * The {@code ptr} is a 64-bit pointer (represented as {@code long} in Java)
     * that points to a Rust data structure or callback function. This pointer is:
     * </p>
     * <ul>
     *   <li>Allocated in Rust memory space</li>
     *   <li>Passed from Rust to Java via JNI</li>
     *   <li>Stored by Java and passed back to Rust on each event</li>
     *   <li>Used by Rust to identify which callback to invoke</li>
     * </ul>
     * <p>
     * <b>Important!</b> The pointer value must remain valid for the lifetime of the registration.
     * the Rust code is responsible for managing the memory and freeing it when the receiver is unregistered.
     * </p>
     *
     * @param ptr A 64-bit pointer to a Rust callback structure or function.
     *                    This pointer is opaque to Java - it's passed back to Rust unchanged.
     *                    Must be a valid pointer allocated in Rust memory space.
     */
    public void nativeRegisterReceiver(long ptr) {
        NetCommand nativeReceiver = new NetCommand() {
            @Override
            public void onKeyPressed(int keyCode, int modifiers) {
                nativeOnKeyPressed(ptr, keyCode, modifiers);
            }

            @Override
            public void onKeyReleased(int keyCode, int modifiers) {
                nativeOnKeyReleased(ptr, keyCode, modifiers);
            }

            @Override
            public void onMouseMove(int x, int y) {
                nativeOnMouseMove(ptr, x, y);
            }

            @Override
            public void onButtonPress(int button, int x, int y) {
                nativeOnButtonPress(ptr, button, x, y);
            }

            @Override
            public void onButtonRelease(int button, int x, int y) {
                nativeOnButtonRelease(ptr, button, x, y);
            }

            @Override
            public void onMouseScroll(int delta, int x, int y) {
                nativeOnMouseScroll(ptr, delta, x, y);
            }
        };
        nativeReceivers.put(ptr, nativeReceiver);
        registerReceiver(nativeReceiver);
    }

    /**
     * JNI method: Unregisters the native receiver callback.
     * <p>
     * This method is called from Rust code via JNI to remove the previously
     * registered callback associated with the given pointer. After calling this
     * method, no more events will be sent to that specific Rust callback.
     * </p>
     * <p>
     * <b>Note:</b> Rust should free the callback structure after calling this method.
     * </p>
     *
     * @param ptr The pointer that was used when registering the receiver.
     * @return true if the receiver was found and unregistered, false if no receiver
     *         was registered with this pointer.
     */
    public boolean nativeUnregisterReceiver(long ptr) {
        NetCommand receiver = nativeReceivers.remove(ptr);
        if (receiver != null) {
            unregisterReceiver(receiver);
            return true;
        }
        return false;
    }

    /**
     * JNI callback: Notifies Rust of a key press event.
     * <p>
     * This method is implemented in Rust and called from Java when a key is pressed.
     * </p>
     * <h3>About the ptr parameter:</h3>
     * <p>
     * The {@code ptr} is the same 64-bit pointer that was originally passed to
     * {@link #nativeRegisterReceiver(long)}. It identifies which Rust callback
     * should receive this event. Java does not interpret this value - it simply
     * passes it back to Rust unchanged.
     * </p>
     *
     * @param ptr      The callback pointer originally passed to nativeRegisterReceiver.
     * @param keyCode  The Android key code (e.g., KeyEvent.KEYCODE_A = 29).
     * @param modifiers Modifier flags (bitmask: 1=SHIFT, 2=ALT, 4=CTRL, 8=META).
     */
    private native void nativeOnKeyPressed(long ptr, int keyCode, int modifiers);

    /**
     * JNI callback: Notifies Rust of a key release event.
     * <p>
     * This method is implemented in Rust and called from Java when a key is released.
     * </p>
     *
     * @param ptr      The callback pointer originally passed to nativeRegisterReceiver.
     * @param keyCode  The Android key code (e.g., KeyEvent.KEYCODE_A = 29).
     * @param modifiers Modifier flags (bitmask: 1=SHIFT, 2=ALT, 4=CTRL, 8=META).
     */
    private native void nativeOnKeyReleased(long ptr, int keyCode, int modifiers);

    /**
     * JNI callback: Notifies Rust of a mouse movement event.
     * <p>
     * This method is implemented in Rust and called from Java when the mouse moves.
     * </p>
     *
     * @param ptr The callback pointer originally passed to nativeRegisterReceiver.
     * @param x   The X coordinate (relative to screen origin).
     * @param y   The Y coordinate (relative to screen origin).
     */
    private native void nativeOnMouseMove(long ptr, int x, int y);

    /**
     * JNI callback: Notifies Rust of a mouse button press event.
     * <p>
     * This method is implemented in Rust and called from Java when a mouse button is pressed.
     * </p>
     *
     * @param ptr    The callback pointer originally passed to nativeRegisterReceiver.
     * @param button The button identifier (1=left, 2=middle, 3=right, etc.).
     * @param x      The X coordinate at the time of press.
     * @param y      The Y coordinate at the time of press.
     */
    private native void nativeOnButtonPress(long ptr, int button, int x, int y);

    /**
     * JNI callback: Notifies Rust of a mouse button release event.
     * <p>
     * This method is implemented in Rust and called from Java when a mouse button is released.
     * </p>
     *
     * @param ptr    The callback pointer originally passed to nativeRegisterReceiver.
     * @param button The button identifier (1=left, 2=middle, 3=right, etc.).
     * @param x      The X coordinate at the time of release.
     * @param y      The Y coordinate at the time of release.
     */
    private native void nativeOnButtonRelease(long ptr, int button, int x, int y);

    /**
     * JNI callback: Notifies Rust of a mouse scroll event.
     * <p>
     * This method is implemented in Rust and called from Java when the mouse wheel scrolls.
     * </p>
     *
     * @param ptr   The callback pointer originally passed to nativeRegisterReceiver.
     * @param delta The scroll delta (positive for scroll up, negative for scroll down).
     * @param x     The X coordinate at the time of scroll.
     * @param y     The Y coordinate at the time of scroll.
     */
    private native void nativeOnMouseScroll(long ptr, int delta, int x, int y);

    private void startProcessingThread() {
        if (isProcessing.compareAndSet(false, true)) {
            processingThread = new Thread(this::processEvents);
            processingThread.setName("NetBluetoothHandler-Processing");
            processingThread.setDaemon(true);
            processingThread.start();
        }
    }

    private void stopProcessingThread() {
        if (isProcessing.compareAndSet(true, false)) {
            if (processingThread != null) {
                processingThread.interrupt();
                processingThread = null;
            }
        }
    }

    private void processEvents() {
        Log.i(TAG, "Event processing thread started");

        while (isProcessing.get() && !Thread.currentThread().isInterrupted()) {
            try {
                NetBluetoothInputEvent event = eventQueue.poll(64, TimeUnit.MILLISECONDS);
                if (event != null) {
                    handleEvent(event);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        Log.i(TAG, "Event processing thread stopped");
    }

    private void handleEvent(NetBluetoothInputEvent event) {
        if (event.getType() == NetBluetoothInputEvent.InputType.KEYBOARD) {
            NetBluetoothInputEvent.KeyEvent keyEvent = event.getKeyEvent();
            if (keyEvent != null) {
                handleKeyboardEvent(event.getKeyEvent());
            }
        } else if (event.getType() == NetBluetoothInputEvent.InputType.MOUSE) {
            NetBluetoothInputEvent.MouseEvent mouseEvent = event.getMouseEvent();
            if (mouseEvent != null) {
                handleMouseEvent(mouseEvent);
            }
        }
    }

    private void handleKeyboardEvent(NetBluetoothInputEvent.KeyEvent keyEvent) {
        int keyCode = keyEvent.keyCode();
        int modifiers = keyEvent.modifiers();

        if (keyEvent.action() == NetBluetoothInputEvent.KeyEvent.KeyAction.PRESS) {
            // Add to pressed keys set
            pressedKeys.add(keyCode);
            // Notify all receivers
            for (NetCommand receiver : receivers) {
                try {
                    receiver.onKeyPressed(keyCode, modifiers);
                } catch (Exception e) {
                    Log.e(TAG, "Error in receiver onKeyPressed", e);
                }
            }
        } else if (keyEvent.action() == NetBluetoothInputEvent.KeyEvent.KeyAction.RELEASE) {
            // Remove from pressed keys set (deduplication)
            pressedKeys.remove(keyCode);
            // Notify all receivers
            for (NetCommand receiver : receivers) {
                try {
                    receiver.onKeyReleased(keyCode, modifiers);
                } catch (Exception e) {
                    Log.e(TAG, "Error in receiver onKeyReleased", e);
                }
            }
        }
    }

    private void handleMouseEvent(NetBluetoothInputEvent.MouseEvent mouseEvent) {
        int x = mouseEvent.x();
        int y = mouseEvent.y();
        int buttons = mouseEvent.buttons();
        int scrollDelta = mouseEvent.scrollDelta();
        NetBluetoothInputEvent.MouseEvent.MouseEventType eventType = mouseEvent.eventType();

        switch (eventType) {
            case MOVE:
                for (NetCommand receiver : receivers) {
                    try {
                        receiver.onMouseMove(x, y);
                    } catch (Exception e) {
                        Log.e(TAG, "Error in receiver onMouseMove", e);
                    }
                }
                break;
            case BUTTON_PRESS:
                for (NetCommand receiver : receivers) {
                    try {
                        receiver.onButtonPress(buttons, x, y);
                    } catch (Exception e) {
                        Log.e(TAG, "Error in receiver onButtonPress", e);
                    }
                }
                break;
            case BUTTON_RELEASE:
                for (NetCommand receiver : receivers) {
                    try {
                        receiver.onButtonRelease(buttons, x, y);
                    } catch (Exception e) {
                        Log.e(TAG, "Error in receiver onButtonRelease", e);
                    }
                }
                break;
            case SCROLL:
                for (NetCommand receiver : receivers) {
                    try {
                        receiver.onMouseScroll(scrollDelta, x, y);
                    } catch (Exception e) {
                        Log.e(TAG, "Error in receiver onMouseScroll", e);
                    }
                }
                break;
        }
    }
}
