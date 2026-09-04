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

import androidx.annotation.NonNull;

/**
 * Listener interface for receiving Bluetooth input events.
 *
 * <p>This interface defines callbacks for receiving keyboard and mouse input events
 * from Bluetooth HID devices. Implement this interface to handle input events in your
 * application.</p>
 *
 * <p><b>Usage Example:</b></p>
 * <pre>{@code
 * NetBluetoothHidManager manager = new NetBluetoothHidManager(context);
 * manager.setInputListener(new NetBluetoothInputListener() {
 *     @Override
 *     public void onInputEvent(NetBluetoothInputEvent event) {
 *         if (event.getType() == NetBluetoothInputEvent.InputType.KEYBOARD) {
 *             handleKeyboardEvent(event.getKeyEvent());
 *         } else if (event.getType() == NetBluetoothInputEvent.InputType.MOUSE) {
 *             handleMouseEvent(event.getMouseEvent());
 *         }
 *     }
 *
 *     @Override
 *     public void onDeviceConnected(NetBluetoothDevice device) {
 *         Log.d("Input", "Device connected: " + device.getName());
 *     }
 *
 *     @Override
 *     public void onDeviceDisconnected(NetBluetoothDevice device) {
 *         Log.d("Input", "Device disconnected: " + device.getName());
 *     }
 * });
 * }</pre>
 */
public interface NetBluetoothInputListener {

    /**
     * Called when an input event is received from a Bluetooth HID device.
     *
     * @param event The input event. Will not be null.
     */
    void onInputEvent(@NonNull NetBluetoothInputEvent event);

    /**
     * Called when a Bluetooth HID device is connected.
     *
     * @param device The connected device. Will not be null.
     */
    void onDeviceConnected(@NonNull NetBluetoothDevice device);

    /**
     * Called when a Bluetooth HID device is disconnected.
     *
     * @param device The disconnected device. Will not be null.
     */
    void onDeviceDisconnected(@NonNull NetBluetoothDevice device);

    /**
     * Called when an error occurs during input event processing.
     *
     * @param error The error message describing what went wrong.
     */
    void onError(@NonNull String error);
}
