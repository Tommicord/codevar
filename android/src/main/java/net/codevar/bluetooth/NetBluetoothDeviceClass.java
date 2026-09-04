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

/**
 * Enumeration representing the different types of Bluetooth device classes.
 *
 * <p>This enum categorizes Bluetooth devices based on the technology they support.
 * Each value represents a specific Bluetooth protocol or mode of operation that
 * a device may use for communication.</p>
 *
 * <p><b>Device Class Types:</b></p>
 * <ul>
 *   <li><b>BLE</b> - Bluetooth Low Energy devices, optimized for low power consumption</li>
 *   <li><b>CLASSIC</b> - Classic Bluetooth (BR/EDR) devices, for higher bandwidth applications</li>
 *   <li><b>DUAL</b> - Devices supporting both BLE and Classic Bluetooth simultaneously</li>
 *   <li><b>UNKNOWN</b> - Device type could not be determined or is not recognized</li>
 * </ul>
 *
 * <p><b>Usage Example:</b></p>
 * <pre>{@code
 * NetBluetoothDevice device = new NetBluetoothDevice("Sensor", macAddress, NetBluetoothDeviceClass.BLE);
 * if (device.getClazz() == NetBluetoothDeviceClass.BLE) {
 *     // Handle BLE-specific logic
 * }
 * }</pre>
 */
public enum NetBluetoothDeviceClass {
    /**
     * Bluetooth Low Energy (BLE) device.
     *
     * <p>BLE devices are designed for low power consumption and are commonly used in
     * sensors, wearables, and IoT devices. They operate in the 2.4 GHz ISM band and
     * support short-range wireless communication with minimal energy requirements.</p>
     */
    BLE,

    /**
     * Classic Bluetooth device.
     *
     * <p>Classic Bluetooth (also known as Bluetooth Basic Rate/Enhanced Data Rate - BR/EDR)
     * devices support higher data rates and are used for audio streaming, file transfer,
     * and other bandwidth-intensive applications. This is the traditional Bluetooth
     * technology found in many older devices.</p>
     */
    CLASSIC,

    /**
     * Bluetooth Dual-Mode device.
     *
     * <p>Dual-mode devices support both Bluetooth Low Energy and Classic Bluetooth protocols
     * simultaneously. These devices can communicate with both BLE and Classic Bluetooth devices,
     * providing maximum compatibility. Most modern smartphones and tablets are dual-mode devices.</p>
     */
    DUAL,

    /**
     * Unknown Bluetooth device type.
     *
     * <p>This value is used when the device class cannot be determined or when the device
     * uses a Bluetooth protocol that is not recognized by the system. It serves as a fallback
     * for devices that don't fit into the other categories.</p>
     */
    UNKNOWN
}
