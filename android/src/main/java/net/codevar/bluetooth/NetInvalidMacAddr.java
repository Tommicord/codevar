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
 * Exception thrown when an invalid Bluetooth MAC address is encountered.
 *
 * <p>This runtime exception is used throughout the Bluetooth package to indicate
 * errors related to MAC address validation and processing. It is thrown when
 * attempting to create or use a MAC address that does not meet the required
 * specifications.</p>
 *
 * <p><b>Common Causes:</b></p>
 * <ul>
 *   <li>Providing a null byte array to {@link NetBluetoothMacAddress#NetBluetoothMacAddress(byte[])}</li>
 *   <li>Providing a byte array with incorrect length (not exactly 6 bytes)</li>
 *   <li>Providing a zero-filled MAC address (00:00:00:00:00:00)</li>
 *   <li>Attempting to format an invalid or null buffer in {@link NetBluetoothMacAddressPrinter}</li>
 * </ul>
 *
 * <p><b>Usage Example:</b></p>
 * <pre>{@code
 * try {
 *     byte[] invalidMac = new byte[]{0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
 *     NetBluetoothMacAddress mac = new NetBluetoothMacAddress(invalidMac);
 * } catch (NetInvalidMacAddr e) {
 *     // Handle the invalid MAC address error
 *     System.err.println("Invalid MAC address: " + e.getMessage());
 * }
 *
 * try {
 *     byte[] wrongLength = new byte[]{0x5F, 0x2E, 0x3D};
 *     NetBluetoothMacAddress mac = new NetBluetoothMacAddress(wrongLength);
 * } catch (NetInvalidMacAddr e) {
 *     // Handle the length validation error
 *     System.err.println("MAC address has wrong length: " + e.getMessage());
 * }
 * }</pre>
 *
 * <p>This exception extends {@link RuntimeException}, meaning it is an unchecked exception.
 * Callers are not required to catch it, but should handle it appropriately when working
 * with MAC addresses from untrusted sources.</p>
 */
public class NetInvalidMacAddr extends RuntimeException {
    /**
     * Constructs a new NetInvalidMacAddr with no detail message.
     *
     * <p>This constructor creates an exception with no specific error message.
     * It is typically used when the reason for the invalid MAC address is self-evident
     * from the context (e.g., a null input).</p>
     */
    public NetInvalidMacAddr() {
        super();
    }

    /**
     * Constructs a new NetInvalidMacAddr with the specified detail message.
     *
     * <p>This constructor creates an exception with a descriptive error message
     * explaining why the MAC address is invalid. The message can provide specific
     * details about the validation failure.</p>
     *
     * @param message The detail message explaining the cause of the exception.
     *                This message can be retrieved later via {@link Throwable#getMessage()}.
     */
    public NetInvalidMacAddr(String message) {
        super(message);
    }

    /**
     * Constructs a new NetInvalidMacAddr with the specified detail message and cause.
     *
     * <p>This constructor creates an exception with both a descriptive error message
     * and a reference to the underlying cause. This is useful when the invalid MAC address
     * error is the result of another exception or error condition.</p>
     *
     * @param message The detail message explaining the cause of the exception.
     *                This message can be retrieved later via {@link Throwable#getMessage()}.
     * @param cause   The underlying cause of the exception. This can be retrieved later via
     *                {@link Throwable#getCause()}. A null value is permitted and indicates
     *                that the cause is nonexistent or unknown.
     */
    public NetInvalidMacAddr(String message, Throwable cause) {
        super(message, cause);
    }
}
