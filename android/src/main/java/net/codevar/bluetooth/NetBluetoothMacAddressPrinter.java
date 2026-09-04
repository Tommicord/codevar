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
 * Converts Bluetooth MAC addresses to their string representation in hexadecimal format.
 *
 * <p>This class provides functionality to format 6-byte MAC addresses into the standard
 * colon-separated hexadecimal notation (e.g., "5F:2E:3D:7C:4F:1A"). It maintains a reference
 * to a {@link NetBluetoothMacAddress} object and can convert it to a string on demand.
 * The printer can be reused for multiple MAC addresses by setting different addresses.</p>
 *
 * <p><b>Key Features:</b></p>
 * <ul>
 *   <li>Converts 6-byte MAC addresses to standard "XX:XX:XX:XX:XX:XX" format</li>
 *   <li>Uses uppercase hexadecimal letters (A-F)</li>
 *   <li>Provides a fallback address (FF:FF:FF:FF:FF:FF) when no MAC address is set</li>
 *   <li>Reusable for multiple MAC addresses through setter method</li>
 *   <li>Efficient character array-based conversion without string concatenation</li>
 * </ul>
 *
 * <p><b>Usage Example:</b></p>
 * <pre>{@code
 * // Create printer with initial MAC address
 * NetBluetoothMacAddress mac = new NetBluetoothMacAddress(new byte[]{0x5F, 0x2E, 0x3D, 0x7C, 0x4F, 0x1A});
 * NetBluetoothMacAddressPrinter printer = new NetBluetoothMacAddressPrinter(mac);
 * System.out.println(printer.toString()); // Output: "5F:2E:3D:7C:4F:1A"
 *
 * // Change to a different MAC address
 * NetBluetoothMacAddress mac2 = new NetBluetoothMacAddress(new byte[]{0x01, 0x02, 0x03, 0x04, 0x05, 0x06});
 * printer.setMacAddress(mac2);
 * System.out.println(printer.toString()); // Output: "01:02:03:04:05:06"
 *
 * // Create empty printer (will return fallback address)
 * NetBluetoothMacAddressPrinter emptyPrinter = new NetBluetoothMacAddressPrinter();
 * System.out.println(emptyPrinter.toString()); // Output: "FF:FF:FF:FF:FF:FF"
 * }</pre>
 */
public class NetBluetoothMacAddressPrinter {
    /**
     * The MAC address object to be converted to string format.
     *
     * <p>This field holds a reference to the {@link NetBluetoothMacAddress} that will be
     * formatted when {@link #toString()} is called. It can be null, in which case a fallback
     * address (FF:FF:FF:FF:FF:FF) is returned.</p>
     */
    private NetBluetoothMacAddress macAddress;

    /**
     * Creates a new MAC address printer without an initial MAC address.
     *
     * <p>This constructor initializes the printer with no MAC address set. Calling
     * {@link #toString()} on this instance will return the fallback address "FF:FF:FF:FF:FF:FF".
     * Use {@link #setMacAddress(NetBluetoothMacAddress)} to set a MAC address before conversion.</p>
     */
    public NetBluetoothMacAddressPrinter() {
    }

    /**
     * Creates a new MAC address printer with an initial MAC address.
     *
     * <p>This constructor initializes the printer with the specified MAC address. The address
     * can be changed later using {@link #setMacAddress(NetBluetoothMacAddress)}.</p>
     *
     * @param macAddress The MAC address to be formatted. Can be null, in which case
     *                   {@link #toString()} will return the fallback address.
     */
    public NetBluetoothMacAddressPrinter(NetBluetoothMacAddress macAddress) {
        this.macAddress = macAddress;
    }

    /**
     * Returns the currently stored MAC address object.
     *
     * <p>This method returns the reference to the {@link NetBluetoothMacAddress} object that
     * is currently set for printing. The returned object may be null if no address has been set.</p>
     *
     * @return The stored {@link NetBluetoothMacAddress} object, or null if no address is set.
     */
    public NetBluetoothMacAddress getMacAddress() {
        return macAddress;
    }

    /**
     * Sets a new MAC address for printing.
     *
     * <p>This method updates the MAC address that will be formatted when {@link #toString()}
     * is called. The printer can be reused for multiple MAC addresses by calling this method
     * with different addresses.</p>
     *
     * @param other The new MAC address to set. Can be null, in which case {@link #toString()}
     *              will return the fallback address.
     */
    public void setMacAddress(NetBluetoothMacAddress other) {
        this.macAddress = other;
    }

    /**
     * Generates a byte array containing hexadecimal character codes (0-9, A-F).
     *
     * <p>This private method creates a lookup table for hexadecimal digit characters.
     * The array contains the ASCII codes for characters '0' through '9' (0x30-0x39) and
     * 'A' through 'F' (0x41-0x46). This is used to efficiently convert byte values to
     * their hexadecimal character representations.</p>
     *
     * <p><b>Output:</b> A 16-byte array with values: {0x30, 0x31, ..., 0x39, 0x41, 0x42, ..., 0x46}</p>
     *
     * @return A byte array containing ASCII codes for hexadecimal characters 0-9 and A-F.
     */
    private byte[] getHexArray() {
        byte[] hexArray = new byte[16]; // Allocate space for 0123456789ABCDEF
        for (int i = 0; i < 10; ++i) {
            hexArray[i] = (byte) (0x30 + i);
        }
        for (int i = 10; i < 16; ++i) {
            hexArray[i] = (byte) (0x41 + (i - 10));
        }
        return hexArray;
    }

    /**
     * Converts a 6-byte buffer to a colon-separated hexadecimal string.
     *
     * <p>This private method performs the actual conversion of a byte array to a formatted
     * MAC address string. Each byte is converted to two hexadecimal characters, and colons
     * are inserted between each byte pair. The conversion uses bitwise operations to extract
     * the high and low nibbles of each byte.</p>
     *
     * <p><b>Algorithm:</b></p>
     * <ul>
     *   <li>For each byte, extract the high nibble (bits 4-7) using {@code v >>> 4}</li>
     *   <li>For each byte, extract the low nibble (bits 0-3) using {@code v & 0x0F}</li>
     *   <li>Map each nibble to its hexadecimal character using the lookup table</li>
     *   <li>Insert colon separators between byte pairs</li>
     * </ul>
     *
     * @param buffer The 6-byte array to convert. Must not be null and must have exactly 6 bytes.
     * @return A string in format "XX:XX:XX:XX:XX:XX" where each XX is a two-digit uppercase hexadecimal number.
     * @throws NetInvalidMacAddr if the buffer is null or does not have exactly 6 bytes.
     */
    private String getStringify(byte[] buffer) {
        if (buffer == null || buffer.length != 6) {
            throw new NetInvalidMacAddr("The printer cannot output");
        }
        byte[] hexArray = getHexArray();
        int capacity = buffer.length * 3 - 1;
        char[] hexBuffer = new char[capacity];

        for (int i = 0; i < buffer.length; ++i) {
            int v = buffer[i] & 0xFF;
            hexBuffer[i * 3] = (char) hexArray[v >>> 4];
            hexBuffer[i * 3 + 1] = (char) hexArray[v & 0x0F];
            if (i < buffer.length - 1) {
                hexBuffer[i * 3 + 2] = ':';
            }
        }
        return new String(hexBuffer);
    }

    /**
     * Returns a fallback MAC address string when no MAC address is set.
     *
     * <p>This private method generates the string "FF:FF:FF:FF:FF:FF", which is used
     * as a fallback when the printer's MAC address is null. This represents a broadcast
     * MAC address in networking contexts.</p>
     *
     * @return The string "FF:FF:FF:FF:FF:FF".
     */
    private String getFallbackString() {
        return getStringify(new byte[]{(byte) 0xFF, (byte) 0xFF, (byte) 0xFF, (byte) 0xFF, (byte) 0xFF, (byte) 0xFF});
    }

    /**
     * Returns the string representation of the MAC address in standard hexadecimal format.
     *
     * <p>This method converts the stored MAC address to a string in the format "XX:XX:XX:XX:XX:XX",
     * where each XX is a two-digit uppercase hexadecimal number. If no MAC address is set
     * (i.e., {@link #macAddress} is null), this method returns the fallback address "FF:FF:FF:FF:FF:FF".</p>
     *
     * <p><b>Examples:</b></p>
     * <ul>
     *   <li>Bytes {0x5F, 0x2E, 0x3D, 0x7C, 0x4F, 0x1A} → "5F:2E:3D:7C:4F:1A"</li>
     *   <li>Bytes {0x00, 0x01, 0x02, 0x03, 0x04, 0x05} → "00:01:02:03:04:05"</li>
     *   <li>null → "FF:FF:FF:FF:FF:FF" (fallback)</li>
     * </ul>
     *
     * <p>This method is annotated with {@code @NonNull} to guarantee it never returns null.</p>
     *
     * @return A non-null string representing the MAC address in format "XX:XX:XX:XX:XX:XX",
     * or "FF:FF:FF:FF:FF:FF" if no MAC address is set.
     */
    @NonNull
    @Override
    public String toString() {
        if (macAddress == null) {
            return getFallbackString();
        }
        return getStringify(macAddress.getBuffer());
    }
}
