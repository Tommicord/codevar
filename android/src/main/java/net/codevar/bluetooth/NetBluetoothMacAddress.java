package net.codevar.bluetooth;

import androidx.annotation.NonNull;

/**
 * Represents a Bluetooth MAC address with validation and string formatting capabilities.
 *
 * <p>This class provides a type-safe wrapper for Bluetooth MAC addresses, which are 6-byte
 * (48-bit) identifiers used to uniquely identify Bluetooth devices. The class validates
 * input, prevents invalid addresses (such as zero-filled addresses), and provides formatted
 * string representation in the standard colon-separated hexadecimal format (e.g., "5F:2E:3D:7C:4F:1A").</p>
 *
 * <p><b>Key Features:</b></p>
 * <ul>
 *   <li>Immutable storage of 6-byte MAC addresses</li>
 *   <li>Input validation to ensure only valid MAC addresses are accepted</li>
 *   <li>Automatic rejection of zero-filled (00:00:00:00:00:00) addresses</li>
 *   <li>Formatted string output in standard hexadecimal notation</li>
 *   <li>Defensive copying to prevent external modification of internal state</li>
 * </ul>
 *
 * <p><b>Usage Example:</b></p>
 * <pre>{@code
 * // Create from byte array
 * byte[] macBytes = new byte[]{0x5F, 0x2E, 0x3D, 0x7C, 0x4F, 0x1A};
 * NetBluetoothMacAddress mac = new NetBluetoothMacAddress(macBytes);
 * System.out.println(mac.toString()); // Output: "5F:2E:3D:7C:4F:1A"
 *
 * // Get raw bytes (returns a copy)
 * byte[] rawBytes = mac.getBuffer();
 *
 * // Create empty instance (for later population)
 * NetBluetoothMacAddress emptyMac = new NetBluetoothMacAddress();
 * }</pre>
 */
public final class NetBluetoothMacAddress {
    /**
     * The standard length of a Bluetooth MAC address in bytes (6 bytes = 48 bits).
     *
     * <p>This constant is used for validation to ensure that any MAC address provided
     * to the constructor has exactly 6 bytes, which is the IEEE 802 standard for
     * Bluetooth MAC addresses.</p>
     */
    public static final int MACADDR_LENGTH = 6;

    /**
     * Shared printer instance for converting MAC addresses to string format.
     *
     * <p>This static printer is reused across all instances to avoid unnecessary object
     * allocation when converting MAC addresses to their string representation.</p>
     */
    private static final NetBluetoothMacAddressPrinter printer = new NetBluetoothMacAddressPrinter();

    /**
     * Internal buffer storing the 6-byte MAC address.
     *
     * <p>The buffer is final to ensure immutability of the MAC address once constructed.
     * All access to the buffer is through defensive copying to prevent external modification.</p>
     */
    private final byte[] buffer;

    /**
     * Constructs a new NetBluetoothMacAddress with an uninitialized (zero-filled) buffer.
     *
     * <p>This constructor allocates a 6-byte buffer initialized to all zeros. This is useful
     * when the MAC address will be populated later through other means. Note that if this
     * zero-filled address is later used in contexts that validate MAC addresses, it may be
     * rejected as invalid.</p>
     *
     * <p><b>Note:</b> The buffer will contain {0x00, 0x00, 0x00, 0x00, 0x00, 0x00} after construction.</p>
     */
    public NetBluetoothMacAddress() {
        this.buffer = new byte[MACADDR_LENGTH];
    }

    /**
     * Constructs a new NetBluetoothMacAddress from a byte array.
     *
     * <p>This constructor validates the input byte array and creates a new MAC address instance.
     * The input is validated for the following conditions:</p>
     * <ul>
     *   <li>The byte array must not be null</li>
     *   <li>The byte array must have exactly 6 bytes ({@link #MACADDR_LENGTH})</li>
     *   <li>The byte array must not be entirely zero-filled (00:00:00:00:00:00)</li>
     * </ul>
     *
     * <p>The input bytes are copied into an internal buffer, so modifications to the original
     * array will not affect this instance.</p>
     *
     * @param bytes The 6-byte array representing the MAC address. Must not be null and must
     *              have exactly 6 bytes.
     * @throws NetInvalidMacAddr if the byte array is null, has incorrect length, or is
     *                           entirely zero-filled.
     */
    public NetBluetoothMacAddress(byte[] bytes) throws NetInvalidMacAddr {
        if (bytes == null) {
            throw new NetInvalidMacAddr();
        }
        if (bytes.length != MACADDR_LENGTH) {
            throw new NetInvalidMacAddr(
                    "Invalid mac address, " +
                            "the length must be exactly " + MACADDR_LENGTH + " bytes " +
                            "This should not happen or the app is broken!"
            );
        }
        this.buffer = new byte[MACADDR_LENGTH];
        System.arraycopy(bytes, 0, buffer, 0, MACADDR_LENGTH);
        if (isZeroMacAddress()) {
            throw new NetInvalidMacAddr("Invalid mac address");
        }
        printer.setMacAddress(this);
    }

    /**
     * Constructs a new NetBluetoothMacAddress from a string representation.
     *
     * <p>This constructor parses a MAC address string in the standard colon-separated
     * hexadecimal format (e.g., "5F:2E:3D:7C:4F:1A"). The input is validated for the
     * following conditions:</p>
     * <ul>
     *   <li>The string must not be null</li>
     *   <li>The string must be in format "XX:XX:XX:XX:XX:XX" where XX are hex digits</li>
     *   <li>The string must not represent a zero-filled address (00:00:00:00:00:00)</li>
     * </ul>
     *
     * @param macAddress The MAC address string in format "XX:XX:XX:XX:XX:XX".
     * @throws NetInvalidMacAddr if the string is null, has invalid format, or is zero-filled.
     */
    public NetBluetoothMacAddress(String macAddress) throws NetInvalidMacAddr {
        byte[] parsed = parse(macAddress);
        this.buffer = parsed;
        if (isZeroMacAddress()) {
            throw new NetInvalidMacAddr("Invalid mac address");
        }
        printer.setMacAddress(this);
    }

    /**
     * Parses a MAC address string into a byte array.
     *
     * <p>This method parses a MAC address string in the format "XX:XX:XX:XX:XX:XX"
     * where each XX is a two-digit hexadecimal number. The parsing is done without
     * using regular expressions.</p>
     *
     * @param macAddress The MAC address string to parse.
     * @return A 6-byte array containing the parsed MAC address.
     * @throws NetInvalidMacAddr if the string is null, has invalid format, or contains
     *                           invalid hexadecimal characters.
     */
    private byte[] parse(String macAddress) throws NetInvalidMacAddr {
        if (macAddress == null) {
            throw new NetInvalidMacAddr("MAC address string cannot be null");
        }
        if (macAddress.isEmpty()) {
            throw new NetInvalidMacAddr("MAC address string cannot be empty");
        }

        byte[] bytes = new byte[MACADDR_LENGTH];
        int bytePos = 0;
        int charPos = 0;

        for (int i = 0; i < macAddress.length(); i++) {
            char c = macAddress.charAt(i);

            if (c == ':') {
                if (charPos != 2) {
                    throw new NetInvalidMacAddr("Invalid MAC address format: expected 2 hex digits before colon at position " + i);
                }
                charPos = 0;
                bytePos++;
                if (bytePos >= MACADDR_LENGTH) {
                    throw new NetInvalidMacAddr("Invalid MAC address format: too many bytes");
                }
            } else {
                int digit = hexCharToInt(c);
                if (digit == -1) {
                    throw new NetInvalidMacAddr("Invalid MAC address format: invalid hex character '" + c + "' at position " + i);
                }

                if (charPos == 0) {
                    bytes[bytePos] = (byte) (digit << 4);
                    charPos = 1;
                } else if (charPos == 1) {
                    bytes[bytePos] |= (byte) digit;
                    charPos = 2;
                } else {
                    throw new NetInvalidMacAddr("Invalid MAC address format: expected colon at position " + i);
                }
            }
        }

        if (bytePos != MACADDR_LENGTH - 1 || charPos != 2) {
            throw new NetInvalidMacAddr("Invalid MAC address format: incomplete address");
        }

        return bytes;
    }

    /**
     * Converts a hexadecimal character to its integer value.
     *
     * @param c The character to convert.
     * @return The integer value (0-15) of the hex character, or -1 if invalid.
     */
    private int hexCharToInt(char c) {
        if (c >= '0' && c <= '9') {
            return c - '0';
        }
        if (c >= 'A' && c <= 'F') {
            return c - 'A' + 10;
        }
        if (c >= 'a' && c <= 'f') {
            return c - 'a' + 10;
        }
        return -1;
    }

    /**
     * Checks if the MAC address is entirely zero-filled.
     *
     * <p>This method determines whether all 6 bytes of the MAC address are 0x00.
     * A zero-filled MAC address (00:00:00:00:00:00) is considered invalid and is
     * rejected by the constructor.</p>
     *
     * @return true if all bytes in the buffer are 0x00, false otherwise.
     */
    private boolean isZeroMacAddress() {
        boolean allZeros = true;
        for (byte b : this.buffer) {
            if (b != 0) {
                allZeros = false;
                break;
            }
        }
        return allZeros;
    }

    /**
     * Returns a copy of the MAC address bytes.
     *
     * <p>This method returns a defensive copy of the internal buffer to prevent external
     * code from modifying the MAC address stored in this instance. The returned array
     * will always have exactly 6 bytes.</p>
     *
     * @return A new byte array containing a copy of the 6-byte MAC address.
     */
    public byte[] getBuffer() {
        return buffer.clone();
    }

    /**
     * Returns a string representation of the MAC address in standard hexadecimal format.
     *
     * <p>The MAC address is formatted as six two-digit hexadecimal numbers separated by
     * colons, using uppercase letters (e.g., "5F:2E:3D:7C:4F:1A"). This is the standard
     * notation used for displaying MAC addresses.</p>
     *
     * <p>This method is annotated with {@code @NonNull} to guarantee it never returns null.</p>
     *
     * @return A non-null string representing the MAC address in format "XX:XX:XX:XX:XX:XX".
     */
    @NonNull
    @Override
    public String toString() {
        if (printer.getMacAddress() == null) {
            printer.setMacAddress(this);
        }
        return printer.toString();
    }
}
