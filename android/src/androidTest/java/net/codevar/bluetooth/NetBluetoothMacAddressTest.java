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

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import org.junit.Test;
import org.junit.runner.RunWith;

import static org.junit.Assert.*;

/**
 * Comprehensive unit tests for NetBluetoothMacAddress.
 * <p>
 * These tests verify the parsing of MAC address strings, validation,
 * and byte array construction functionality.
 * </p>
 */
@RunWith(AndroidJUnit4.class)
public class NetBluetoothMacAddressTest {

    @Test
    public void testConstructorFromByteArray() throws NetInvalidMacAddr {
        byte[] macBytes = new byte[]{0x5F, 0x2E, 0x3D, 0x7C, 0x4F, 0x1A};
        NetBluetoothMacAddress mac = new NetBluetoothMacAddress(macBytes);
        assertNotNull(mac);
        assertArrayEquals(macBytes, mac.getBuffer());
    }

    @Test
    public void testConstructorFromByteArrayNull() {
        try {
            new NetBluetoothMacAddress((byte[]) null);
            fail("Should throw NetInvalidMacAddr for null byte array");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testConstructorFromByteArrayWrongLength() {
        try {
            new NetBluetoothMacAddress(new byte[]{0x5F, 0x2E});
            fail("Should throw NetInvalidMacAddr for wrong length");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testConstructorFromByteArrayZeroFilled() {
        try {
            new NetBluetoothMacAddress(new byte[]{0x00, 0x00, 0x00, 0x00, 0x00, 0x00});
            fail("Should throw NetInvalidMacAddr for zero-filled address");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testConstructorDefault() {
        NetBluetoothMacAddress mac = new NetBluetoothMacAddress();
        assertNotNull(mac);
        assertEquals(NetBluetoothMacAddress.MACADDR_LENGTH, mac.getBuffer().length);
    }

    @Test
    public void testConstructorFromStringValid() throws NetInvalidMacAddr {
        NetBluetoothMacAddress mac = new NetBluetoothMacAddress("5F:2E:3D:7C:4F:1A");
        assertNotNull(mac);
        byte[] expected = new byte[]{0x5F, 0x2E, 0x3D, 0x7C, 0x4F, 0x1A};
        assertArrayEquals(expected, mac.getBuffer());
    }

    @Test
    public void testConstructorFromStringUppercase() throws NetInvalidMacAddr {
        NetBluetoothMacAddress mac = new NetBluetoothMacAddress("AA:BB:CC:DD:EE:FF");
        assertNotNull(mac);
        byte[] expected = new byte[]{(byte) 0xAA, (byte) 0xBB, (byte) 0xCC, (byte) 0xDD, (byte) 0xEE, (byte) 0xFF};
        assertArrayEquals(expected, mac.getBuffer());
    }

    @Test
    public void testConstructorFromStringLowercase() throws NetInvalidMacAddr {
        NetBluetoothMacAddress mac = new NetBluetoothMacAddress("aa:bb:cc:dd:ee:ff");
        assertNotNull(mac);
        byte[] expected = new byte[]{(byte) 0xAA, (byte) 0xBB, (byte) 0xCC, (byte) 0xDD, (byte) 0xEE, (byte) 0xFF};
        assertArrayEquals(expected, mac.getBuffer());
    }

    @Test
    public void testConstructorFromStringMixedCase() throws NetInvalidMacAddr {
        NetBluetoothMacAddress mac = new NetBluetoothMacAddress("Aa:Bb:Cc:Dd:Ee:Ff");
        assertNotNull(mac);
        byte[] expected = new byte[]{(byte) 0xAA, (byte) 0xBB, (byte) 0xCC, (byte) 0xDD, (byte) 0xEE, (byte) 0xFF};
        assertArrayEquals(expected, mac.getBuffer());
    }

    @Test
    public void testConstructorFromStringNull() {
        try {
            new NetBluetoothMacAddress((String) null);
            fail("Should throw NetInvalidMacAddr for null string");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testConstructorFromStringEmpty() {
        try {
            new NetBluetoothMacAddress("");
            fail("Should throw NetInvalidMacAddr for empty string");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testConstructorFromStringTooShort() {
        try {
            new NetBluetoothMacAddress("00:11:22:33:44");
            fail("Should throw NetInvalidMacAddr for too short string");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testConstructorFromStringTooLong() {
        try {
            new NetBluetoothMacAddress("00:11:22:33:44:55:66");
            fail("Should throw NetInvalidMacAddr for too long string");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testConstructorFromStringWrongSeparator() {
        try {
            new NetBluetoothMacAddress("00-11-22-33-44-55");
            fail("Should throw NetInvalidMacAddr for wrong separator");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testConstructorFromStringNoSeparator() {
        try {
            new NetBluetoothMacAddress("001122334455");
            fail("Should throw NetInvalidMacAddr for no separator");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testConstructorFromStringInvalidHexChar() {
        try {
            new NetBluetoothMacAddress("00:11:22:33:44:GG");
            fail("Should throw NetInvalidMacAddr for invalid hex character");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testConstructorFromStringInvalidHexCharSpace() {
        try {
            new NetBluetoothMacAddress("00:11:22:33:44:5 ");
            fail("Should throw NetInvalidMacAddr for space character");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testConstructorFromStringMissingColon() {
        try {
            new NetBluetoothMacAddress("00:11:22:33:4455");
            fail("Should throw NetInvalidMacAddr for missing colon");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testConstructorFromStringExtraColon() {
        try {
            new NetBluetoothMacAddress("00:11:22:33:44:55:");
            fail("Should throw NetInvalidMacAddr for extra colon");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testConstructorFromStringSingleDigit() {
        try {
            new NetBluetoothMacAddress("0:11:22:33:44:55");
            fail("Should throw NetInvalidMacAddr for single digit");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testConstructorFromStringTripleDigit() {
        try {
            new NetBluetoothMacAddress("000:11:22:33:44:55");
            fail("Should throw NetInvalidMacAddr for triple digit");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testConstructorFromStringZeroFilled() {
        try {
            new NetBluetoothMacAddress("00:00:00:00:00:00");
            fail("Should throw NetInvalidMacAddr for zero-filled address");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testConstructorFromStringAllZerosExceptOne() throws NetInvalidMacAddr {
        NetBluetoothMacAddress mac = new NetBluetoothMacAddress("00:00:00:00:00:01");
        assertNotNull(mac);
        byte[] expected = new byte[]{0x00, 0x00, 0x00, 0x00, 0x00, 0x01};
        assertArrayEquals(expected, mac.getBuffer());
    }

    @Test
    public void testConstructorFromStringAllOnes() throws NetInvalidMacAddr {
        NetBluetoothMacAddress mac = new NetBluetoothMacAddress("FF:FF:FF:FF:FF:FF");
        assertNotNull(mac);
        byte[] expected = new byte[]{(byte) 0xFF, (byte) 0xFF, (byte) 0xFF, (byte) 0xFF, (byte) 0xFF, (byte) 0xFF};
        assertArrayEquals(expected, mac.getBuffer());
    }

    @Test
    public void testConstructorFromStringWithLeadingZeros() throws NetInvalidMacAddr {
        NetBluetoothMacAddress mac = new NetBluetoothMacAddress("00:01:02:03:04:05");
        assertNotNull(mac);
        byte[] expected = new byte[]{0x00, 0x01, 0x02, 0x03, 0x04, 0x05};
        assertArrayEquals(expected, mac.getBuffer());
    }

    @Test
    public void testGetBufferReturnsCopy() throws NetInvalidMacAddr {
        byte[] original = new byte[]{0x5F, 0x2E, 0x3D, 0x7C, 0x4F, 0x1A};
        NetBluetoothMacAddress mac = new NetBluetoothMacAddress(original);
        byte[] copy = mac.getBuffer();

        // Modify the copy
        copy[0] = 0x00;

        // Original should be unchanged
        assertArrayEquals(original, mac.getBuffer());
    }

    @Test
    public void testToString() throws NetInvalidMacAddr {
        NetBluetoothMacAddress mac = new NetBluetoothMacAddress(new byte[]{0x5F, 0x2E, 0x3D, 0x7C, 0x4F, 0x1A});
        String str = mac.toString();
        assertNotNull(str);
        assertEquals("5F:2E:3D:7C:4F:1A", str);
    }

    @Test
    public void testToStringUppercase() throws NetInvalidMacAddr {
        NetBluetoothMacAddress mac = new NetBluetoothMacAddress(new byte[]{(byte) 0xAA, (byte) 0xBB, (byte) 0xCC, (byte) 0xDD, (byte) 0xEE, (byte) 0xFF});
        String str = mac.toString();
        assertNotNull(str);
        assertEquals("AA:BB:CC:DD:EE:FF", str);
    }

    @Test
    public void testToStringLowercaseBytes() throws NetInvalidMacAddr {
        NetBluetoothMacAddress mac = new NetBluetoothMacAddress(new byte[]{0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f});
        String str = mac.toString();
        assertNotNull(str);
        assertEquals("0A:0B:0C:0D:0E:0F", str);
    }

    @Test
    public void testParseAllDigits() throws NetInvalidMacAddr {
        NetBluetoothMacAddress mac = new NetBluetoothMacAddress("12:34:56:78:90:AB");
        byte[] expected = new byte[]{0x12, 0x34, 0x56, 0x78, (byte) 0x90, (byte) 0xAB};
        assertArrayEquals(expected, mac.getBuffer());
    }

    @Test
    public void testParseBoundaryValues() throws NetInvalidMacAddr {
        NetBluetoothMacAddress mac = new NetBluetoothMacAddress("00:FF:00:FF:00:FF");
        byte[] expected = new byte[]{0x00, (byte) 0xFF, 0x00, (byte) 0xFF, 0x00, (byte) 0xFF};
        assertArrayEquals(expected, mac.getBuffer());
    }

    @Test
    public void testParseWithSpaces() {
        try {
            new NetBluetoothMacAddress("00:11:22:33:44:55 ");
            fail("Should throw NetInvalidMacAddr for trailing space");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testParseWithLeadingSpace() {
        try {
            new NetBluetoothMacAddress(" 00:11:22:33:44:55");
            fail("Should throw NetInvalidMacAddr for leading space");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testMacAddrLengthConstant() {
        assertEquals(6, NetBluetoothMacAddress.MACADDR_LENGTH);
    }

    @Test
    public void testConstructorFromStringMultipleColons() {
        try {
            new NetBluetoothMacAddress("00::11:22:33:44:55");
            fail("Should throw NetInvalidMacAddr for multiple colons");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testConstructorFromStringOnlyColons() {
        try {
            new NetBluetoothMacAddress("::::::");
            fail("Should throw NetInvalidMacAddr for only colons");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testConstructorFromStringSpecialChars() {
        try {
            new NetBluetoothMacAddress("00:11:22:33:44:5@");
            fail("Should throw NetInvalidMacAddr for special character");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testConstructorFromStringUnicodeChars() {
        try {
            new NetBluetoothMacAddress("00:11:22:33:44:5α");
            fail("Should throw NetInvalidMacAddr for unicode character");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testConstructorFromStringTabSeparator() {
        try {
            new NetBluetoothMacAddress("00\t11\t22\t33\t44\t55");
            fail("Should throw NetInvalidMacAddr for tab separator");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testConstructorFromStringDotSeparator() {
        try {
            new NetBluetoothMacAddress("00.11.22.33.44.55");
            fail("Should throw NetInvalidMacAddr for dot separator");
        } catch (NetInvalidMacAddr e) {
            // Expected
        }
    }

    @Test
    public void testRoundTripStringToBytesToString() throws NetInvalidMacAddr {
        String original = "5F:2E:3D:7C:4F:1A";
        NetBluetoothMacAddress mac = new NetBluetoothMacAddress(original);
        String result = mac.toString();
        assertEquals(original, result);
    }

    @Test
    public void testRoundTripBytesToStringToBytes() throws NetInvalidMacAddr {
        byte[] original = new byte[]{0x5F, 0x2E, 0x3D, 0x7C, 0x4F, 0x1A};
        NetBluetoothMacAddress mac = new NetBluetoothMacAddress(original);
        String str = mac.toString();
        NetBluetoothMacAddress mac2 = new NetBluetoothMacAddress(str);
        assertArrayEquals(original, mac2.getBuffer());
    }
}
