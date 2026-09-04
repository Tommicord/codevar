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

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;

/**
 * A byte stream wrapper for Bluetooth socket communication.
 *
 * <p>This class provides buffered read and write operations for Bluetooth communication,
 * with support for reading specific byte counts, reading until a delimiter, and
 * writing byte arrays with error handling.</p>
 *
 * <p><b>Usage Example:</b></p>
 * <pre>{@code
 * NetBluetoothSocket socket = new NetBluetoothSocket(device, uuid);
 * if (socket.connect()) {
 *     NetBluetoothByteStream stream = new NetBluetoothByteStream(socket);
 *     byte[] data = stream.readBytes(64);
 *     stream.writeBytes(response);
 *     socket.disconnect();
 * }
 * }</pre>
 */
public class NetBluetoothByteStream {
    private static final String TAG = "Net:ByteStream";
    private static final int DEFAULT_BUFFER_SIZE = 1024;
    private static final int READ_TIMEOUT_MS = 5000;

    private final NetBluetoothSocket socket;
    private final byte[] readBuffer;
    private InputStream inputStream;
    private OutputStream outputStream;

    /**
     * Constructs a new NetBluetoothByteStream for the given socket.
     *
     * @param socket The Bluetooth socket to wrap. Must be connected. Must not be null.
     * @throws IllegalStateException if the socket is not connected.
     */
    public NetBluetoothByteStream(@NonNull NetBluetoothSocket socket) {
        if (!socket.isConnected()) {
            throw new IllegalStateException("Socket must be connected");
        }
        this.socket = socket;
        this.inputStream = socket.getInputStream();
        this.outputStream = socket.getOutputStream();
        this.readBuffer = new byte[DEFAULT_BUFFER_SIZE];
    }

    /**
     * Reads a specific number of bytes from the stream.
     *
     * <p>This method blocks until the specified number of bytes are read or an error occurs.</p>
     *
     * @param count The number of bytes to read. Must be positive.
     * @return The bytes read, or null if an error occurs.
     */
    @Nullable
    public byte[] readBytes(int count) {
        if (count <= 0) {
            Log.e(TAG, "Invalid byte count: " + count);
            return null;
        }

        if (inputStream == null) {
            Log.e(TAG, "Input stream is null");
            return null;
        }

        try {
            byte[] buffer = new byte[count];
            int totalRead = readWithTimeout(inputStream, buffer, 0, count, READ_TIMEOUT_MS);
            if (totalRead == count) {
                return buffer;
            } else {
                Log.w(TAG, "Incomplete read: expected " + count + ", got " + totalRead);
                return Arrays.copyOf(buffer, totalRead);
            }
        } catch (IOException e) {
            Log.e(TAG, "Error reading bytes", e);
            return null;
        }
    }

    /**
     * Reads bytes from the stream until a delimiter is found.
     *
     * @param delimiter The delimiter byte to search for.
     * @return The bytes read including the delimiter, or null if an error occurs.
     */
    @Nullable
    public byte[] readUntil(byte delimiter) {
        if (inputStream == null) {
            Log.e(TAG, "Input stream is null");
            return null;
        }

        try {
            java.io.ByteArrayOutputStream buffer = new java.io.ByteArrayOutputStream();
            int b;
            while ((b = inputStream.read()) != -1) {
                buffer.write(b);
                if (b == delimiter) {
                    break;
                }
            }
            return buffer.toByteArray();
        } catch (IOException e) {
            Log.e(TAG, "Error reading until delimiter", e);
            return null;
        }
    }

    /**
     * Reads all available bytes from the stream without blocking.
     *
     * @return The bytes read, or an empty array if no bytes are available.
     */
    @NonNull
    public byte[] readAvailable() {
        if (inputStream == null) {
            Log.e(TAG, "Input stream is null");
            return new byte[0];
        }

        try {
            int available = inputStream.available();
            if (available <= 0) {
                return new byte[0];
            }
            byte[] buffer = new byte[available];
            int read = inputStream.read(buffer);
            if (read > 0) {
                return Arrays.copyOf(buffer, read);
            }
            return new byte[0];
        } catch (IOException e) {
            Log.e(TAG, "Error reading available bytes", e);
            return new byte[0];
        }
    }

    /**
     * Writes bytes to the stream.
     *
     * @param data The bytes to write. Must not be null.
     * @return true if the write was successful, false otherwise.
     */
    public boolean writeBytes(@NonNull byte[] data) {
        if (outputStream == null) {
            Log.e(TAG, "Output stream is null");
            return false;
        }

        try {
            outputStream.write(data);
            outputStream.flush();
            return true;
        } catch (IOException e) {
            Log.e(TAG, "Error writing bytes", e);
            return false;
        }
    }

    /**
     * Writes a single byte to the stream.
     *
     * @param data The byte to write.
     * @return true if the write was successful, false otherwise.
     */
    public boolean writeByte(byte data) {
        return writeBytes(new byte[]{data});
    }

    /**
     * Closes the stream and releases resources.
     */
    public void close() {
        try {
            if (inputStream != null) {
                inputStream.close();
            }
            if (outputStream != null) {
                outputStream.close();
            }
        } catch (IOException e) {
            Log.e(TAG, "Error closing streams", e);
        } finally {
            // Nullify both input and output streams
            // This prevents the function isOpen returning
            // true when the byte stream is closed
            inputStream = null;
            outputStream = null;
        }
    }

    /**
     * Returns whether the stream is open and ready for communication.
     *
     * @return true if the stream is open, false otherwise.
     */
    public boolean isOpen() {
        return socket.isConnected() && inputStream != null && outputStream != null;
    }

    /**
     * Helper method to read bytes with timeout.
     */
    private int readWithTimeout(InputStream stream, byte[] buffer, int offset, int count, long timeoutMs)
            throws IOException {
        long startTime = System.currentTimeMillis();
        int totalRead = 0;

        while (totalRead < count) {
            int available = stream.available();
            if (available > 0) {
                int read = stream.read(buffer, offset + totalRead, Math.min(available, count - totalRead));
                if (read == -1) {
                    break;
                }
                totalRead += read;
            } else {
                if (System.currentTimeMillis() - startTime > timeoutMs) {
                    throw new IOException("Read timeout");
                }
                try {
                    Thread.sleep(10);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IOException("Read interrupted", e);
                }
            }
        }
        return totalRead;
    }
}
