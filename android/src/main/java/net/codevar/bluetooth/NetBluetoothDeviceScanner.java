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

import android.Manifest;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothManager;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.ScanFilter;
import android.bluetooth.le.ScanResult;
import android.bluetooth.le.ScanSettings;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A Bluetooth device scanner supporting both Classic and BLE device discovery.
 *
 * <p>This class provides comprehensive Bluetooth scanning capabilities with the following features:</p>
 * <ul>
 *   <li>Support for both Classic Bluetooth and BLE (Bluetooth Low Energy) scanning</li>
 *   <li>Configurable scan duration and automatic stopping</li>
 *   <li>Device filtering by name, MAC address, service UUIDs, and device class</li>
 *   <li>Callback-based architecture for device discovery events</li>
 *   <li>Automatic permission checking and validation</li>
 *   <li>Thread-safe operations with proper state management</li>
 *   <li>Duplicate device detection and deduplication</li>
 *   <li>Signal strength (RSSI) monitoring and filtering</li>
 *   <li>Integration with existing {@link NetBluetoothDevice} and {@link NetBluetoothMacAddress} classes</li>
 * </ul>
 *
 * <p><b>Usage Example:</b></p>
 * <pre>{@code
 * // Create scanner with context
 * NetBluetoothDeviceScanner scanner = new NetBluetoothDeviceScanner(context);
 *
 * // Set scan callback
 * scanner.setScanCallback(new NetBluetoothDeviceScanner.DeviceScanCallback() {
 *     @Override
 *     public void onDeviceDiscovered(NetBluetoothDevice device, int rssi) {
 *         Log.d("Bluetooth", "Found device: " + device.getName() + " at " + device.getAddress());
 *     }
 *
 *     @Override
 *     public void onScanComplete(List<NetBluetoothDevice> devices) {
 *         Log.d("Bluetooth", "Scan complete. Found " + devices.size() + " devices");
 *     }
 *
 *     @Override
 *     public void onScanFailed(ScanError error) {
 *         Log.e("Bluetooth", "Scan failed: " + error);
 *     }
 * });
 *
 * // Start BLE scan for 10 seconds
 * scanner.startLeScan(10000);
 *
 * // Or start Classic Bluetooth scan
 * scanner.startClassicScan(10000);
 * }</pre>
 *
 * <p><b>Thread Safety:</b> All public methods are thread-safe and can be called from any thread.
 * Callbacks are delivered on the main thread (UI thread) for safe UI updates.</p>
 *
 * <p><b>Permissions:</b> Requires BLUETOOTH_SCAN, BLUETOOTH_CONNECT, and ACCESS_FINE_LOCATION
 * permissions for Android 12+. For earlier versions, requires BLUETOOTH and ACCESS_FINE_LOCATION.</p>
 */
public class NetBluetoothDeviceScanner {
    /**
     * Default scan duration in milliseconds (10 seconds).
     */
    public static final long DEFAULT_SCAN_DURATION = 10000;
    /**
     * Minimum Android API level for BLE scanning (API 18).
     */
    public static final int MIN_BLE_API_LEVEL = Build.VERSION_CODES.JELLY_BEAN_MR2;
    /**
     * Minimum Android API level for BLE scanning with filters (API 21).
     */
    public static final int MIN_BLE_FILTER_API_LEVEL = Build.VERSION_CODES.LOLLIPOP;
    private static final String TAG = "Net:DeviceScanner";
    private final Context context;
    private final Handler mainHandler;
    private final AtomicBoolean isScanning;
    private final List<NetBluetoothDevice> discoveredDevices;
    private final ScanFilterBuilder filterBuilder;
    private BluetoothAdapter bluetoothAdapter;
    private BluetoothLeScanner bleScanner;
    private android.bluetooth.le.ScanCallback bleScanCallback;
    private BluetoothAdapter.LeScanCallback legacyLeScanCallback;
    private DeviceScanCallback userScanCallback;
    private Runnable scanTimeoutRunnable;

    /**
     * Constructs a new NetBluetoothDeviceScanner.
     *
     * @param context The Android context. Must not be null. Application context will be used
     *                to avoid memory leaks.
     * @throws IllegalArgumentException if context is null
     */
    public NetBluetoothDeviceScanner(@NonNull Context context) {
        if (context == null) {
            throw new IllegalArgumentException("Context cannot be null");
        }

        this.context = context.getApplicationContext();
        this.mainHandler = new Handler(Looper.getMainLooper());
        this.isScanning = new AtomicBoolean(false);
        this.discoveredDevices = new ArrayList<>();
        this.filterBuilder = new ScanFilterBuilder();

        initializeBluetooth();
    }

    /**
     * Initializes the Bluetooth adapter.
     */
    private void initializeBluetooth() {
        BluetoothManager bluetoothManager = (BluetoothManager) context.getSystemService(Context.BLUETOOTH_SERVICE);
        if (bluetoothManager != null) {
            this.bluetoothAdapter = bluetoothManager.getAdapter();
        }

        if (bluetoothAdapter != null && Build.VERSION.SDK_INT >= MIN_BLE_FILTER_API_LEVEL) {
            this.bleScanner = bluetoothAdapter.getBluetoothLeScanner();
        }
    }

    /**
     * Checks if Bluetooth is available and enabled on the device.
     *
     * @return true if Bluetooth is available and enabled, false otherwise.
     */
    public boolean isBluetoothAvailable() {
        return bluetoothAdapter != null && bluetoothAdapter.isEnabled();
    }

    /**
     * Checks if the required Bluetooth permissions are granted.
     *
     * @return true if all required permissions are granted, false otherwise.
     */
    public boolean hasRequiredPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            return context.checkSelfPermission(Manifest.permission.BLUETOOTH_SCAN) == PackageManager.PERMISSION_GRANTED
                    && context.checkSelfPermission(Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED
                    && context.checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED;
        } else {
            return context.checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED;
        }
    }

    /**
     * Checks if BLE (Bluetooth Low Energy) is supported on this device.
     *
     * @return true if BLE is supported, false otherwise.
     */
    public boolean isBleSupported() {
        return context.getPackageManager().hasSystemFeature(PackageManager.FEATURE_BLUETOOTH_LE);
    }

    /**
     * Sets the callback for scan events.
     *
     * @param callback The callback to receive scan events. Can be null to remove existing callback.
     */
    public void setScanCallback(@Nullable DeviceScanCallback callback) {
        this.userScanCallback = callback;
    }

    /**
     * Gets the current scan filter builder.
     *
     * @return The ScanFilterBuilder for configuring scan filters.
     */
    @NonNull
    public ScanFilterBuilder getFilterBuilder() {
        return filterBuilder;
    }

    /**
     * Starts a BLE (Bluetooth Low Energy) scan with the default duration.
     *
     * <p>This method will start scanning for BLE devices and automatically stop after
     * {@link #DEFAULT_SCAN_DURATION} milliseconds. Use {@link #stopScan()} to stop manually.</p>
     *
     * @return true if the scan was started successfully, false otherwise.
     * @throws SecurityException     if required permissions are not granted
     * @throws IllegalStateException if Bluetooth is not available or already scanning
     */
    public boolean startLeScan() {
        return startLeScan(DEFAULT_SCAN_DURATION);
    }

    /**
     * Starts a BLE (Bluetooth Low Energy) scan with a specified duration.
     *
     * <p>This method will start scanning for BLE devices and automatically stop after
     * the specified duration. Use {@link #stopScan()} to stop manually.</p>
     *
     * @param durationMs The scan duration in milliseconds. Must be positive.
     * @return true if the scan was started successfully, false otherwise.
     * @throws SecurityException        if required permissions are not granted
     * @throws IllegalStateException    if Bluetooth is not available or already scanning
     * @throws IllegalArgumentException if durationMs is not positive
     */
    public boolean startLeScan(long durationMs) {
        if (durationMs <= 0) {
            throw new IllegalArgumentException("Scan duration must be positive");
        }

        if (!hasRequiredPermissions()) {
            notifyScanFailed(ScanError.PERMISSION_DENIED);
            throw new SecurityException("Required Bluetooth permissions not granted");
        }

        if (!isBluetoothAvailable()) {
            notifyScanFailed(ScanError.BLUETOOTH_NOT_AVAILABLE);
            return false;
        }

        if (!isBleSupported()) {
            notifyScanFailed(ScanError.BLE_NOT_SUPPORTED);
            return false;
        }

        if (isScanning.getAndSet(true)) {
            notifyScanFailed(ScanError.ALREADY_SCANNING);
            return false;
        }

        discoveredDevices.clear();

        if (Build.VERSION.SDK_INT >= MIN_BLE_FILTER_API_LEVEL) {
            return startLeScanLollipop(durationMs);
        } else if (Build.VERSION.SDK_INT >= MIN_BLE_API_LEVEL) {
            return startLeScanLegacy(durationMs);
        } else {
            notifyScanFailed(ScanError.BLE_NOT_SUPPORTED);
            return false;
        }
    }

    /**
     * Starts BLE scan using the modern API (API 21+).
     */
    @RequiresApi(Build.VERSION_CODES.LOLLIPOP)
    private boolean startLeScanLollipop(long durationMs) {
        if (bleScanner == null) {
            notifyScanFailed(ScanError.BLE_NOT_SUPPORTED);
            return false;
        }

        try {
            ScanSettings settings = new ScanSettings.Builder()
                    .setScanMode(ScanSettings.SCAN_MODE_BALANCED)
                    .setCallbackType(ScanSettings.CALLBACK_TYPE_ALL_MATCHES)
                    .setMatchMode(ScanSettings.MATCH_MODE_STICKY)
                    .setReportDelay(0)
                    .build();

            List<ScanFilter> filters = filterBuilder.buildScanFilters();

            bleScanCallback = new android.bluetooth.le.ScanCallback() {
                @Override
                public void onScanResult(int callbackType, ScanResult result) {
                    handleLeScanResult(result);
                }

                @Override
                public void onBatchScanResults(List<ScanResult> results) {
                    for (ScanResult result : results) {
                        handleLeScanResult(result);
                    }
                }

                @Override
                public void onScanFailed(int errorCode) {
                    handleScanFailure(errorCode);
                }
            };

            bleScanner.startScan(filters, settings, bleScanCallback);
            scheduleScanTimeout(durationMs);
            return true;

        } catch (Exception e) {
            Log.e(TAG, "Failed to start BLE scan", e);
            isScanning.set(false);
            notifyScanFailed(ScanError.INTERNAL_ERROR);
            return false;
        }
    }

    /**
     * Starts BLE scan using the legacy API (API 18-20).
     */
    private boolean startLeScanLegacy(long durationMs) {
        if (bluetoothAdapter == null) {
            notifyScanFailed(ScanError.BLE_NOT_SUPPORTED);
            return false;
        }

        try {
            legacyLeScanCallback = new BluetoothAdapter.LeScanCallback() {
                @Override
                public void onLeScan(BluetoothDevice device, int rssi, byte[] scanRecord) {
                    handleLegacyLeScanResult(device, rssi, scanRecord);
                }
            };

            boolean started = bluetoothAdapter.startLeScan(legacyLeScanCallback);
            if (started) {
                scheduleScanTimeout(durationMs);
            } else {
                isScanning.set(false);
                notifyScanFailed(ScanError.INTERNAL_ERROR);
            }
            return started;

        } catch (Exception e) {
            Log.e(TAG, "Failed to start legacy BLE scan", e);
            isScanning.set(false);
            notifyScanFailed(ScanError.INTERNAL_ERROR);
            return false;
        }
    }

    /**
     * Handles a BLE scan result from the modern API.
     */
    @RequiresApi(Build.VERSION_CODES.LOLLIPOP)
    private void handleLeScanResult(ScanResult result) {
        BluetoothDevice device = result.getDevice();
        int rssi = result.getRssi();

        if (!filterBuilder.matchesFilters(device, rssi)) {
            return;
        }

        NetBluetoothDevice netDevice = createNetBluetoothDevice(device, NetBluetoothDeviceClass.BLE);
        if (netDevice != null) {
            netDevice.setRssi(rssi);
            netDevice.updateLastSeen();
            registerOrUpdateDevice(netDevice);
        }

        notifyDeviceDiscovered(netDevice, rssi);
    }

    /**
     * Handles a BLE scan result from the legacy API.
     */
    private void handleLegacyLeScanResult(BluetoothDevice device, int rssi, byte[] scanRecord) {
        if (!filterBuilder.matchesFilters(device, rssi)) {
            return;
        }

        NetBluetoothDevice netDevice = createNetBluetoothDevice(device, NetBluetoothDeviceClass.BLE);
        if (netDevice != null) {
            netDevice.setRssi(rssi);
            netDevice.updateLastSeen();
            registerOrUpdateDevice(netDevice);
        }

        notifyDeviceDiscovered(netDevice, rssi);
    }

    /**
     * Starts a Classic Bluetooth scan with the default duration.
     *
     * <p>Classic Bluetooth scanning uses the discovery API which is slower and less
     * efficient than BLE scanning. It's recommended to use BLE scanning when possible.</p>
     *
     * @return true if the scan was started successfully, false otherwise.
     * @throws SecurityException     if required permissions are not granted
     * @throws IllegalStateException if Bluetooth is not available or already scanning
     */
    public boolean startClassicScan() {
        return startClassicScan(DEFAULT_SCAN_DURATION);
    }

    /**
     * Starts a Classic Bluetooth scan with a specified duration.
     *
     * <p>Classic Bluetooth scanning uses the discovery API which is slower and less
     * efficient than BLE scanning. It's recommended to use BLE scanning when possible.</p>
     *
     * @param durationMs The scan duration in milliseconds. Must be positive.
     * @return true if the scan was started successfully, false otherwise.
     * @throws SecurityException        if required permissions are not granted
     * @throws IllegalStateException    if Bluetooth is not available or already scanning
     * @throws IllegalArgumentException if durationMs is not positive
     */
    public boolean startClassicScan(long durationMs) {
        if (durationMs <= 0) {
            throw new IllegalArgumentException("Scan duration must be positive");
        }

        if (!hasRequiredPermissions()) {
            notifyScanFailed(ScanError.PERMISSION_DENIED);
            throw new SecurityException("Required Bluetooth permissions not granted");
        }

        if (!isBluetoothAvailable()) {
            notifyScanFailed(ScanError.BLUETOOTH_NOT_AVAILABLE);
            return false;
        }

        if (isScanning.getAndSet(true)) {
            notifyScanFailed(ScanError.ALREADY_SCANNING);
            return false;
        }

        discoveredDevices.clear();

        try {
            boolean started = bluetoothAdapter.startDiscovery();
            if (started) {
                scheduleScanTimeout(durationMs);
            } else {
                isScanning.set(false);
                notifyScanFailed(ScanError.INTERNAL_ERROR);
            }
            return started;

        } catch (Exception e) {
            Log.e(TAG, "Failed to start Classic Bluetooth scan", e);
            isScanning.set(false);
            notifyScanFailed(ScanError.INTERNAL_ERROR);
            return false;
        }
    }

    /**
     * Stops any ongoing Bluetooth scan.
     *
     * <p>This method can be called from any thread and will safely stop any ongoing
     * BLE or Classic Bluetooth scan. If no scan is in progress, this method does nothing.</p>
     */
    public void stopScan() {
        if (!isScanning.getAndSet(false)) {
            return;
        }
        if (scanTimeoutRunnable != null) {
            mainHandler.removeCallbacks(scanTimeoutRunnable);
            scanTimeoutRunnable = null;
        }

        // Stop BLE scan
        if (Build.VERSION.SDK_INT >= MIN_BLE_FILTER_API_LEVEL && bleScanner != null && bleScanCallback != null) {
            try {
                bleScanner.stopScan(bleScanCallback);
            } catch (Exception e) {
                Log.e(TAG, "Error stopping BLE scan", e);
            }
            bleScanCallback = null;
        }

        // Stop legacy BLE scan
        if (Build.VERSION.SDK_INT >= MIN_BLE_API_LEVEL && bluetoothAdapter != null && legacyLeScanCallback != null) {
            try {
                bluetoothAdapter.stopLeScan(legacyLeScanCallback);
            } catch (Exception e) {
                Log.e(TAG, "Error stopping legacy BLE scan", e);
            }
            legacyLeScanCallback = null;
        }

        // Stop Classic Bluetooth discovery
        if (bluetoothAdapter != null && bluetoothAdapter.isDiscovering()) {
            try {
                bluetoothAdapter.cancelDiscovery();
            } catch (Exception e) {
                Log.e(TAG, "Error stopping Classic Bluetooth discovery", e);
            }
        }

        // Notify completion
        synchronized (discoveredDevices) {
            notifyScanComplete(new ArrayList<>(discoveredDevices));
        }
    }

    /**
     * Schedules a timeout for the current scan.
     */
    private void scheduleScanTimeout(long durationMs) {
        scanTimeoutRunnable = () -> {
            Log.d(TAG, "Scan timeout reached");
            stopScan();
        };
        mainHandler.postDelayed(scanTimeoutRunnable, durationMs);
    }

    /**
     * Handles scan failure from the modern BLE API.
     */
    @RequiresApi(Build.VERSION_CODES.LOLLIPOP)
    private void handleScanFailure(int errorCode) {
        ScanError error = switch (errorCode) {
            case android.bluetooth.le.ScanCallback.SCAN_FAILED_ALREADY_STARTED ->
                    ScanError.ALREADY_SCANNING;
            case android.bluetooth.le.ScanCallback.SCAN_FAILED_APPLICATION_REGISTRATION_FAILED ->
                    ScanError.INTERNAL_ERROR;
            case android.bluetooth.le.ScanCallback.SCAN_FAILED_FEATURE_UNSUPPORTED ->
                    ScanError.BLE_NOT_SUPPORTED;
            case android.bluetooth.le.ScanCallback.SCAN_FAILED_INTERNAL_ERROR ->
                    ScanError.INTERNAL_ERROR;
            default -> ScanError.INTERNAL_ERROR;
        };
        isScanning.set(false);
        notifyScanFailed(error);
    }

    /**
     * Creates a NetBluetoothDevice from a BluetoothDevice.
     */
    private NetBluetoothDevice createNetBluetoothDevice(BluetoothDevice device, NetBluetoothDeviceClass deviceClass) {
        try {
            byte[] macBytes = getMacAddressBytes(device.getAddress());
            NetBluetoothMacAddress macAddress = new NetBluetoothMacAddress(macBytes);
            return new NetBluetoothDevice(device.getName(), macAddress, deviceClass);
        } catch (NetInvalidMacAddr e) {
            Log.e(TAG, "Invalid MAC address for device: " + device.getAddress(), e);
            return null;
        }
    }

    /**
     * Registers a new device or updates an existing device in the discovered devices list.
     *
     * <p>This method checks if a device with the same MAC address already exists in the list.
     * If it does, it updates the existing device's properties (RSSI, last seen timestamp).
     * If it doesn't, it adds the new device to the list.</p>
     *
     * @param device The device to register or update. Must not be null.
     */
    private void registerOrUpdateDevice(NetBluetoothDevice device) {
        if (device == null) {
            return;
        }

        synchronized (discoveredDevices) {
            for (int i = 0; i < discoveredDevices.size(); i++) {
                NetBluetoothDevice existingDevice = discoveredDevices.get(i);
                if (existingDevice.matches(device)) {
                    // Update existing device
                    existingDevice.setRssi(device.getRssi());
                    existingDevice.updateLastSeen();
                    if (device.getName() != null && !device.getName().isEmpty()) {
                        existingDevice.setName(device.getName());
                    }
                    return;
                }
            }
            // Add new device
            discoveredDevices.add(device);
        }
    }

    /**
     * Finds a device by its MAC address in the discovered devices list.
     *
     * @param macAddress The MAC address to search for. Can be null.
     * @return The device if found, null otherwise.
     */
    @Nullable
    public NetBluetoothDevice findDeviceByMacAddress(String macAddress) {
        if (macAddress == null) {
            return null;
        }

        synchronized (discoveredDevices) {
            for (NetBluetoothDevice device : discoveredDevices) {
                if (device.hasMacAddress(macAddress)) {
                    return device;
                }
            }
        }
        return null;
    }

    /**
     * Removes a device from the discovered devices list by its MAC address.
     *
     * @param macAddress The MAC address of the device to remove. Can be null.
     * @return true if the device was found and removed, false otherwise.
     */
    public boolean removeDeviceByMacAddress(String macAddress) {
        if (macAddress == null) {
            return false;
        }

        synchronized (discoveredDevices) {
            for (int i = 0; i < discoveredDevices.size(); i++) {
                NetBluetoothDevice device = discoveredDevices.get(i);
                if (device.hasMacAddress(macAddress)) {
                    discoveredDevices.remove(i);
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Removes stale devices that haven't been seen for a specified duration.
     *
     * @param maxAgeMs The maximum age in milliseconds. Devices older than this will be removed.
     * @return The number of devices removed.
     */
    public int removeStaleDevices(long maxAgeMs) {
        int removedCount = 0;
        long currentTime = System.currentTimeMillis();

        synchronized (discoveredDevices) {
            for (int i = discoveredDevices.size() - 1; i >= 0; i--) {
                NetBluetoothDevice device = discoveredDevices.get(i);
                if (currentTime - device.getLastSeenTimestamp() > maxAgeMs) {
                    discoveredDevices.remove(i);
                    removedCount++;
                }
            }
        }
        return removedCount;
    }

    /**
     * Gets the number of discovered devices.
     *
     * @return The number of devices currently in the discovered devices list.
     */
    public int getDiscoveredDeviceCount() {
        synchronized (discoveredDevices) {
            return discoveredDevices.size();
        }
    }

    /**
     * Converts a MAC address string to a byte array.
     */
    private byte[] getMacAddressBytes(String macAddress) {
        String[] parts = macAddress.split(":");
        byte[] bytes = new byte[6];
        for (int i = 0; i < parts.length; i++) {
            bytes[i] = (byte) Integer.parseInt(parts[i], 16);
        }
        return bytes;
    }

    /**
     * Notifies the callback that a device was discovered.
     */
    private void notifyDeviceDiscovered(NetBluetoothDevice device, int rssi) {
        if (device != null && userScanCallback != null) {
            mainHandler.post(() -> userScanCallback.onDeviceDiscovered(device, rssi));
        }
    }

    /**
     * Notifies the callback that the scan is complete.
     */
    private void notifyScanComplete(List<NetBluetoothDevice> devices) {
        if (userScanCallback != null) {
            mainHandler.post(() -> userScanCallback.onScanComplete(devices));
        }
    }

    /**
     * Notifies the callback that the scan failed.
     */
    private void notifyScanFailed(ScanError error) {
        if (userScanCallback != null) {
            mainHandler.post(() -> userScanCallback.onScanFailed(error));
        }
    }

    /**
     * Gets the list of discovered devices from the last scan.
     *
     * @return A new list containing all discovered devices. Returns an empty list if no scan has been performed.
     */
    @NonNull
    public List<NetBluetoothDevice> getDiscoveredDevices() {
        synchronized (discoveredDevices) {
            return new ArrayList<>(discoveredDevices);
        }
    }

    /**
     * Clears the list of discovered devices.
     */
    public void clearDiscoveredDevices() {
        synchronized (discoveredDevices) {
            discoveredDevices.clear();
        }
    }

    /**
     * Checks if a scan is currently in progress.
     *
     * @return true if scanning, false otherwise.
     */
    public boolean isScanning() {
        return isScanning.get();
    }

    /**
     * Enumeration of possible scan errors.
     */
    public enum ScanError {
        /**
         * Bluetooth is not available or not enabled on the device
         */
        BLUETOOTH_NOT_AVAILABLE,

        /**
         * BLE is not supported on this device
         */
        BLE_NOT_SUPPORTED,

        /**
         * Required permissions are not granted
         */
        PERMISSION_DENIED,

        /**
         * A scan is already in progress
         */
        ALREADY_SCANNING,

        /**
         * An internal error occurred during scanning
         */
        INTERNAL_ERROR
    }

    /**
     * Callback interface for Bluetooth scan events.
     */
    public interface DeviceScanCallback {
        /**
         * Called when a Bluetooth device is discovered during scanning.
         *
         * @param device The discovered device. Will not be null.
         * @param rssi   The signal strength in dBm. Values typically range from -100 to 0.
         */
        void onDeviceDiscovered(@NonNull NetBluetoothDevice device, int rssi);

        /**
         * Called when the scan completes successfully.
         *
         * @param devices A list of all discovered devices during the scan. Will not be null.
         */
        void onScanComplete(@NonNull List<NetBluetoothDevice> devices);

        /**
         * Called when the scan fails due to an error.
         *
         * @param error The error that caused the scan failure. Will not be null.
         */
        void onScanFailed(@NonNull ScanError error);
    }

    /**
     * Builder class for configuring scan filters.
     */
    public static class ScanFilterBuilder {
        private final List<UUID> serviceUuids = new ArrayList<>();
        private String deviceNameFilter;
        private String macAddressFilter;
        private int minRssi = Integer.MIN_VALUE;
        private int maxRssi = Integer.MAX_VALUE;

        /**
         * Sets a filter for device name.
         *
         * @param name The device name to filter for. Can be null to remove the filter.
         *             Supports partial matching (contains).
         * @return This builder for method chaining.
         */
        @NonNull
        public ScanFilterBuilder setDeviceName(@Nullable String name) {
            this.deviceNameFilter = name;
            return this;
        }

        /**
         * Sets a filter for MAC address.
         *
         * @param macAddress The MAC address to filter for. Can be null to remove the filter.
         *                   Must be in format "XX:XX:XX:XX:XX:XX".
         * @return This builder for method chaining.
         */
        @NonNull
        public ScanFilterBuilder setMacAddress(@Nullable String macAddress) {
            this.macAddressFilter = macAddress;
            return this;
        }

        /**
         * Sets the minimum RSSI (signal strength) filter.
         *
         * @param rssi The minimum RSSI in dBm. Only devices with RSSI >= this value will be reported.
         *             Use Integer.MIN_VALUE to disable this filter.
         * @return This builder for method chaining.
         */
        @NonNull
        public ScanFilterBuilder setMinRssi(int rssi) {
            this.minRssi = rssi;
            return this;
        }

        /**
         * Sets the maximum RSSI (signal strength) filter.
         *
         * @param rssi The maximum RSSI in dBm. Only devices with RSSI <= this value will be reported.
         *             Use Integer.MAX_VALUE to disable this filter.
         * @return This builder for method chaining.
         */
        @NonNull
        public ScanFilterBuilder setMaxRssi(int rssi) {
            this.maxRssi = rssi;
            return this;
        }

        /**
         * Adds a service UUID filter.
         *
         * @param uuid The service UUID to filter for. Must not be null.
         * @return This builder for method chaining.
         */
        @NonNull
        public ScanFilterBuilder addServiceUuid(@NonNull UUID uuid) {
            if (uuid != null && !serviceUuids.contains(uuid)) {
                serviceUuids.add(uuid);
            }
            return this;
        }

        /**
         * Clears all service UUID filters.
         *
         * @return This builder for method chaining.
         */
        @NonNull
        public ScanFilterBuilder clearServiceUuids() {
            serviceUuids.clear();
            return this;
        }

        /**
         * Clears all filters.
         *
         * @return This builder for method chaining.
         */
        @NonNull
        public ScanFilterBuilder clearFilters() {
            deviceNameFilter = null;
            macAddressFilter = null;
            minRssi = Integer.MIN_VALUE;
            maxRssi = Integer.MAX_VALUE;
            serviceUuids.clear();
            return this;
        }

        /**
         * Checks if a device matches the configured filters.
         */
        boolean matchesFilters(BluetoothDevice device, int rssi) {
            // Check RSSI
            if (rssi < minRssi || rssi > maxRssi) {
                return false;
            }

            // Check MAC address
            if (macAddressFilter != null && !macAddressFilter.equalsIgnoreCase(device.getAddress())) {
                return false;
            }

            // Check device name
            if (deviceNameFilter != null) {
                String deviceName = null;
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.ECLAIR) {
                    deviceName = device.getName();
                    return deviceName != null && deviceName.toLowerCase().contains(deviceNameFilter.toLowerCase());
                }
            }

            return true;
        }

        /**
         * Builds the ScanFilter list for the modern BLE API.
         */
        @RequiresApi(Build.VERSION_CODES.LOLLIPOP)
        List<ScanFilter> buildScanFilters() {
            List<ScanFilter> filters = new ArrayList<>();

            if (macAddressFilter != null) {
                ScanFilter filter = new ScanFilter.Builder()
                        .setDeviceAddress(macAddressFilter)
                        .build();
                filters.add(filter);
            }

            if (deviceNameFilter != null) {
                ScanFilter filter = new ScanFilter.Builder()
                        .setDeviceName(deviceNameFilter)
                        .build();
                filters.add(filter);
            }

            for (UUID uuid : serviceUuids) {
                ScanFilter filter = new ScanFilter.Builder()
                        .setServiceUuid(new android.os.ParcelUuid(uuid))
                        .build();
                filters.add(filter);
            }

            return filters;
        }
    }
}
