//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Vulkan pipeline context: loader, device, queues and the offscreen
//! dma-buf render target.
//!
//! it owns everything that outlives a frame — the shared Vulkan loader library, instance, physical and
//! logical device, graphics queue, command pool, and the swapchain
//! replacement. Because rendering is swapchain-less, the "swapchain"
//! role is played by a single offscreen [`vk::Image`] whose memory is
//! exported as a dma-buf file descriptor (see [`RenderTarget`]); the
//! Wayland layer in [`crate::ui_display`] presents it through the
//! linux-dmabuf protocol, so no window-system instance extension and
//! no `VkSwapchainKHR` are required.
//!
//! Frame recording, submission and synchronization live one layer up,
//! in [`crate::ui_renderer`], which borrows the accessors exposed
//! here
//!
//! # Library lifetime
//!
//! The loader library handle is opened once per [`PipelineContext`]
//! and intentionally never closed: unloading the Vulkan loader while
//! ICD driver threads may still run is unsafe, and every real
//! application keeps it mapped for its process lifetime.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::ffi::CStr;
use core::fmt;

use ash::vk;
use codevar_wl_protocol::DRM_FORMAT_XRGB8888;

/// Vulkan format of the render target: memory-layout counterpart of
/// `DRM_FORMAT_XRGB8888` (bytes `B, G, R, X` per pixel in memory).
const TARGET_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;

/// Device extensions without which dma-buf export and sync_file hand-off are
/// impossible. The second element is the name used in error messages.
const REQUIRED_DEVICE_EXTENSIONS: [(&CStr, &str); 4] = [
    (vk::KHR_EXTERNAL_MEMORY_FD_NAME, "VK_KHR_external_memory_fd"),
    (
        vk::EXT_EXTERNAL_MEMORY_DMA_BUF_NAME,
        "VK_EXT_external_memory_dma_buf",
    ),
    (vk::KHR_EXTERNAL_SEMAPHORE_FD_NAME, "VK_KHR_external_semaphore_fd"),
    (
        vk::EXT_IMAGE_DRM_FORMAT_MODIFIER_NAME,
        "VK_EXT_image_drm_format_modifier",
    ),
];

/// Usages required of the render target image.
fn target_usage() -> vk::ImageUsageFlags {
    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC
}

/// An owned operating-system file descriptor (a dma-buf or `sync_file`).
///
/// This is the `no_std` counterpart of `std::os::fd::OwnedFd`: the raw
/// descriptor is closed exactly once when the value drops, moved with
/// [`OwnedFd::into_raw`], adopted with [`OwnedFd::from_raw`] and inspected
/// with [`OwnedFd::as_raw`].
///
/// On non-unix, non-Windows targets dropping is a no-op; the dma-buf and
/// `sync_file` Vulkan extensions this type wraps are only reachable on
/// unix (and Win32 handles are not small integers, so they never enter
/// this type).
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct OwnedFd {
    raw: libc::c_int,
}

impl OwnedFd {
    /// Adopts a raw descriptor that is already open.
    ///
    /// # Safety
    ///
    /// `raw` must be an open descriptor that is owned by the caller and
    /// has not been borrowed by any other owner. After this call the
    /// descriptor is closed by [`OwnedFd`]'s `Drop`.
    #[inline]
    #[must_use]
    pub const unsafe fn from_raw(raw: libc::c_int) -> Self {
        Self { raw }
    }

    /// Returns the raw descriptor without giving up ownership.
    #[inline]
    #[must_use]
    pub const fn as_raw(&self) -> libc::c_int {
        self.raw
    }

    /// Returns the raw descriptor and gives up ownership; the caller must
    /// close it.
    #[inline]
    #[must_use]
    pub const fn into_raw(self) -> libc::c_int {
        let raw = self.raw;
        core::mem::forget(self);
        raw
    }
}

impl Drop for OwnedFd {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: `raw` was open when adopted by `from_raw` and this `Drop`
        // runs exactly once per descriptor (moves go through `into_raw`,
        // which forgets `self`); closing a descriptor twice is therefore
        // impossible.
        unsafe {
            libc::close(self.raw);
        }
        #[cfg(windows)]
        {
            windows_link::link!(
                "kernel32.dll" "system"
                fn CloseHandle(hObject: *mut core::ffi::c_void) -> i32
            );
            // SAFETY: as above for the descriptor itself; `CloseHandle` only
            // fails on an already-closed handle, which cannot happen here.
            unsafe {
                let _ = CloseHandle(self.raw as isize as *mut core::ffi::c_void);
            }
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = self.raw;
        }
    }
}

/// Error returned by Vulkan loader, instance, device and render-target
/// setup.
#[derive(Debug)]
pub enum PipelineError {
    /// Opening the platform Vulkan loader library (or resolving
    /// `vkGetInstanceProcAddr`) failed.
    EntryLoad(String),
    /// The loader does not support Vulkan 1.3, or `vkCreateInstance` failed.
    InstanceCreate(vk::Result),
    /// No Vulkan physical device was exposed by the loader.
    NoPhysicalDevice,
    /// The loader or every physical device reports an API version below 1.3.
    ApiVersionTooLow {
        /// The highest API version encountered (`vk::API_VERSION_x_y` packing).
        found: u32,
    },
    /// The selected physical device does not advertise a required extension.
    ExtensionMissing(&'static str),
    /// The selected physical device does not support a required external
    /// handle type (dma-buf memory or sync_file semaphores).
    ExternalHandleUnsupported(&'static str),
    /// A required core feature (for example `dynamicRendering`) is missing.
    FeatureUnsupported(&'static str),
    /// No queue family with `VK_QUEUE_GRAPHICS_BIT` exists.
    QueueUnavailable,
    /// `vkCreateDevice` failed.
    DeviceCreate(vk::Result),
    /// A physical-device enumeration or property query failed.
    PhysicalDeviceQuery(vk::Result),
    /// The modifier list passed to [`PipelineContext::new`] was empty.
    EmptyModifierList,
    /// None of the requested DRM format modifiers is usable for the render
    /// target (format, usage and dma-buf export combination unsupported).
    NoCompatibleModifier,
    /// `vkCreateImage` failed.
    ImageCreate(vk::Result),
    /// `vkCreateImageView` failed.
    ImageViewCreate(vk::Result),
    /// No memory type both allowed by the image and suitable (device-local)
    /// could be found.
    NoSuitableMemoryType,
    /// `vkAllocateMemory` failed.
    MemoryAllocate(vk::Result),
    /// `vkBindImageMemory` failed.
    MemoryBind(vk::Result),
    /// Exporting the image memory as a dma-buf fd failed.
    MemoryFdExport(vk::Result),
    /// Querying the driver-chosen DRM format modifier failed.
    ModifierQuery(vk::Result),
    /// Creating the graphics command pool failed.
    CommandAllocation(vk::Result),
    /// An internal invariant was violated; this indicates a bug in this module.
    Internal(&'static str),
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EntryLoad(err) => write!(f, "failed to load the Vulkan loader library: {err}"),
            Self::InstanceCreate(err) => write!(f, "Vulkan instance creation failed: {err:?}"),
            Self::NoPhysicalDevice => write!(f, "no Vulkan physical device is available"),
            Self::ApiVersionTooLow { found } => {
                write!(f, "Vulkan 1.3 is required but only {found:#010x} is available")
            }
            Self::ExtensionMissing(name) => {
                write!(f, "required device extension {name} is not supported")
            }
            Self::ExternalHandleUnsupported(name) => {
                write!(f, "required external handle type {name} is not supported")
            }
            Self::FeatureUnsupported(name) => write!(f, "required Vulkan feature {name} is missing"),
            Self::QueueUnavailable => write!(f, "no graphics queue family is available"),
            Self::DeviceCreate(err) => write!(f, "logical device creation failed: {err:?}"),
            Self::PhysicalDeviceQuery(err) => {
                write!(f, "physical device query failed: {err:?}")
            }
            Self::EmptyModifierList => {
                write!(f, "the DRM modifier list for the render target is empty")
            }
            Self::NoCompatibleModifier => {
                write!(
                    f,
                    "none of the requested DRM format modifiers is usable for XRGB8888"
                )
            }
            Self::ImageCreate(err) => write!(f, "render target image creation failed: {err:?}"),
            Self::ImageViewCreate(err) => write!(f, "render target view creation failed: {err:?}"),
            Self::NoSuitableMemoryType => {
                write!(f, "no suitable memory type for the render target image")
            }
            Self::MemoryAllocate(err) => {
                write!(f, "render target memory allocation failed: {err:?}")
            }
            Self::MemoryBind(err) => write!(f, "binding render target memory failed: {err:?}"),
            Self::MemoryFdExport(err) => write!(f, "dma-buf fd export failed: {err:?}"),
            Self::ModifierQuery(err) => {
                write!(f, "querying the chosen DRM format modifier failed: {err:?}")
            }
            Self::CommandAllocation(err) => {
                write!(f, "graphics command pool creation failed: {err:?}")
            }
            Self::Internal(msg) => write!(f, "internal Vulkan pipeline error: {msg}"),
        }
    }
}

impl core::error::Error for PipelineError {}

#[cfg(unix)]
fn open_library() -> Result<*mut core::ffi::c_void, PipelineError> {
    const CANDIDATES: [&CStr; 3] = [c"libvulkan.so.1", c"libvulkan.so", c"libvulkan.1.dylib"];
    let mut last_error = String::from("no candidate library name was tried");
    for name in CANDIDATES {
        // SAFETY: `name` is a valid NUL-terminated constant string and the
        // RTLD_* flags are defined POSIX values. A null return only means
        // the library could not be opened; the handle is checked.
        let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if !handle.is_null() {
            return Ok(handle);
        }
        // SAFETY: `dlerror` returns either null or a pointer to a static
        // NUL-terminated message owned by the dynamic loader.
        let message = unsafe { libc::dlerror() };
        last_error = if message.is_null() {
            format!("dlopen({}) failed", name.to_string_lossy())
        } else {
            // SAFETY: non-null `dlerror` results are valid C strings.
            let text = unsafe { CStr::from_ptr(message) };
            format!(
                "dlopen({}) failed: {}",
                name.to_string_lossy(),
                text.to_string_lossy()
            )
        };
    }
    Err(PipelineError::EntryLoad(last_error))
}

#[cfg(unix)]
fn library_symbol(handle: *mut core::ffi::c_void, name: &CStr) -> *mut core::ffi::c_void {
    // SAFETY: `handle` came from a successful `dlopen` in `open_library`
    // and `name` is a valid NUL-terminated symbol name. A null return
    // means the symbol is absent and is reported as an error.
    unsafe { libc::dlsym(handle, name.as_ptr()) }
}

#[cfg(windows)]
fn open_library() -> Result<*mut core::ffi::c_void, PipelineError> {
    windows_link::link!(
        "kernel32.dll" "system"
        fn LoadLibraryA(lpLibFileName: *const u8) -> *mut core::ffi::c_void
    );
    windows_link::link!(
        "kernel32.dll" "system"
        fn GetLastError() -> u32
    );
    // SAFETY: the name is a valid NUL-terminated ASCII string constant.
    let handle = unsafe { LoadLibraryA(b"vulkan-1.dll\0".as_ptr()) };
    if handle.is_null() {
        // SAFETY: `GetLastError` is always callable and returns the error
        // of the immediately preceding failed Win32 call.
        let code = unsafe { GetLastError() };
        return Err(PipelineError::EntryLoad(format!(
            "LoadLibraryA(vulkan-1.dll) failed with error {code}"
        )));
    }
    Ok(handle)
}

#[cfg(windows)]
fn library_symbol(handle: *mut core::ffi::c_void, name: &CStr) -> *mut core::ffi::c_void {
    windows_link::link!(
        "kernel32.dll" "system"
        fn GetProcAddress(hModule: *mut core::ffi::c_void, lpProcName: *const u8) -> *mut core::ffi::c_void
    );
    // SAFETY: `handle` came from a successful `LoadLibraryA` and `name` is
    // NUL-terminated; a null return means the symbol is absent.
    unsafe { GetProcAddress(handle, name.as_ptr()) }
}

#[cfg(not(any(unix, windows)))]
fn open_library() -> Result<*mut core::ffi::c_void, PipelineError> {
    Err(PipelineError::EntryLoad(String::from(
        "no Vulkan loader backend exists for this target platform",
    )))
}

#[cfg(not(any(unix, windows)))]
fn library_symbol(_handle: *mut core::ffi::c_void, _name: &CStr) -> *mut core::ffi::c_void {
    core::ptr::null_mut()
}

/// An open handle to the platform's shared Vulkan loader library.
///
/// The handle keeps `vkGetInstanceProcAddr` (and every function pointer
/// resolved from it) alive; see the module documentation for why it is
/// never released. The raw handle is inert between `dlopen`/`LoadLibrary`
/// and the process exit, which makes it safe to share across threads. The
/// field is intentionally never read: keeping it stored *is* its use.
pub struct VulkanLibrary {
    _handle: *mut core::ffi::c_void,
}

// SAFETY: the library handle is only ever passed to the dynamic loader
// (`dlsym`/`GetProcAddress`), which is thread-safe by contract, and it is
// never closed, so no thread can observe an unloaded library.
unsafe impl Send for VulkanLibrary {}
// SAFETY: see the `Send` implementation above.
unsafe impl Sync for VulkanLibrary {}

impl VulkanLibrary {
    /// Opens the platform Vulkan loader and builds an [`ash::Entry`] from
    /// its `vkGetInstanceProcAddr`.
    ///
    /// No `std`, `libloading` or link-time Vulkan dependency is involved:
    /// the shared library is opened with `dlopen` (unix) or
    /// `LoadLibraryA` (Windows) and the entry point is resolved manually.
    ///
    /// # Errors
    ///
    /// * [`PipelineError::EntryLoad`] — no candidate library could be
    ///   opened, or it does not export `vkGetInstanceProcAddr`.
    pub fn load() -> Result<(Self, ash::Entry), PipelineError> {
        let handle = open_library()?;
        let symbol = library_symbol(handle, c"vkGetInstanceProcAddr");
        if symbol.is_null() {
            return Err(PipelineError::EntryLoad(String::from(
                "the Vulkan loader does not export vkGetInstanceProcAddr",
            )));
        }
        // SAFETY: `symbol` is the address of the loader's
        // `vkGetInstanceProcAddr`, whose canonical signature is
        // `PFN_vkGetInstanceProcAddr`; a successful `dlsym`/`GetProcAddress`
        // for that name guarantees both the signature and that the pointer
        // stays valid for the process lifetime (the handle is never closed).
        let get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr = unsafe { core::mem::transmute(symbol) };
        let static_fn = ash::StaticFn {
            get_instance_proc_addr,
        };
        // SAFETY: `static_fn.get_instance_proc_addr` follows the Vulkan 1.0
        // contract and remains valid for at least the lifetime of the
        // returned entry, because the library handle backing it is never
        // released (see the type documentation).
        let entry = unsafe { ash::Entry::from_static_fn(static_fn) };
        Ok((Self { _handle: handle }, entry))
    }
}

/// Description of the exported render target for the Wayland/dmabuf layer.
///
/// The dma-buf fd refers to the fully-composited-ready image contents;
/// `modifier`, `stride` and `offset` describe plane 0 so the compositor can
/// build a `wl_buffer` via `zwp_linux_dmabuf_v1`.
#[derive(Debug)]
pub struct RenderTarget {
    /// dma-buf file descriptor for plane 0 (owned; the Wayland layer sends
    /// it via `SCM_RIGHTS`, which duplicates it).
    pub dmabuf_fd: OwnedFd,
    /// DRM fourcc — always `DRM_FORMAT_XRGB8888` (`0x34325258`).
    pub drm_format: u32,
    /// DRM format modifier chosen by the driver at image creation.
    pub modifier: u64,
    /// Plane 0 stride in bytes.
    pub stride: u32,
    /// Plane 0 byte offset.
    pub offset: u32,
    /// Render target width in pixels.
    pub width: u32,
    /// Render target height in pixels.
    pub height: u32,
}

/// Checks whether `name` is present in the driver's advertised extension list.
fn extension_supported(available: &[vk::ExtensionProperties], name: &CStr) -> bool {
    let wanted = name.to_bytes();
    available.iter().any(|extension| {
        let advertised: &[core::ffi::c_char] = &extension.extension_name;
        advertised.len() > wanted.len()
            && advertised[wanted.len()] == 0
            && wanted
                .iter()
                .zip(&advertised[..wanted.len()])
                .all(|(&expected, &actual)| actual as u8 == expected)
    })
}

/// Returns the index of the first queue family with graphics support.
fn find_graphics_family(instance: &ash::Instance, physical: vk::PhysicalDevice) -> Option<u32> {
    // SAFETY: `physical` comes from `enumerate_physical_devices` and the
    // instance is alive; the driver fills the returned slice.
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
    families
        .iter()
        .position(|family| {
            family
                .queue_flags
                .contains(vk::QueueFlags::GRAPHICS)
        })
        .map(|index| index as u32)
}

/// Preference score for physical-device selection: discrete beats integrated
/// beats virtual GPU, everything else ranks last.
fn device_score(device_type: vk::PhysicalDeviceType) -> u32 {
    match device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 3,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
        _ => 0,
    }
}

/// Picks a memory type from `type_bits`, preferring one with all of
/// `preferred` flags and falling back to the first type the image accepts.
fn find_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    preferred: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let mut fallback = None;
    for (index, memory_type) in properties.memory_types[..properties.memory_type_count as usize]
        .iter()
        .enumerate()
    {
        if type_bits & (1 << index) == 0 {
            continue;
        }
        if memory_type.property_flags.contains(preferred) {
            return Some(index as u32);
        }
        if fallback.is_none() {
            fallback = Some(index as u32);
        }
    }
    fallback
}

/// Verifies that the physical device can export dma-buf memory and import and
/// export `sync_file` semaphores.
fn check_external_support(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> Result<(), PipelineError> {
    let buffer_info = vk::PhysicalDeviceExternalBufferInfo::default()
        .flags(vk::BufferCreateFlags::empty())
        .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let mut buffer_properties = vk::ExternalBufferProperties::default();
    // SAFETY: `physical` is valid and both structures are correctly typed and
    // live across the call; the driver only writes `buffer_properties`.
    unsafe {
        instance.get_physical_device_external_buffer_properties(
            physical,
            &buffer_info,
            &mut buffer_properties,
        );
    }
    let dma_buf_ok = buffer_properties
        .external_memory_properties
        .compatible_handle_types
        .contains(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
        && buffer_properties
            .external_memory_properties
            .external_memory_features
            .contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE);
    if !dma_buf_ok {
        return Err(PipelineError::ExternalHandleUnsupported(
            "VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT",
        ));
    }

    let semaphore_info = vk::PhysicalDeviceExternalSemaphoreInfo::default()
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
    let mut semaphore_properties = vk::ExternalSemaphoreProperties::default();
    // SAFETY: as above, with a well-formed external semaphore query.
    unsafe {
        instance.get_physical_device_external_semaphore_properties(
            physical,
            &semaphore_info,
            &mut semaphore_properties,
        );
    }
    let sync_fd_ok = semaphore_properties
        .compatible_handle_types
        .contains(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
        && semaphore_properties
            .external_semaphore_features
            .contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE)
        && semaphore_properties
            .external_semaphore_features
            .contains(vk::ExternalSemaphoreFeatureFlags::IMPORTABLE);
    if !sync_fd_ok {
        return Err(PipelineError::ExternalHandleUnsupported(
            "VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT",
        ));
    }
    Ok(())
}

/// Checks whether the driver can create the render-target image with `modifier`
/// (this implements VUID-VkImageDrmFormatModifierListCreateInfoEXT-pDrmFormatModifiers-02263,
/// including dma-buf exportability of the resulting image).
fn modifier_supported(instance: &ash::Instance, physical: vk::PhysicalDevice, modifier: u64) -> bool {
    let mut modifier_info = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
        .drm_format_modifier(modifier)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let mut external_info = vk::PhysicalDeviceExternalImageFormatInfo::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let image_info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(TARGET_FORMAT)
        .ty(vk::ImageType::TYPE_2D)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(target_usage())
        .flags(vk::ImageCreateFlags::empty())
        .push_next(&mut modifier_info)
        .push_next(&mut external_info);
    let mut external_properties = vk::ExternalImageFormatProperties::default();
    let mut format_properties = vk::ImageFormatProperties2::default().push_next(&mut external_properties);
    // SAFETY: all chained structures outlive the call and the driver only
    // writes the output structures. `Err` (typically
    // `VK_ERROR_FORMAT_NOT_SUPPORTED`) simply means the modifier is unusable.
    let query_ok = unsafe {
        instance.get_physical_device_image_format_properties2(physical, &image_info, &mut format_properties)
    }
    .is_ok();
    query_ok
        && external_properties
            .external_memory_properties
            .compatible_handle_types
            .contains(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
        && external_properties
            .external_memory_properties
            .external_memory_features
            .contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE)
}

/// Creates the instance after verifying the loader reports Vulkan 1.3.
fn create_instance(entry: &ash::Entry) -> Result<ash::Instance, PipelineError> {
    // SAFETY: `entry` is a freshly built, valid loader handle.
    let loader_version =
        unsafe { entry.try_enumerate_instance_version() }.map_err(PipelineError::InstanceCreate)?;
    let loader_version = loader_version.unwrap_or(vk::API_VERSION_1_0);
    if loader_version < vk::API_VERSION_1_3 {
        return Err(PipelineError::ApiVersionTooLow {
            found: loader_version,
        });
    }

    let app_info = vk::ApplicationInfo::default()
        .application_name(c"codevar")
        .application_version(1)
        .engine_name(c"codevar")
        .engine_version(1)
        .api_version(vk::API_VERSION_1_3);
    // No instance extensions: rendering is fully offscreen and the
    // Wayland/dmabuf layer lives outside Vulkan.
    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    // SAFETY: `app_info` outlives `instance_info`; both are well-formed.
    unsafe { entry.create_instance(&instance_info, None) }.map_err(PipelineError::InstanceCreate)
}

/// Picks the best Vulkan 1.3 graphics-capable physical device and its
/// graphics queue family.
fn select_physical_device(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, u32), PipelineError> {
    // SAFETY: `instance` is valid; the driver allocates the returned vector.
    let devices =
        unsafe { instance.enumerate_physical_devices() }.map_err(PipelineError::PhysicalDeviceQuery)?;
    if devices.is_empty() {
        return Err(PipelineError::NoPhysicalDevice);
    }

    let mut best: Option<(u32, vk::PhysicalDevice, u32)> = None;
    let mut highest_api_seen = vk::API_VERSION_1_0;
    let mut v13_seen = false;
    let mut v13_without_graphics = false;
    for &physical in &devices {
        // SAFETY: `physical` is a valid device handle from enumeration.
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        highest_api_seen = highest_api_seen.max(properties.api_version);
        if properties.api_version < vk::API_VERSION_1_3 {
            continue;
        }
        v13_seen = true;
        let Some(queue_family) = find_graphics_family(instance, physical) else {
            v13_without_graphics = true;
            continue;
        };
        let score = device_score(properties.device_type);
        if best.is_none_or(|(best_score, _, _)| score > best_score) {
            best = Some((score, physical, queue_family));
        }
    }
    best.map(|(_, physical, family)| (physical, family))
        .ok_or(if v13_seen && v13_without_graphics {
            PipelineError::QueueUnavailable
        } else if v13_seen {
            PipelineError::NoPhysicalDevice
        } else {
            PipelineError::ApiVersionTooLow {
                found: highest_api_seen,
            }
        })
}

/// Verifies the required extensions, external handle types and the
/// `dynamicRendering` feature on `physical`.
fn validate_capabilities(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> Result<(), PipelineError> {
    // SAFETY: `physical` is valid; the driver fills the returned vector.
    let available_extensions = unsafe { instance.enumerate_device_extension_properties(physical) }
        .map_err(PipelineError::PhysicalDeviceQuery)?;
    for (name, label) in REQUIRED_DEVICE_EXTENSIONS {
        if !extension_supported(&available_extensions, name) {
            return Err(PipelineError::ExtensionMissing(label));
        }
    }
    check_external_support(instance, physical)?;

    let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut features13);
    // SAFETY: the feature chain is well-formed and outlives the call.
    unsafe { instance.get_physical_device_features2(physical, &mut features2) };
    if features13.dynamic_rendering == vk::FALSE {
        return Err(PipelineError::FeatureUnsupported("dynamicRendering"));
    }
    Ok(())
}

/// Creates the logical device with the required extensions and returns it
/// together with its graphics queue.
fn create_device(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    queue_family: u32,
) -> Result<(ash::Device, vk::Queue), PipelineError> {
    let queue_priorities = [1.0_f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&queue_priorities);
    let extension_names: Vec<*const core::ffi::c_char> = REQUIRED_DEVICE_EXTENSIONS
        .iter()
        .map(|(name, _)| name.as_ptr())
        .collect();
    let mut enabled13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(core::slice::from_ref(&queue_info))
        .enabled_extension_names(&extension_names)
        .push_next(&mut enabled13);
    // SAFETY: the queue info, extension name pointers and feature chain are
    // all live and reference constants/statics for the duration of the call.
    let device = unsafe { instance.create_device(physical, &device_info, None) }
        .map_err(PipelineError::DeviceCreate)?;
    // SAFETY: the queue family was validated to exist with one queue.
    let queue = unsafe { device.get_device_queue(queue_family, 0) };
    Ok((device, queue))
}

/// Owns partially-created Vulkan objects while [`PipelineContext::new`] runs
/// so that every early `Err` return releases whatever was already created.
///
/// On the success path each handle is moved back out with [`take_created`],
/// after which every slot is `None` and `Drop` becomes a no-op.
#[derive(Default)]
struct Cleanup {
    instance: Option<ash::Instance>,
    device: Option<ash::Device>,
    image: Option<vk::Image>,
    view: Option<vk::ImageView>,
    memory: Option<vk::DeviceMemory>,
    command_pool: Option<vk::CommandPool>,
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        let Some(device) = self.device.as_ref() else {
            if let Some(instance) = self.instance.as_ref() {
                // SAFETY: the instance handle was created by `Entry::create_instance`
                // and has not been destroyed yet (the success path empties the slot
                // before `Drop` runs). The entry that loaded its function pointers
                // is declared before `Cleanup`, so it outlives this call.
                unsafe { instance.destroy_instance(None) };
            }
            return;
        };
        // SAFETY: every handle below was created on `device` and has not been
        // destroyed yet. `device_wait_idle` failing (device lost) is ignored:
        // destroying the objects of a lost device is still permitted and is the
        // only way to release the host-side resources.
        unsafe {
            let _ = device.device_wait_idle();
            if let Some(pool) = self.command_pool.take() {
                device.destroy_command_pool(pool, None);
            }
            if let Some(view) = self.view.take() {
                device.destroy_image_view(view, None);
            }
            if let Some(image) = self.image.take() {
                device.destroy_image(image, None);
            }
            if let Some(memory) = self.memory.take() {
                device.free_memory(memory, None);
            }
            device.destroy_device(None);
        }
        self.device = None;
        if let Some(instance) = self.instance.take() {
            // SAFETY: see the early-return branch above; the device that used
            // this instance has just been destroyed.
            unsafe { instance.destroy_instance(None) };
        }
    }
}

/// Moves a resource out of a [`Cleanup`] slot during finalization.
///
/// Every slot is guaranteed to be populated on the success path; the error
/// branch only exists because `Option::take` cannot express that statically.
fn take_created<T>(slot: &mut Option<T>) -> Result<T, PipelineError> {
    slot.take().ok_or(PipelineError::Internal(
        "Vulkan resource missing during finalization",
    ))
}

/// Swapchain-less pipeline context producing an exported dma-buf image.
///
/// Created through [`PipelineContext::new`]; resources are borrowed with
/// the accessor methods below and the exported image layout
/// is available through [`PipelineContext::render_target`]. Dropping the
/// context waits for the device to go idle and destroys every Vulkan
/// object it owns.
pub struct PipelineContext {
    /// Keeps the shared Vulkan loader library mapped so every function
    /// pointer resolved from it stays valid (see the module docs).
    _library: VulkanLibrary,
    /// Kept alive with the library: its dispatch tables are the source of
    /// the instance-level function pointers used during setup.
    _entry: ash::Entry,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    queue_family: u32,
    device: ash::Device,
    queue: vk::Queue,
    ext_semaphore_fd: ash::khr::external_semaphore_fd::Device,
    command_pool: vk::CommandPool,
    image: vk::Image,
    image_view: vk::ImageView,
    memory: vk::DeviceMemory,
    render_target: RenderTarget,
}

impl PipelineContext {
    /// Creates the pipeline context, offscreen target image, dma-buf memory
    /// export and graphics command pool.
    ///
    /// `modifiers` is the list of DRM modifiers advertised by the compositor's
    /// linux-dmabuf feedback for XRGB8888 (must be non-empty; the driver picks
    /// one via [`vk::ImageDrmFormatModifierListCreateInfoEXT`], then the chosen
    /// modifier and plane layout are queried back with
    /// `vkGetImageDrmFormatModifierPropertiesEXT` and `vkGetImageSubresourceLayout`).
    ///
    /// The image is created with `VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT`
    /// rather than `VK_IMAGE_TILING_OPTIMAL`: the modifier query requires it
    /// (VUID-vkGetImageDrmFormatModifierPropertiesEXT-image-02272) and it is the
    /// canonical export pattern of VK_EXT_image_drm_format_modifier.
    ///
    /// # Errors
    ///
    /// * [`PipelineError::EmptyModifierList`] — `modifiers` was empty; this is
    ///   detected before the Vulkan loader is opened, so it also fires on
    ///   machines without any GPU or driver.
    /// * [`PipelineError::EntryLoad`] — the platform loader library could not
    ///   be opened or does not export `vkGetInstanceProcAddr`.
    /// * [`PipelineError::InstanceCreate`], [`PipelineError::DeviceCreate`] —
    ///   loader/instance/device setup failed.
    /// * [`PipelineError::ApiVersionTooLow`], [`PipelineError::NoPhysicalDevice`],
    ///   [`PipelineError::QueueUnavailable`] — no suitable Vulkan 1.3 graphics
    ///   device.
    /// * [`PipelineError::ExtensionMissing`] — a required device extension is
    ///   absent.
    /// * [`PipelineError::ExternalHandleUnsupported`] — dma-buf or `sync_file`
    ///   handle types are unsupported.
    /// * [`PipelineError::EmptyModifierList`], [`PipelineError::NoCompatibleModifier`]
    ///   — the modifier list is empty or no entry works for this image.
    /// * Image, memory, fd export and command-pool errors mirror the failing
    ///   Vulkan call (`vk::Result` is carried along).
    ///
    /// All partially-created objects are destroyed before returning an error.
    pub fn new(width: u32, height: u32, modifiers: &[u64]) -> Result<Self, PipelineError> {
        if modifiers.is_empty() {
            return Err(PipelineError::EmptyModifierList);
        }

        // Open the loader manually (no `std`, no `libloading`).
        let (library, entry) = VulkanLibrary::load()?;
        let instance = create_instance(&entry)?;

        // From here on, any error must release `instance` (and everything
        // created after it); `Cleanup`'s `Drop` does that. It is declared after
        // `library`/`entry`, so they outlive the instance on every path.
        let mut cleanup = Cleanup {
            instance: Some(instance.clone()),
            device: None,
            image: None,
            view: None,
            memory: None,
            command_pool: None,
        };

        let (physical, queue_family) = select_physical_device(&instance)?;
        validate_capabilities(&instance, physical)?;
        let (device, queue) = create_device(&instance, physical, queue_family)?;
        cleanup.device = Some(device.clone());

        // Function tables for the fd-based extensions enabled on the device.
        let ext_memory_fd = ash::khr::external_memory_fd::Device::new(&instance, &device);
        let ext_drm = ash::ext::image_drm_format_modifier::Device::new(&instance, &device);
        let ext_semaphore_fd = ash::khr::external_semaphore_fd::Device::new(&instance, &device);

        let compatible_modifiers: Vec<u64> = modifiers
            .iter()
            .copied()
            .filter(|&modifier| modifier_supported(&instance, physical, modifier))
            .collect();
        if compatible_modifiers.is_empty() {
            return Err(PipelineError::NoCompatibleModifier);
        }

        // Note the chain: ImageCreateInfo -> modifier list -> external memory.
        // The image uses DRM-format-modifier tiling so the driver-chosen
        // modifier can be queried back after creation.
        let mut modifier_list = vk::ImageDrmFormatModifierListCreateInfoEXT::default()
            .drm_format_modifiers(&compatible_modifiers);
        let mut external_memory = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(TARGET_FORMAT)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(target_usage())
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut modifier_list)
            .push_next(&mut external_memory);
        // SAFETY: the pNext chain (`modifier_list` -> `external_memory`) and the
        // modifier slice outlive the call.
        let image = unsafe { device.create_image(&image_info, None) }.map_err(PipelineError::ImageCreate)?;
        cleanup.image = Some(image);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(TARGET_FORMAT)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        // SAFETY: `image` is valid and the view info references it correctly.
        let image_view =
            unsafe { device.create_image_view(&view_info, None) }.map_err(PipelineError::ImageViewCreate)?;
        cleanup.view = Some(image_view);

        // SAFETY: the image handle is valid.
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        // SAFETY: `physical` is valid and the driver fills the properties.
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        let memory_type = find_memory_type(
            &memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or(PipelineError::NoSuitableMemoryType)?;

        let mut export_info = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        // A dedicated allocation is what dma-buf consumers expect in practice
        // and is permitted (VK_EXT_image_drm_format_modifier does not require it).
        let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let allocation_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type)
            .push_next(&mut export_info)
            .push_next(&mut dedicated_info);
        // SAFETY: the export/dedicated chain outlives the call and the size
        // matches `requirements.size` as required for dedicated allocations.
        let memory = unsafe { device.allocate_memory(&allocation_info, None) }
            .map_err(PipelineError::MemoryAllocate)?;
        cleanup.memory = Some(memory);
        // SAFETY: `memory` satisfies the image's requirements and is large
        // enough; offset 0 with an exclusive-sharing image is always valid.
        unsafe { device.bind_image_memory(image, memory, 0) }.map_err(PipelineError::MemoryBind)?;

        // Export the allocation as a dma-buf fd for the Wayland layer.
        let get_fd_info = vk::MemoryGetFdInfoKHR::default()
            .memory(memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        // SAFETY: the extension is enabled and `memory` was allocated with
        // `VkExportMemoryAllocateInfo` advertising the DMA_BUF handle type.
        let raw_fd =
            unsafe { ext_memory_fd.get_memory_fd(&get_fd_info) }.map_err(PipelineError::MemoryFdExport)?;
        if raw_fd < 0 {
            return Err(PipelineError::Internal(
                "vkGetMemoryFdKHR returned a negative file descriptor",
            ));
        }
        // SAFETY: on success `vkGetMemoryFdKHR` transfers ownership of a valid
        // open file descriptor to the application, so `OwnedFd` may adopt it.
        let dmabuf_fd = unsafe { OwnedFd::from_raw(raw_fd) };

        let mut modifier_properties = vk::ImageDrmFormatModifierPropertiesEXT::default();
        // SAFETY: the extension is enabled and the image was created with
        // `VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT` (required by this query).
        unsafe { ext_drm.get_image_drm_format_modifier_properties(image, &mut modifier_properties) }
            .map_err(PipelineError::ModifierQuery)?;
        let plane_subresource = vk::ImageSubresource {
            aspect_mask: vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
            mip_level: 0,
            array_layer: 0,
        };
        // SAFETY: memory-plane layout is queryable for DRM-modifier images and
        // plane 0 exists (the format has a single memory plane).
        let plane_layout = unsafe { device.get_image_subresource_layout(image, plane_subresource) };
        let stride = u32::try_from(plane_layout.row_pitch)
            .map_err(|_| PipelineError::Internal("plane stride exceeds the u32 range"))?;
        let offset = u32::try_from(plane_layout.offset)
            .map_err(|_| PipelineError::Internal("plane offset exceeds the u32 range"))?;
        let render_target = RenderTarget {
            dmabuf_fd,
            drm_format: DRM_FORMAT_XRGB8888,
            modifier: modifier_properties.drm_format_modifier,
            stride,
            offset,
            width,
            height,
        };
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(queue_family);
        // SAFETY: the graphics family index is valid for this device.
        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .map_err(PipelineError::CommandAllocation)?;
        cleanup.command_pool = Some(command_pool);

        // Success: move everything out of the cleanup guard.
        Ok(PipelineContext {
            _library: library,
            _entry: entry,
            instance: take_created(&mut cleanup.instance)?,
            physical,
            queue_family,
            device: take_created(&mut cleanup.device)?,
            queue,
            ext_semaphore_fd,
            command_pool: take_created(&mut cleanup.command_pool)?,
            image,
            image_view: take_created(&mut cleanup.view)?,
            memory: take_created(&mut cleanup.memory)?,
            render_target,
        })
    }

    /// The exported render target (dma-buf fd + layout info).
    #[inline]
    pub fn render_target(&self) -> &RenderTarget {
        &self.render_target
    }

    /// The Vulkan instance this context was created on.
    #[inline]
    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    /// The selected physical device (for memory-property and capability
    /// queries).
    #[inline]
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical
    }

    /// The logical device owning every object of this context.
    #[inline]
    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    /// The graphics queue used for frame submissions.
    #[inline]
    pub fn queue(&self) -> vk::Queue {
        self.queue
    }

    /// Index of the graphics queue family the command pool was created on.
    #[inline]
    pub fn queue_family(&self) -> u32 {
        self.queue_family
    }

    /// Command pool frame command buffers that is allocated
    #[inline]
    pub fn command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }

    /// Dispatch table for `VK_KHR_external_semaphore_fd`, used by the
    /// renderer to import/export `sync_file` semaphores.
    #[inline]
    pub fn external_semaphore_fd(&self) -> &ash::khr::external_semaphore_fd::Device {
        &self.ext_semaphore_fd
    }

    /// The offscreen render-target image (for layout barriers).
    #[inline]
    pub fn image(&self) -> vk::Image {
        self.image
    }

    /// Color attachment view of the render target (for dynamic rendering).
    #[inline]
    pub fn image_view(&self) -> vk::ImageView {
        self.image_view
    }

    /// Render target width in pixels.
    #[inline]
    pub fn width(&self) -> u32 {
        self.render_target.width
    }

    /// Render target height in pixels.
    #[inline]
    pub fn height(&self) -> u32 {
        self.render_target.height
    }

    /// Vulkan format of the color attachment (`B8G8R8A8_UNORM`, the
    /// memory-layout counterpart of `DRM_FORMAT_XRGB8888`).
    #[inline]
    pub const fn color_format() -> vk::Format {
        TARGET_FORMAT
    }
}

impl Drop for PipelineContext {
    fn drop(&mut self) {
        // SAFETY: `drop` has exclusive access to the context, so no frame can
        // be recording or submitting concurrently. A failed wait (device lost)
        // is ignored: the GPU is gone and destroying the handles is still the
        // correct way to release the host-side resources.
        unsafe {
            let _ = self.device.device_wait_idle();
        }
        // SAFETY: the device is idle, this context has exclusive access, and
        // every handle below was created by `new` on this device and has not
        // been destroyed. Destroying the instance last is required because its
        // dispatch tables are used by the device calls above; `_entry` and
        // `_library` only drop after this body runs.
        unsafe {
            self.device
                .destroy_command_pool(self.command_pool, None);
            self.device
                .destroy_image_view(self.image_view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The modifier list is validated before the Vulkan loader is touched, so
    /// this error path must work on machines without any GPU or driver.
    #[test]
    fn empty_modifier_list_is_rejected() {
        assert!(matches!(
            PipelineContext::new(640, 640, &[]),
            Err(PipelineError::EmptyModifierList)
        ));
    }

    /// Dropped descriptors must be closed exactly once: the descriptor is
    /// invalid after the owner drops and must not be closed again.
    #[cfg(unix)]
    #[test]
    fn owned_fd_closes_on_drop() {
        let mut fds = [0 as libc::c_int; 2];
        // SAFETY: `pipe` writes two valid descriptors into the array.
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
        let descriptor = fds[0];
        let owned = unsafe { OwnedFd::from_raw(descriptor) };
        drop(owned);
        // SAFETY: probing a possibly-closed descriptor is exactly what this
        // test verifies; `F_GETFD` never dereferences the descriptor.
        let result = unsafe { libc::fcntl(descriptor, libc::F_GETFD) };
        assert_eq!(result, -1);
        // SAFETY: `fds[1]` is still open and owned by this test.
        unsafe { libc::close(fds[1]) };
    }

    /// `into_raw` hands ownership over: the descriptor must stay open.
    #[cfg(unix)]
    #[test]
    fn owned_fd_into_raw_keeps_descriptor_open() {
        let mut fds = [0 as libc::c_int; 2];
        // SAFETY: `pipe` writes two valid descriptors into the array.
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
        let owned = unsafe { OwnedFd::from_raw(fds[0]) };
        let raw = owned.into_raw();
        // SAFETY: the descriptor was not closed by `into_raw`.
        assert_ne!(unsafe { libc::fcntl(raw, libc::F_GETFD) }, -1);
        // SAFETY: the caller re-took ownership and closes it here.
        unsafe {
            libc::close(raw);
            libc::close(fds[1]);
        }
    }
}
