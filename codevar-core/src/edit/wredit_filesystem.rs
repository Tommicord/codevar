//! Copyright 2026 Codevar
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! File system handler for managing file pointers and operations.
//!
//! This module provides a robust file system handler that stores only pointers
//! to opened files on disk, enabling efficient file management for collaborative
//! editing features.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Maximum number of simultaneously open files
const MAX_OPEN_FILES: usize = 256;

/// File handle type for tracking opened files
#[derive(Debug, Clone)]
pub struct FileHandle {
    /// Unique file identifier
    pub id: u32,
    /// File path
    pub path: PathBuf,
    /// File size in bytes
    pub size: u64,
    /// File modification time
    pub modified_time: u64,
    /// File flags
    pub flags: u32,
}

impl FileHandle {
    /// Creates a new file handle.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique file identifier
    /// * `path` - File path
    /// * `size` - File size
    /// * `modified_time` - File modification time
    pub fn new(id: u32, path: PathBuf, size: u64, modified_time: u64) -> Self {
        Self {
            id,
            path,
            size,
            modified_time,
            flags: 0,
        }
    }

    /// Returns the file name without path.
    pub fn file_name(&self) -> Option<&str> {
        self.path.file_name()?.to_str()
    }

    /// Returns the file extension.
    pub fn extension(&self) -> Option<&str> {
        self.path.extension()?.to_str()
    }
}

/// File system error types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileSystemError {
    /// File not found
    FileNotFound,
    /// Permission denied
    PermissionDenied,
    /// File already open
    AlreadyOpen,
    /// Maximum open files reached
    MaxFilesReached,
    /// Invalid file handle
    InvalidHandle,
    /// I/O error
    IoError(String),
    /// Path does not exist
    PathNotExist,
}

/// Result type for file system operations
pub type FileSystemResult<T> = Result<T, FileSystemError>;

/// File system handler for managing file pointers.
pub struct FileSystemHandler {
    /// Map of file handles by ID
    file_handles: RwLock<HashMap<u32, FileHandle>>,
    /// Map of file IDs by path
    path_to_id: RwLock<HashMap<PathBuf, u32>>,
    /// Map of open file handles
    open_files: RwLock<HashMap<u32, Arc<RwLock<File>>>>,
    /// Next file ID
    next_id: Arc<RwLock<u32>>,
    /// Base directory for relative paths
    base_dir: PathBuf,
}

impl FileSystemHandler {
    /// Creates a new file system handler.
    ///
    /// # Arguments
    ///
    /// * `base_dir` - Base directory for relative paths
    ///
    /// # Returns
    ///
    /// A new FileSystemHandler instance or an error if the base directory doesn't exist
    pub fn new<P: AsRef<Path>>(base_dir: P) -> FileSystemResult<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();

        if !base_dir.exists() {
            return Err(FileSystemError::PathNotExist);
        }

        Ok(Self {
            file_handles: RwLock::new(HashMap::new()),
            path_to_id: RwLock::new(HashMap::new()),
            open_files: RwLock::new(HashMap::new()),
            next_id: Arc::new(RwLock::new(1)),
            base_dir,
        })
    }

    /// Opens a file and returns its handle.
    ///
    /// # Arguments
    ///
    /// * `path` - File path (relative to base directory or absolute)
    ///
    /// # Returns
    ///
    /// File handle or error
    pub fn open_file<P: AsRef<Path>>(&self, path: P) -> FileSystemResult<FileHandle> {
        let full_path = self.resolve_path(path)?;

        let path_to_id = self
            .path_to_id
            .read()
            .map_err(|_| FileSystemError::InvalidHandle)?;
        if path_to_id.contains_key(&full_path) {
            return Err(FileSystemError::AlreadyOpen);
        }

        if !full_path.exists() {
            return Err(FileSystemError::FileNotFound);
        }

        let metadata = fs::metadata(&full_path)
            .map_err(|e| FileSystemError::IoError(e.to_string()))?;

        let modified_time = metadata
            .modified()
            .map_err(|e| FileSystemError::IoError(e.to_string()))?
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(std::time::Duration::from_secs(0))
            .as_secs();

        let open_files = self
            .open_files
            .read()
            .map_err(|_| FileSystemError::InvalidHandle)?;
        if open_files.len() >= MAX_OPEN_FILES {
            return Err(FileSystemError::MaxFilesReached);
        }
        let id = {
            let mut next_id = self
                .next_id
                .write()
                .map_err(|_| FileSystemError::InvalidHandle)?;
            let id = *next_id;
            *next_id = id.wrapping_add(1);
            id
        };
        let file = File::open(&full_path)
            .map_err(|e| FileSystemError::IoError(e.to_string()))?;
        let file_handle =
            FileHandle::new(id, full_path.clone(), metadata.len(), modified_time);

        self.file_handles
            .write()
            .map_err(|_| FileSystemError::InvalidHandle)?
            .insert(id, file_handle.clone());
        self.path_to_id
            .write()
            .map_err(|_| FileSystemError::InvalidHandle)?
            .insert(full_path, id);
        self.open_files
            .write()
            .map_err(|_| FileSystemError::InvalidHandle)?
            .insert(id, Arc::new(RwLock::new(file)));

        Ok(file_handle)
    }

    /// Creates a new file and returns its handle.
    ///
    /// # Arguments
    ///
    /// * `path` - File path (relative to base directory or absolute)
    ///
    /// # Returns
    ///
    /// File handle or error
    pub fn create_file<P: AsRef<Path>>(&self, path: P) -> FileSystemResult<FileHandle> {
        let full_path = self.resolve_path(path)?;
        if full_path.exists() {
            return Err(FileSystemError::AlreadyOpen);
        }

        let open_files = self
            .open_files
            .read()
            .map_err(|_| FileSystemError::InvalidHandle)?;
        if open_files.len() >= MAX_OPEN_FILES {
            return Err(FileSystemError::MaxFilesReached);
        }
        let id = {
            let mut next_id = self
                .next_id
                .write()
                .map_err(|_| FileSystemError::InvalidHandle)?;
            let id = *next_id;
            *next_id = id.wrapping_add(1);
            id
        };

        let file = File::create(&full_path)
            .map_err(|e| FileSystemError::IoError(e.to_string()))?;

        let metadata = fs::metadata(&full_path)
            .map_err(|e| FileSystemError::IoError(e.to_string()))?;

        let modified_time = metadata
            .modified()
            .map_err(|e| FileSystemError::IoError(e.to_string()))?
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(std::time::Duration::from_secs(0))
            .as_secs();

        let file_handle =
            FileHandle::new(id, full_path.clone(), metadata.len(), modified_time);

        self.file_handles
            .write()
            .map_err(|_| FileSystemError::InvalidHandle)?
            .insert(id, file_handle.clone());
        self.path_to_id
            .write()
            .map_err(|_| FileSystemError::InvalidHandle)?
            .insert(full_path, id);
        self.open_files
            .write()
            .map_err(|_| FileSystemError::InvalidHandle)?
            .insert(id, Arc::new(RwLock::new(file)));

        Ok(file_handle)
    }

    /// Closes a file by its ID.
    ///
    /// # Arguments
    ///
    /// * `id` - File ID
    ///
    /// # Returns
    ///
    /// Result indicating success or failure
    pub fn close_file(&self, id: u32) -> FileSystemResult<()> {
        let file_handle = self
            .file_handles
            .read()
            .map_err(|_| FileSystemError::InvalidHandle)?
            .get(&id)
            .cloned()
            .ok_or(FileSystemError::InvalidHandle)?;

        self.file_handles
            .write()
            .map_err(|_| FileSystemError::InvalidHandle)?
            .remove(&id);
        self.path_to_id
            .write()
            .map_err(|_| FileSystemError::InvalidHandle)?
            .remove(&file_handle.path);
        self.open_files
            .write()
            .map_err(|_| FileSystemError::InvalidHandle)?
            .remove(&id);

        Ok(())
    }

    /// Reads data from a file.
    ///
    /// # Arguments
    ///
    /// * `id` - File ID
    /// * `offset` - Byte offset to start reading
    /// * `buffer` - Buffer to read into
    ///
    /// # Returns
    ///
    /// Number of bytes read or error
    pub fn read_file(
        &self,
        id: u32,
        offset: u64,
        buffer: &mut [u8],
    ) -> FileSystemResult<usize> {
        let file = self
            .open_files
            .read()
            .map_err(|_| FileSystemError::InvalidHandle)?
            .get(&id)
            .cloned()
            .ok_or(FileSystemError::InvalidHandle)?;

        let mut file_guard = file.write().map_err(|_| FileSystemError::InvalidHandle)?;
        file_guard
            .seek(SeekFrom::Start(offset))
            .map_err(|e| FileSystemError::IoError(e.to_string()))?;

        let bytes_read = file_guard
            .read(buffer)
            .map_err(|e| FileSystemError::IoError(e.to_string()))?;

        Ok(bytes_read)
    }

    /// Writes data to a file.
    ///
    /// # Arguments
    ///
    /// * `id` - File ID
    /// * `offset` - Byte offset to start writing
    /// * `data` - Data to write
    ///
    /// # Returns
    ///
    /// Number of bytes written or error
    pub fn write_file(
        &self,
        id: u32,
        offset: u64,
        data: &[u8],
    ) -> FileSystemResult<usize> {
        let file = self
            .open_files
            .read()
            .map_err(|_| FileSystemError::InvalidHandle)?
            .get(&id)
            .cloned()
            .ok_or(FileSystemError::InvalidHandle)?;

        let mut file_guard = file.write().map_err(|_| FileSystemError::InvalidHandle)?;
        file_guard
            .seek(SeekFrom::Start(offset))
            .map_err(|e| FileSystemError::IoError(e.to_string()))?;

        let bytes_written = file_guard
            .write(data)
            .map_err(|e| FileSystemError::IoError(e.to_string()))?;

        let mut file_handles = self
            .file_handles
            .write()
            .map_err(|_| FileSystemError::InvalidHandle)?;
        if let Some(handle) = file_handles.get_mut(&id) {
            let new_size = offset.saturating_add(bytes_written as u64);
            if new_size > handle.size {
                handle.size = new_size;
            }
        }

        Ok(bytes_written)
    }

    /// Gets a file handle by ID.
    ///
    /// # Arguments
    ///
    /// * `id` - File ID
    ///
    /// # Returns
    ///
    /// File handle or error
    pub fn file_handle(&self, id: u32) -> FileSystemResult<FileHandle> {
        self.file_handles
            .read()
            .map_err(|_| FileSystemError::InvalidHandle)?
            .get(&id)
            .cloned()
            .ok_or(FileSystemError::InvalidHandle)
    }

    /// Gets a file handle by path.
    ///
    /// # Arguments
    ///
    /// * `path` - File path
    ///
    /// # Returns
    ///
    /// File handle or error
    pub fn file_handle_by_path<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> FileSystemResult<FileHandle> {
        let full_path = self.resolve_path(path)?;
        let id = self
            .path_to_id
            .read()
            .map_err(|_| FileSystemError::InvalidHandle)?
            .get(&full_path)
            .cloned()
            .ok_or(FileSystemError::FileNotFound)?;

        self.file_handle(id)
    }

    /// Returns the number of open files.
    pub fn open_file_count(&self) -> usize {
        match self.open_files.read() {
            Ok(guard) => guard.len(),
            Err(_) => 0,
        }
    }

    /// Returns all open file handles.
    pub fn all_file_handles(&self) -> Vec<FileHandle> {
        match self.file_handles.read() {
            Ok(guard) => guard.values().cloned().collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Resolves a path to an absolute path.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to resolve
    ///
    /// # Returns
    ///
    /// Resolved absolute path or error
    fn resolve_path<P: AsRef<Path>>(&self, path: P) -> FileSystemResult<PathBuf> {
        let path = path.as_ref();

        if path.is_absolute() {
            Ok(path.to_path_buf())
        } else {
            let full_path = self.base_dir.join(path);
            if !full_path.starts_with(&self.base_dir) {
                return Err(FileSystemError::PathNotExist);
            }
            Ok(full_path)
        }
    }
}
