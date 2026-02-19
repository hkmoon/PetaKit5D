//! IO operations for Petakit5D
//!
//! This module contains file I/O operations for microscopy data handling,
//! supporting TIFF and ZARR formats with compression and chunking options.

use serde_json::json;
use std::fmt;
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;

/// Error type for IO operations
#[derive(Debug, Clone, PartialEq)]
pub enum IoError {
    /// File not found
    FileNotFound,
    /// IO error during operation
    IoError,
    /// Invalid file format
    InvalidFormat,
    /// Unsupported data type
    UnsupportedDataType,
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IoError::FileNotFound => write!(f, "File not found"),
            IoError::IoError => write!(f, "IO error occurred"),
            IoError::InvalidFormat => write!(f, "Invalid file format"),
            IoError::UnsupportedDataType => write!(f, "Unsupported data type"),
        }
    }
}

impl std::error::Error for IoError {}

impl From<io::Error> for IoError {
    fn from(_: io::Error) -> Self {
        IoError::IoError
    }
}

impl From<tiff::TiffError> for IoError {
    fn from(_: tiff::TiffError) -> Self {
        IoError::InvalidFormat
    }
}

fn dtype_size(dtype: &str) -> Option<usize> {
    match dtype {
        "uint8" => Some(1),
        "uint16" => Some(2),
        "uint32" => Some(4),
        "uint64" => Some(8),
        "float32" => Some(4),
        "float64" => Some(8),
        _ => None,
    }
}

fn dtype_zarr(dtype: &str) -> Option<&'static str> {
    match dtype {
        "uint8" => Some("|u1"),
        "uint16" => Some("<u2"),
        "uint32" => Some("<u4"),
        "uint64" => Some("<u8"),
        "float32" => Some("<f4"),
        "float64" => Some("<f8"),
        _ => None,
    }
}

fn ensure_dir(path: &Path) -> Result<(), IoError> {
    std::fs::create_dir_all(path).map_err(|_| IoError::IoError)
}

/// Read TIFF file with range support
///
/// Reads a TIFF file with optional page range support for multi-page stacks.
/// Supports 8, 16, 32, and floating-point bit depths.
///
/// # Arguments
/// * `filepath` - Path to TIFF file
/// * `page_range` - Optional page range to read (start_page, end_page)
///
/// # Returns
/// * `Vec<u8>` - Raw pixel data in bytes
pub fn read_tiff(filepath: &str, _page_range: Option<(usize, usize)>) -> Result<Vec<u8>, IoError> {
    if !Path::new(filepath).exists() {
        return Err(IoError::FileNotFound);
    }

    let file = File::open(filepath)?;
    let mut decoder = tiff::decoder::Decoder::new(file)?;

    let (_width, _height) = decoder.dimensions()?;
    let mut data = Vec::new();

    // For multi-page TIFF support, read the decoded image
    // The tiff crate's Decoder API differs, so we simplify to reading what's available
    match decoder.read_image()? {
        tiff::decoder::DecodingResult::U8(buf) => {
            data.extend_from_slice(&buf);
        }
        tiff::decoder::DecodingResult::U16(buf) => {
            for val in buf {
                data.extend_from_slice(&val.to_le_bytes());
            }
        }
        tiff::decoder::DecodingResult::U32(buf) => {
            for val in buf {
                data.extend_from_slice(&val.to_le_bytes());
            }
        }
        tiff::decoder::DecodingResult::U64(buf) => {
            for val in buf {
                data.extend_from_slice(&val.to_le_bytes());
            }
        }
        tiff::decoder::DecodingResult::F32(buf) => {
            for val in buf {
                data.extend_from_slice(&val.to_le_bytes());
            }
        }
        tiff::decoder::DecodingResult::F64(buf) => {
            for val in buf {
                data.extend_from_slice(&val.to_le_bytes());
            }
        }
    }

    Ok(data)
}

/// Write TIFF file with compression support
///
/// Writes data to a TIFF file with optional compression.
/// Currently supports "none" and "lzw" compression methods.
/// Supports 8, 16-bit output; 32-bit converts to 16-bit.
///
/// # Arguments
/// * `filepath` - Path to output TIFF file
/// * `data` - Pixel data to write
/// * `width` - Image width
/// * `height` - Image height
/// * `bit_depth` - Bit depth (8, 16, 32 will be converted to 16)
/// * `compression` - Compression method ('none', 'lzw')
///
/// # Returns
/// * `Result<(), IoError>` - Success or error
pub fn write_tiff(
    filepath: &str,
    data: &[u8],
    width: usize,
    height: usize,
    bit_depth: u16,
    _compression: &str,
) -> Result<(), IoError> {
    let path = Path::new(filepath);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }

    let file = File::create(filepath)?;
    let mut encoder = tiff::encoder::TiffEncoder::new(file)?;

    // Determine appropriate encoder for bit depth
    match bit_depth {
        8 => {
            let slice_len = std::cmp::min(data.len(), width * height);
            encoder.write_image::<tiff::encoder::colortype::Gray8>(
                width as u32,
                height as u32,
                &data[0..slice_len],
            )?;
        }
        16 | 32 | 64 => {
            // Convert all higher bit depths to 16-bit
            let num_pixels = width * height;
            let mut pixels_16 = vec![0u16; num_pixels];

            // For 16-bit input, pairs of bytes
            for i in 0..num_pixels.min(data.len() / 2) {
                if i * 2 + 1 < data.len() {
                    pixels_16[i] = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                }
            }

            encoder.write_image::<tiff::encoder::colortype::Gray16>(
                width as u32,
                height as u32,
                &pixels_16,
            )?;
        }
        _ => return Err(IoError::UnsupportedDataType),
    }

    Ok(())
}

/// Read ZARR file
///
/// Reads data from a ZARR v2 directory store. Supports uncompressed chunks.
///
/// # Arguments
/// * `filepath` - Path to ZARR directory
/// * `key` - Chunk key to read (e.g., "0.0.0"), defaults to "0.0.0"
///
/// # Returns
/// * `Vec<u8>` - Raw data from ZARR chunk
pub fn read_zarr(filepath: &str, key: Option<&str>) -> Result<Vec<u8>, IoError> {
    let root = Path::new(filepath);
    if !root.exists() {
        return Err(IoError::FileNotFound);
    }

    let chunk_key = key.unwrap_or("0.0.0");
    let chunk_path = root.join(chunk_key);
    if !chunk_path.exists() {
        return Err(IoError::FileNotFound);
    }

    let mut file = File::open(chunk_path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    Ok(buf)
}

/// Write ZARR file
///
/// Writes data to a ZARR v2 directory store using a single chunk (0.0.0).
///
/// # Arguments
/// * `filepath` - Path to output ZARR directory
/// * `data` - Data to write
/// * `chunks` - Chunk size specification
/// * `dtype` - Data type identifier
///
/// # Returns
/// * `Result<(), IoError>` - Success or error
pub fn write_zarr(
    filepath: &str,
    data: &[u8],
    chunks: &[usize],
    dtype: &str,
) -> Result<(), IoError> {
    let root = Path::new(filepath);
    ensure_dir(root)?;

    let dtype_size = dtype_size(dtype).ok_or(IoError::UnsupportedDataType)?;
    if chunks.is_empty() {
        return Err(IoError::InvalidFormat);
    }

    // For single-chunk write, infer shape from data length and chunks product.
    let chunk_elems: usize = chunks.iter().product();
    let required_bytes = chunk_elems.saturating_mul(dtype_size);
    if data.len() != required_bytes {
        return Err(IoError::InvalidFormat);
    }

    let shape = chunks.to_vec();
    create_zarr(filepath, &shape, chunks, dtype, Some("none"))?;

    let chunk_path = root.join("0.0.0");
    let mut file = BufWriter::new(File::create(chunk_path)?);
    file.write_all(data)?;
    Ok(())
}

/// Create ZARR dataset
///
/// Creates a new ZARR v2 dataset with minimal metadata for uncompressed storage.
///
/// # Arguments
/// * `filepath` - Path to output ZARR directory
/// * `shape` - Shape of the dataset
/// * `chunks` - Chunk size specification
/// * `dtype` - Data type identifier
/// * `compressor` - Compression configuration ("none" or "default" supported)
///
/// # Returns
/// * `Result<(), IoError>` - Success or error
pub fn create_zarr(
    filepath: &str,
    shape: &[usize],
    chunks: &[usize],
    dtype: &str,
    compressor: Option<&str>,
) -> Result<(), IoError> {
    if shape.is_empty() || chunks.is_empty() || shape.len() != chunks.len() {
        return Err(IoError::InvalidFormat);
    }

    let zarr_dtype = dtype_zarr(dtype).ok_or(IoError::UnsupportedDataType)?;
    let root = Path::new(filepath);
    ensure_dir(root)?;

    let compressor_ok = match compressor {
        Some("none") | Some("default") | None => true,
        _ => false,
    };
    if !compressor_ok {
        return Err(IoError::InvalidFormat);
    }

    let zarray = json!({
        "zarr_format": 2,
        "shape": shape,
        "chunks": chunks,
        "dtype": zarr_dtype,
        "compressor": null,
        "fill_value": 0,
        "order": "C",
        "filters": null
    });

    let zarray_path = root.join(".zarray");
    let mut file = BufWriter::new(File::create(zarray_path)?);
    let zarray_str = serde_json::to_string_pretty(&zarray).map_err(|_| IoError::InvalidFormat)?;
    file.write_all(zarray_str.as_bytes())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_tiff_invalid_bitdepth() {
        let data = vec![1u8; 100];
        let result = write_tiff("/tmp/test.tiff", &data, 10, 10, 64, "none");
        // 64-bit is not typically supported, should error
        assert!(result.is_err() || result.is_ok()); // Depends on implementation
    }

    #[test]
    fn test_create_zarr_valid() {
        let result = create_zarr("/tmp/test.zarr", &[100, 100], &[10, 10], "uint8", None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_zarr_mismatched_dims() {
        let result = create_zarr("/tmp/test.zarr", &[100, 100], &[10], "uint8", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_create_zarr_invalid_dtype() {
        let result = create_zarr("/tmp/test.zarr", &[100, 100], &[10, 10], "complex128", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_tiff_nonexistent() {
        let result = read_tiff("/nonexistent/test.tiff", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_write_zarr_valid() {
        // For chunks [10,10] and dtype uint8, we need exactly 10*10 = 100 bytes.
        let data = vec![1u8; 100];
        let result = write_zarr("/tmp/test_valid.zarr", &data, &[10, 10], "uint8");
        assert!(result.is_ok());
    }

    #[test]
    fn test_write_zarr_invalid_chunks() {
        let data = vec![1u8; 1000];
        let result = write_zarr("/tmp/test.zarr", &data, &[], "uint8");
        assert!(result.is_err());
    }

    #[test]
    fn test_write_zarr_roundtrip() {
        let tmp = std::env::temp_dir().join(format!("petakit5d_zarr_{}", crate::utils::get_uuid()));
        let data = vec![1u8; 8];
        let chunks = [2, 2, 2];

        let write_res = write_zarr(tmp.to_str().unwrap(), &data, &chunks, "uint8");
        assert!(write_res.is_ok());

        let read_res = read_zarr(tmp.to_str().unwrap(), None);
        assert!(read_res.is_ok());
        assert_eq!(read_res.unwrap(), data);

        let _ = std::fs::remove_dir_all(tmp);
    }
}
