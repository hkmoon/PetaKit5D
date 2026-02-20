//! Utility functions for Petakit5D
//!
//! This module contains various utility functions for data handling and processing.

use std::fmt;
use std::io::{self, BufRead, BufWriter, Write};
use std::path::Path;
use uuid::Uuid;

/// Error type for utility operations
#[derive(Debug, Clone, PartialEq)]
pub enum UtilityError {
    /// Invalid UUID format
    InvalidUuidFormat,
    /// Invalid file path
    InvalidFilePath,
    /// File operation failed
    FileOperationFailed,
    /// Invalid data type
    InvalidDataType,
    /// Invalid parameter
    InvalidParameter,
}

impl fmt::Display for UtilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UtilityError::InvalidUuidFormat => write!(f, "Invalid UUID format"),
            UtilityError::InvalidFilePath => write!(f, "Invalid file path"),
            UtilityError::FileOperationFailed => write!(f, "File operation failed"),
            UtilityError::InvalidDataType => write!(f, "Invalid data type"),
            UtilityError::InvalidParameter => write!(f, "Invalid parameter"),
        }
    }
}

impl std::error::Error for UtilityError {}

/// Generate UUID
///
/// Generates a UUID string in standard format using v4 (random)
///
/// # Returns
/// * `String` - UUID string in standard format (8-4-4-4-12)
pub fn get_uuid() -> String {
    Uuid::new_v4().to_string()
}

/// Convert MATLAB-style string to comma-separated
///
/// Converts MATLAB array notation to comma-separated format
/// e.g., "[1 2 3]" -> "1,2,3"
///
/// # Arguments
/// * `input` - Input string
///
/// # Returns
/// * `String` - Converted string
pub fn mat2str_comma(input: &str) -> String {
    input
        .replace("[", "")
        .replace("]", "")
        .trim()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(",")
}

/// Get image data type
///
/// Determines the data type of image data based on value ranges
/// Analyzes the data to determine if it fits in narrower types
///
/// # Arguments
/// * `data` - Image data
///
/// # Returns
/// * `String` - Data type identifier ("uint8", "uint16", "uint32", "float32", or "float64")
pub fn get_image_data_type(data: &[f64]) -> String {
    if data.is_empty() {
        return "unknown".to_string();
    }

    let (min, max) = data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
            (min.min(val), max.max(val))
        });

    // Check if all values are non-negative integers
    let all_nonneg_int = data.iter().all(|&x| x >= 0.0 && x.fract() == 0.0);

    if all_nonneg_int {
        if max <= 255.0 {
            "uint8".to_string()
        } else if max <= 65535.0 {
            "uint16".to_string()
        } else if max <= 4294967295.0 {
            "uint32".to_string()
        } else {
            "float64".to_string()
        }
    } else {
        // Check for float32 range
        if min >= f32::MIN as f64 && max <= f32::MAX as f64 {
            "float32".to_string()
        } else {
            "float64".to_string()
        }
    }
}

/// Get image size
///
/// Gets the dimensions of an image
///
/// # Arguments
/// * `data` - Image data (flattened)
/// * `dims` - Dimensions [height, width] or [depth, height, width]
///
/// # Returns
/// * `Vec<usize>` - Size information
pub fn get_image_size(_data: &[f64], dims: &[usize]) -> Vec<usize> {
    dims.to_vec()
}

/// Create directory recursively
///
/// Creates directories recursively, similar to mkdir -p
///
/// # Arguments
/// * `path` - Directory path to create
///
/// # Returns
/// * `Result<(), UtilityError>` - Success or error
pub fn mkdir_recursive(path: &str) -> Result<(), UtilityError> {
    std::fs::create_dir_all(path).map_err(|_| UtilityError::FileOperationFailed)
}

/// Simplify path
///
/// Simplifies a file path by resolving '.' and '..' components
///
/// # Arguments
/// * `path` - Input path
///
/// # Returns
/// * `String` - Simplified path
pub fn simplify_path(path: &str) -> String {
    let p = Path::new(path);
    p.canonicalize()
        .ok()
        .and_then(|p| p.to_str().map(String::from))
        .unwrap_or_else(|| path.to_string())
}

/// Get hostname
///
/// Gets the system hostname
///
/// # Returns
/// * `String` - Hostname or "localhost" if unavailable
pub fn get_hostname() -> String {
    match hostname::get() {
        Ok(h) => match h.into_string() {
            Ok(s) => s,
            Err(_) => "localhost".to_string(),
        },
        Err(_) => "localhost".to_string(),
    }
}

/// Find good factor number
///
/// Finds prime factors of a number for optimizing FFT-based operations
/// Returns all prime factors to allow flexible combinations for FFT scheduling
///
/// # Arguments
/// * `n` - Number to factor
///
/// # Returns
/// * `Vec<usize>` - Prime factors in ascending order
pub fn find_good_factor_number(n: usize) -> Vec<usize> {
    if n <= 1 {
        return vec![n];
    }

    let mut factors = Vec::new();
    let mut num = n;

    // Factor out 2s
    while num % 2 == 0 {
        factors.push(2);
        num /= 2;
    }

    // Factor out odd numbers starting from 3
    let mut i = 3;
    while i * i <= num {
        while num % i == 0 {
            factors.push(i);
            num /= i;
        }
        i += 2;
    }

    // If num is still > 1, it's a prime factor
    if num > 1 {
        factors.push(num);
    }

    // If empty (shouldn't happen), return original
    if factors.is_empty() {
        factors.push(n);
    }

    factors
}

/// Read a text file line by line
///
/// Reads a text file and returns its lines as a vector of strings,
/// with trailing newline characters stripped.
///
/// # Arguments
/// * `filename` - Path to the text file
///
/// # Returns
/// * `Result<Vec<String>, UtilityError>` - Lines or error
///
/// # Errors
/// Returns `UtilityError::FileOperationFailed` if the file cannot be opened or read.
pub fn read_text_file(filename: &str) -> Result<Vec<String>, UtilityError> {
    let file = std::fs::File::open(filename).map_err(|_| UtilityError::FileOperationFailed)?;
    let reader = io::BufReader::new(file);
    let lines: Result<Vec<String>, _> = reader.lines().collect();
    lines.map_err(|_| UtilityError::FileOperationFailed)
}

/// Write text lines to a file
///
/// Writes a slice of strings to a file, joining them with newlines.
///
/// # Arguments
/// * `lines` - Lines to write
/// * `filename` - Output file path
///
/// # Returns
/// * `Result<(), UtilityError>` - Success or error
pub fn write_text_file(lines: &[String], filename: &str) -> Result<(), UtilityError> {
    let file = std::fs::File::create(filename).map_err(|_| UtilityError::FileOperationFailed)?;
    let mut writer = BufWriter::new(file);
    for (i, line) in lines.iter().enumerate() {
        if i > 0 {
            writer
                .write_all(b"\n")
                .map_err(|_| UtilityError::FileOperationFailed)?;
        }
        writer
            .write_all(line.as_bytes())
            .map_err(|_| UtilityError::FileOperationFailed)?;
    }
    Ok(())
}

/// Write a JSON value to a file (pretty-printed)
///
/// Serializes `data` as pretty-printed JSON and writes it to `filename`.
///
/// # Arguments
/// * `data` - JSON-serializable value (serde_json::Value)
/// * `filename` - Output file path
///
/// # Returns
/// * `Result<(), UtilityError>` - Success or error
pub fn write_json_file(
    data: &serde_json::Value,
    filename: &str,
) -> Result<(), UtilityError> {
    let file = std::fs::File::create(filename).map_err(|_| UtilityError::FileOperationFailed)?;
    let mut writer = BufWriter::new(file);
    let json_str =
        serde_json::to_string_pretty(data).map_err(|_| UtilityError::FileOperationFailed)?;
    writer
        .write_all(json_str.as_bytes())
        .map_err(|_| UtilityError::FileOperationFailed)?;
    Ok(())
}

/// Get byte count for a data type string
///
/// Maps common data type names to their byte sizes.
///
/// # Arguments
/// * `dtype` - Data type string (e.g. `"uint8"`, `"uint16"`, `"float32"`)
///
/// # Returns
/// * `Ok(usize)` - Byte count
/// * `Err(UtilityError::InvalidDataType)` - Unsupported type
pub fn data_type_to_byte_number(dtype: &str) -> Result<usize, UtilityError> {
    match dtype {
        "uint8" => Ok(1),
        "uint16" => Ok(2),
        "single" | "float32" => Ok(4),
        "double" | "float64" => Ok(8),
        _ => Err(UtilityError::InvalidDataType),
    }
}

/// Map axis order indices
///
/// Given an `input_axis_order` string (e.g. `"xyz"`) and an
/// `output_axis_order` string (e.g. `"yxz"`), returns 1-based indices that
/// describe where each input axis appears in the output ordering.
///
/// # Arguments
/// * `input_axis_order` - Three-character string using 'x', 'y', 'z'
/// * `output_axis_order` - Three-character string using 'x', 'y', 'z'
///
/// # Returns
/// * `Ok([usize; 3])` - 1-based mapping indices
/// * `Err(UtilityError::InvalidParameter)` - Invalid axis order string
pub fn axis_order_mapping(
    input_axis_order: &str,
    output_axis_order: &str,
) -> Result<[usize; 3], UtilityError> {
    let valid = ["xyz", "yxz", "zyx", "zxy", "yzx", "xzy"];
    let input_lc = input_axis_order.to_lowercase();
    let output_lc = output_axis_order.to_lowercase();

    if input_lc.len() != 3 || output_lc.len() != 3 {
        return Err(UtilityError::InvalidParameter);
    }
    if !valid.contains(&input_lc.as_str()) || !valid.contains(&output_lc.as_str()) {
        return Err(UtilityError::InvalidParameter);
    }

    let output_chars: Vec<char> = output_lc.chars().collect();
    let mut result = [0usize; 3];
    for (i, ch) in input_lc.chars().enumerate() {
        let pos = output_chars
            .iter()
            .position(|&c| c == ch)
            .ok_or(UtilityError::InvalidParameter)?;
        result[i] = pos + 1; // 1-based
    }
    Ok(result)
}

/// Inverse 1D cubic B-spline interpolation
///
/// Interpolates along the columns of `coeffs` (axis 1) using cubic B-spline
/// basis functions with symmetric (mirror) boundary conditions.
///
/// # Arguments
/// * `coeffs` - B-spline coefficients, row-major order with shape `(ny, nx_coeffs)`
/// * `ny` - Number of rows
/// * `nx_coeffs` - Number of coefficient columns
/// * `nx` - Target number of output columns
///
/// # Returns
/// * `Vec<f64>` - Interpolated data, length `ny * nx`
pub fn ib3spline_1d(coeffs: &[f64], ny: usize, nx_coeffs: usize, nx: usize) -> Vec<f64> {
    if coeffs.is_empty() || ny == 0 || nx_coeffs == 0 || nx == 0 {
        return vec![];
    }

    // Pad coefficients symmetrically with 2 on each side along columns
    let padded_cols = nx_coeffs + 4;
    let mut padded = vec![0.0f64; ny * padded_cols];
    for row in 0..ny {
        // Left pad (mirror): index 1 -> coeffs[row][1], index 0 -> coeffs[row][0]
        padded[row * padded_cols] = coeffs[row * nx_coeffs + 1.min(nx_coeffs - 1)];
        padded[row * padded_cols + 1] = coeffs[row * nx_coeffs];
        for col in 0..nx_coeffs {
            padded[row * padded_cols + col + 2] = coeffs[row * nx_coeffs + col];
        }
        // Right pad
        padded[row * padded_cols + nx_coeffs + 2] =
            coeffs[row * nx_coeffs + nx_coeffs.saturating_sub(1)];
        padded[row * padded_cols + nx_coeffs + 3] =
            coeffs[row * nx_coeffs + nx_coeffs.saturating_sub(2).min(nx_coeffs - 1)];
    }

    let scale = nx_coeffs as f64 / nx as f64;
    let mut output = vec![0.0f64; ny * nx];

    for xi in 0..nx {
        // 1-based scaled position matching Python: (xi+1) / (nx / nx_coeffs)
        let x_scaled = (xi + 1) as f64 / scale;
        let x_floor = x_scaled.floor();
        let t = x_scaled - x_floor;

        // Cubic B-spline weights
        let t3 = (1.0 / 6.0) * t * t * t;
        let w0 = (1.0 / 6.0) + 0.5 * t * (t - 1.0) - t3;
        let w2 = t + w0 - 2.0 * t3;
        let w1 = 1.0 - w0 - w2 - t3;
        let weights = [w0, w1, w2, t3];

        for k in 0..4usize {
            let col_idx = (x_floor as isize + k as isize - 1 + 2) as usize; // +2 for padding
            let col_idx = col_idx.min(padded_cols - 1);
            for row in 0..ny {
                output[row * nx + xi] += weights[k] * padded[row * padded_cols + col_idx];
            }
        }
    }

    output
}

/// Inverse 2D cubic B-spline interpolation
///
/// Applies [`ib3spline_1d`] separably along both axes.
///
/// # Arguments
/// * `coeffs` - B-spline coefficients, row-major order with shape `(ny_c, nx_c)`
/// * `ny_c` - Number of coefficient rows
/// * `nx_c` - Number of coefficient columns
/// * `ny` - Target output rows
/// * `nx` - Target output columns
///
/// # Returns
/// * `Vec<f64>` - Interpolated data, length `ny * nx`
pub fn ib3spline_2d(
    coeffs: &[f64],
    ny_c: usize,
    nx_c: usize,
    ny: usize,
    nx: usize,
) -> Vec<f64> {
    // First interpolate along columns (axis 1)
    let interp_x = ib3spline_1d(coeffs, ny_c, nx_c, nx);
    // Transpose ny_c × nx  ->  nx × ny_c, then interpolate along the new columns
    let mut transposed = vec![0.0f64; nx * ny_c];
    for row in 0..ny_c {
        for col in 0..nx {
            transposed[col * ny_c + row] = interp_x[row * nx + col];
        }
    }
    let interp_y = ib3spline_1d(&transposed, nx, ny_c, ny);
    // Transpose back: nx × ny  ->  ny × nx
    let mut result = vec![0.0f64; ny * nx];
    for row in 0..nx {
        for col in 0..ny {
            result[col * nx + row] = interp_y[row * ny + col];
        }
    }
    result
}

/// Fast integer exponentiation (exponentiation by squaring)
///
/// Computes `base^exponent` using O(log n) multiplications.
/// Negative exponents return `1.0 / base^|exponent|`.
///
/// # Arguments
/// * `base` - Base value
/// * `exponent` - Integer exponent
///
/// # Returns
/// * `f64` - Result
pub fn fast_power(base: f64, exponent: i32) -> f64 {
    if exponent == 0 {
        return 1.0;
    }
    let (mut b, mut exp) = if exponent < 0 {
        (1.0 / base, (-exponent) as u32)
    } else {
        (base, exponent as u32)
    };
    let mut result = 1.0f64;
    while exp > 0 {
        if exp & 1 == 1 {
            result *= b;
        }
        b *= b;
        exp >>= 1;
    }
    result
}

/// Get the bounding box of non-zero regions in a 2D or 3D image
///
/// Returns 1-based indices for MATLAB compatibility.
/// For 2D: `(y1, x1, y2, x2)`
/// For 3D: `(y1, x1, z1, y2, x2, z2)`
///
/// # Arguments
/// * `data` - Flattened image data (row-major)
/// * `dims` - Dimensions: `[height, width]` or `[height, width, depth]`
///
/// # Returns
/// * `Vec<usize>` - Bounding box indices (empty if all-zero image or invalid dims)
pub fn get_image_bounding_box(data: &[f64], dims: &[usize]) -> Vec<usize> {
    match dims.len() {
        2 => {
            let (height, width) = (dims[0], dims[1]);
            if data.len() != height * width {
                return vec![];
            }
            let (mut y1, mut x1) = (height, width);
            let (mut y2, mut x2) = (0usize, 0usize);
            let mut found = false;
            for y in 0..height {
                for x in 0..width {
                    if data[y * width + x] != 0.0 {
                        if y < y1 {
                            y1 = y;
                        }
                        if y > y2 {
                            y2 = y;
                        }
                        if x < x1 {
                            x1 = x;
                        }
                        if x > x2 {
                            x2 = x;
                        }
                        found = true;
                    }
                }
            }
            if !found {
                return vec![0, 0, 0, 0];
            }
            vec![y1 + 1, x1 + 1, y2 + 1, x2 + 1]
        }
        3 => {
            let (height, width, depth) = (dims[0], dims[1], dims[2]);
            if data.len() != height * width * depth {
                return vec![];
            }
            let (mut y1, mut x1, mut z1) = (height, width, depth);
            let (mut y2, mut x2, mut z2) = (0usize, 0usize, 0usize);
            let mut found = false;
            for y in 0..height {
                for x in 0..width {
                    for z in 0..depth {
                        if data[y * width * depth + x * depth + z] != 0.0 {
                            if y < y1 {
                                y1 = y;
                            }
                            if y > y2 {
                                y2 = y;
                            }
                            if x < x1 {
                                x1 = x;
                            }
                            if x > x2 {
                                x2 = x;
                            }
                            if z < z1 {
                                z1 = z;
                            }
                            if z > z2 {
                                z2 = z;
                            }
                            found = true;
                        }
                    }
                }
            }
            if !found {
                return vec![0, 0, 0, 0, 0, 0];
            }
            vec![y1 + 1, x1 + 1, z1 + 1, y2 + 1, x2 + 1, z2 + 1]
        }
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_uuid() {
        let uuid1 = get_uuid();
        let uuid2 = get_uuid();
        assert!(uuid1.len() >= 36); // Standard UUID length
        assert_ne!(uuid1, uuid2); // Different UUIDs
    }

    #[test]
    fn test_mat2str_comma() {
        assert_eq!(mat2str_comma("[1 2 3]"), "1,2,3");
        assert_eq!(mat2str_comma("[100 200]"), "100,200");
        assert_eq!(mat2str_comma("[]"), "");
    }

    #[test]
    fn test_get_image_data_type() {
        let uint8_data = vec![1.0, 100.0, 255.0];
        assert_eq!(get_image_data_type(&uint8_data), "uint8");

        let uint16_data = vec![1.0, 1000.0, 65535.0];
        assert_eq!(get_image_data_type(&uint16_data), "uint16");

        let float_data = vec![1.5, 2.7, 3.14];
        assert_eq!(get_image_data_type(&float_data), "float32");
    }

    #[test]
    fn test_get_image_size() {
        let data = vec![1.0; 100];
        let dims = [10, 10];
        let result = get_image_size(&data, &dims);
        assert_eq!(result, vec![10, 10]);
    }

    #[test]
    fn test_mkdir_recursive() {
        // Create a unique test directory
        let test_dir = format!("/tmp/petakit5d_test_{}", get_uuid());
        let result = mkdir_recursive(&test_dir);
        assert!(result.is_ok());

        // Clean up
        let _ = std::fs::remove_dir(&test_dir);
    }

    #[test]
    fn test_simplify_path() {
        // This test might vary by OS, so we just verify it returns a string
        let result = simplify_path("./test/./file.txt");
        assert!(!result.is_empty());
    }

    #[test]
    fn test_get_hostname() {
        let result = get_hostname();
        assert!(!result.is_empty());
        assert_ne!(result, ""); // Should not be empty
    }

    #[test]
    fn test_find_good_factor_number() {
        let factors_12 = find_good_factor_number(12);
        // 12 = 2 * 2 * 3
        assert_eq!(factors_12, vec![2, 2, 3]);

        let factors_7 = find_good_factor_number(7);
        // 7 is prime
        assert_eq!(factors_7, vec![7]);

        let factors_1 = find_good_factor_number(1);
        assert_eq!(factors_1, vec![1]);

        // Verify product equals original
        let n = 60;
        let factors = find_good_factor_number(n);
        let product: usize = factors.iter().product();
        assert_eq!(product, n);
    }

    #[test]
    fn test_read_write_text_file() {
        let path = format!("/tmp/petakit5d_text_{}.txt", get_uuid());
        let lines = vec!["hello".to_string(), "world".to_string()];
        write_text_file(&lines, &path).unwrap();
        let read_back = read_text_file(&path).unwrap();
        assert_eq!(read_back, lines);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_write_text_file_empty() {
        let path = format!("/tmp/petakit5d_text_empty_{}.txt", get_uuid());
        write_text_file(&[], &path).unwrap();
        let read_back = read_text_file(&path).unwrap();
        assert!(read_back.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_read_text_file_not_found() {
        let result = read_text_file("/nonexistent/path/file.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_write_json_file() {
        let path = format!("/tmp/petakit5d_json_{}.json", get_uuid());
        let data = serde_json::json!({"key": "value", "num": 42});
        write_json_file(&data, &path).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("\"key\""));
        assert!(content.contains("42"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_data_type_to_byte_number() {
        assert_eq!(data_type_to_byte_number("uint8").unwrap(), 1);
        assert_eq!(data_type_to_byte_number("uint16").unwrap(), 2);
        assert_eq!(data_type_to_byte_number("float32").unwrap(), 4);
        assert_eq!(data_type_to_byte_number("single").unwrap(), 4);
        assert_eq!(data_type_to_byte_number("double").unwrap(), 8);
        assert_eq!(data_type_to_byte_number("float64").unwrap(), 8);
        assert!(data_type_to_byte_number("complex128").is_err());
    }

    #[test]
    fn test_axis_order_mapping() {
        assert_eq!(axis_order_mapping("xyz", "yxz").unwrap(), [2, 1, 3]);
        assert_eq!(axis_order_mapping("zyx", "xyz").unwrap(), [3, 2, 1]);
        assert_eq!(axis_order_mapping("yxz", "yxz").unwrap(), [1, 2, 3]);
        assert!(axis_order_mapping("xy", "yxz").is_err());
        assert!(axis_order_mapping("xyz", "abc").is_err());
    }

    #[test]
    fn test_fast_power() {
        assert!((fast_power(2.0, 10) - 1024.0).abs() < 1e-9);
        assert!((fast_power(2.0, -3) - 0.125).abs() < 1e-9);
        assert!((fast_power(3.0, 0) - 1.0).abs() < 1e-9);
        assert!((fast_power(3.0, 1) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_get_image_bounding_box_2d() {
        let mut data = vec![0.0f64; 100]; // 10x10
        data[3 * 10 + 2] = 1.0;
        data[6 * 10 + 7] = 1.0;
        let bbox = get_image_bounding_box(&data, &[10, 10]);
        // 1-based: y1=4, x1=3, y2=7, x2=8
        assert_eq!(bbox, vec![4, 3, 7, 8]);
    }

    #[test]
    fn test_get_image_bounding_box_all_zero() {
        let data = vec![0.0f64; 25]; // 5x5
        let bbox = get_image_bounding_box(&data, &[5, 5]);
        assert_eq!(bbox, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_ib3spline_1d_shape() {
        let coeffs = vec![1.0f64; 10 * 20]; // 10 rows, 20 cols
        let out = ib3spline_1d(&coeffs, 10, 20, 40);
        assert_eq!(out.len(), 10 * 40);
    }

    #[test]
    fn test_ib3spline_2d_shape() {
        let coeffs = vec![1.0f64; 10 * 20]; // 10x20
        let out = ib3spline_2d(&coeffs, 10, 20, 20, 40);
        assert_eq!(out.len(), 20 * 40);
    }
}
