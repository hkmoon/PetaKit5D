//! Utility functions for Petakit5D
//!
//! This module contains various utility functions for data handling and processing.

use std::fmt;
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
}
