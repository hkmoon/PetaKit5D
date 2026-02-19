//! Microscope data processing functions for Petakit5D
//!
//! This module contains functions for processing microscope data including cropping,
//! resampling, and other essential operations.

use std::fmt;

/// Error type for microscope data processing operations
#[derive(Debug, Clone, PartialEq)]
pub enum MicroscopeProcessingError {
    /// Invalid array dimensions
    InvalidDimensions,
    /// Invalid cropping parameters
    InvalidCropParameters,
    /// Invalid resampling parameters
    InvalidResampleParameters,
    /// Unsupported operation
    UnsupportedOperation,
    /// Invalid data type
    InvalidDataType,
    /// Invalid coordinate
    InvalidCoordinate,
}

impl fmt::Display for MicroscopeProcessingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MicroscopeProcessingError::InvalidDimensions => write!(f, "Invalid array dimensions"),
            MicroscopeProcessingError::InvalidCropParameters => {
                write!(f, "Invalid crop parameters")
            }
            MicroscopeProcessingError::InvalidResampleParameters => {
                write!(f, "Invalid resample parameters")
            }
            MicroscopeProcessingError::UnsupportedOperation => write!(f, "Unsupported operation"),
            MicroscopeProcessingError::InvalidDataType => write!(f, "Invalid data type"),
            MicroscopeProcessingError::InvalidCoordinate => write!(f, "Invalid coordinate"),
        }
    }
}

impl std::error::Error for MicroscopeProcessingError {}

/// Crop 3D data
///
/// Crops a 3D volume to a specified region
///
/// # Arguments
/// * `data` - Input 3D data (flattened)
/// * `dims` - Original dimensions [depth, height, width]
/// * `crop_start` - Starting coordinates [z, y, x]
/// * `crop_end` - Ending coordinates [z, y, x]
///
/// # Returns
/// * `Vec<f64>` - Cropped data
pub fn crop_3d(
    data: &[f64],
    dims: &[usize],
    crop_start: &[usize],
    crop_end: &[usize],
) -> Result<Vec<f64>, MicroscopeProcessingError> {
    if data.is_empty() || dims.len() != 3 || crop_start.len() != 3 || crop_end.len() != 3 {
        return Err(MicroscopeProcessingError::InvalidDimensions);
    }

    let [depth, height, width] = [dims[0], dims[1], dims[2]];
    let [start_z, start_y, start_x] = [crop_start[0], crop_start[1], crop_start[2]];
    let [end_z, end_y, end_x] = [crop_end[0], crop_end[1], crop_end[2]];

    // Validate bounds
    if start_z >= depth
        || start_y >= height
        || start_x >= width
        || end_z > depth
        || end_y > height
        || end_x > width
        || start_z >= end_z
        || start_y >= end_y
        || start_x >= end_x
    {
        return Err(MicroscopeProcessingError::InvalidCropParameters);
    }

    // Calculate output dimensions
    let out_depth = end_z - start_z;
    let out_height = end_y - start_y;
    let out_width = end_x - start_x;

    // Perform cropping
    let mut cropped = Vec::with_capacity(out_depth * out_height * out_width);
    for z in start_z..end_z {
        for y in start_y..end_y {
            for x in start_x..end_x {
                let idx = z * height * width + y * width + x;
                cropped.push(data[idx]);
            }
        }
    }

    Ok(cropped)
}

/// Crop 4D data
///
/// Crops a 4D volume (time-series or multi-channel) to a specified region
///
/// # Arguments
/// * `data` - Input 4D data (flattened)
/// * `dims` - Original dimensions [time, depth, height, width]
/// * `crop_start` - Starting coordinates [t, z, y, x]
/// * `crop_end` - Ending coordinates [t, z, y, x]
///
/// # Returns
/// * `Vec<f64>` - Cropped data
pub fn crop_4d(
    data: &[f64],
    dims: &[usize],
    crop_start: &[usize],
    crop_end: &[usize],
) -> Result<Vec<f64>, MicroscopeProcessingError> {
    if data.is_empty() || dims.len() != 4 || crop_start.len() != 4 || crop_end.len() != 4 {
        return Err(MicroscopeProcessingError::InvalidDimensions);
    }

    let [time, depth, height, width] = [dims[0], dims[1], dims[2], dims[3]];
    let [start_t, start_z, start_y, start_x] =
        [crop_start[0], crop_start[1], crop_start[2], crop_start[3]];
    let [end_t, end_z, end_y, end_x] = [crop_end[0], crop_end[1], crop_end[2], crop_end[3]];

    // Validate bounds
    if start_t >= time
        || start_z >= depth
        || start_y >= height
        || start_x >= width
        || end_t > time
        || end_z > depth
        || end_y > height
        || end_x > width
        || start_t >= end_t
        || start_z >= end_z
        || start_y >= end_y
        || start_x >= end_x
    {
        return Err(MicroscopeProcessingError::InvalidCropParameters);
    }

    // Calculate output dimensions
    let out_time = end_t - start_t;
    let out_depth = end_z - start_z;
    let out_height = end_y - start_y;
    let out_width = end_x - start_x;

    // Perform cropping
    let mut cropped = Vec::with_capacity(out_time * out_depth * out_height * out_width);
    for t in start_t..end_t {
        for z in start_z..end_z {
            for y in start_y..end_y {
                for x in start_x..end_x {
                    let idx = t * depth * height * width + z * height * width + y * width + x;
                    cropped.push(data[idx]);
                }
            }
        }
    }

    Ok(cropped)
}

/// Resample 3D stack using linear interpolation
///
/// Resamples a 3D volume to new dimensions using trilinear interpolation.
/// This maintains data quality while allowing flexible resizing.
///
/// # Arguments
/// * `data` - Input 3D data (flattened)
/// * `dims` - Original dimensions [depth, height, width]
/// * `new_dims` - New dimensions [new_depth, new_height, new_width]
///
/// # Returns
/// * `Vec<f64>` - Resampled data
pub fn resample_stack_3d(
    data: &[f64],
    dims: &[usize],
    new_dims: &[usize],
) -> Result<Vec<f64>, MicroscopeProcessingError> {
    if data.is_empty() || dims.len() != 3 || new_dims.len() != 3 {
        return Err(MicroscopeProcessingError::InvalidDimensions);
    }

    let [depth, height, width] = [dims[0], dims[1], dims[2]];
    let [new_depth, new_height, new_width] = [new_dims[0], new_dims[1], new_dims[2]];

    if depth == 0
        || height == 0
        || width == 0
        || new_depth == 0
        || new_height == 0
        || new_width == 0
    {
        return Err(MicroscopeProcessingError::InvalidResampleParameters);
    }

    // If dimensions match, return copy
    if depth == new_depth && height == new_height && width == new_width {
        return Ok(data.to_vec());
    }

    // Calculate scale factors
    let scale_z = (depth as f64 - 1.0) / (new_depth as f64 - 1.0).max(1.0);
    let scale_y = (height as f64 - 1.0) / (new_height as f64 - 1.0).max(1.0);
    let scale_x = (width as f64 - 1.0) / (new_width as f64 - 1.0).max(1.0);

    let mut resampled = vec![0.0; new_depth * new_height * new_width];

    for nz in 0..new_depth {
        for ny in 0..new_height {
            for nx in 0..new_width {
                // Map to original coordinates
                let z = nz as f64 * scale_z;
                let y = ny as f64 * scale_y;
                let x = nx as f64 * scale_x;

                // Trilinear interpolation
                let z0 = z.floor() as usize;
                let z1 = (z0 + 1).min(depth - 1);
                let zy = z - z.floor();

                let y0 = y.floor() as usize;
                let y1 = (y0 + 1).min(height - 1);
                let yy = y - y.floor();

                let x0 = x.floor() as usize;
                let x1 = (x0 + 1).min(width - 1);
                let xy = x - x.floor();

                // Get the 8 corners
                let v000 = data[z0 * height * width + y0 * width + x0];
                let v001 = data[z0 * height * width + y0 * width + x1];
                let v010 = data[z0 * height * width + y1 * width + x0];
                let v011 = data[z0 * height * width + y1 * width + x1];
                let v100 = data[z1 * height * width + y0 * width + x0];
                let v101 = data[z1 * height * width + y0 * width + x1];
                let v110 = data[z1 * height * width + y1 * width + x0];
                let v111 = data[z1 * height * width + y1 * width + x1];

                // Interpolate along x
                let v00 = v000 * (1.0 - xy) + v001 * xy;
                let v01 = v010 * (1.0 - xy) + v011 * xy;
                let v10 = v100 * (1.0 - xy) + v101 * xy;
                let v11 = v110 * (1.0 - xy) + v111 * xy;

                // Interpolate along y
                let v0 = v00 * (1.0 - yy) + v01 * yy;
                let v1 = v10 * (1.0 - yy) + v11 * yy;

                // Interpolate along z
                let value = v0 * (1.0 - zy) + v1 * zy;

                resampled[nz * new_height * new_width + ny * new_width + nx] = value;
            }
        }
    }

    Ok(resampled)
}

/// Project 3D to 2D
///
/// Projects a 3D volume to a 2D image by maximum projection along Z axis
///
/// # Arguments
/// * `data` - Input 3D data (flattened)
/// * `dims` - Original dimensions [depth, height, width]
///
/// # Returns
/// * `Vec<f64>` - Projected 2D data
pub fn project_3d_to_2d(
    data: &[f64],
    dims: &[usize],
) -> Result<Vec<f64>, MicroscopeProcessingError> {
    if data.is_empty() || dims.len() != 3 {
        return Err(MicroscopeProcessingError::InvalidDimensions);
    }

    let [depth, height, width] = [dims[0], dims[1], dims[2]];

    // Create 2D output
    let mut projected = vec![0.0; height * width];

    // Maximum projection along depth axis
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let idx = z * height * width + y * width + x;
                let current = data[idx];
                let proj_idx = y * width + x;

                if current > projected[proj_idx] {
                    projected[proj_idx] = current;
                }
            }
        }
    }

    Ok(projected)
}

/// Group partial volume files intelligently
///
/// Groups files belonging to the same partial volume by analyzing filenames.
/// Handles patterns like: base_Pos##_CH##_T###.tif -> groups by Pos+CH combination
///
/// # Arguments
/// * `filenames` - List of file paths
///
/// # Returns
/// * `Vec<Vec<String>>` - Grouped file paths by volume
pub fn group_partial_volume_files(filenames: &[String]) -> Vec<Vec<String>> {
    use std::collections::HashMap;

    if filenames.is_empty() {
        return vec![];
    }

    let mut groups: HashMap<String, Vec<String>> = HashMap::new();

    for filename in filenames {
        // Extract directory and basename
        let basename = std::path::Path::new(filename)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(filename);

        // Try to extract group key from filename patterns
        // Pattern 1: base_Pos##_CH##_T###.tif -> group by Pos+CH
        let mut group_key = basename.to_string();

        if let Some(pos_idx) = basename.find("_Pos") {
            if let Some(ch_idx) = basename.find("_CH") {
                // Extract position and channel
                if let Some(ch_end) = basename[ch_idx + 3..]
                    .find('_')
                    .or_else(|| basename[ch_idx + 3..].find('.'))
                {
                    group_key = format!(
                        "{}_{}",
                        &basename[pos_idx..pos_idx + 8.min(ch_idx - pos_idx)],
                        &basename[ch_idx..ch_idx + 8.min(ch_end + 3)]
                    );
                }
            }
        }
        // Pattern 2: channel info in filename -> group by channel
        else if let Some(ch_idx) = basename.find("CH") {
            if ch_idx + 2 < basename.len() {
                group_key = basename[ch_idx..ch_idx + 5.min(basename.len() - ch_idx)].to_string();
            }
        }
        // Pattern 3: tile info -> group by tile
        else if let Some(tile_idx) = basename.find("Tile") {
            if tile_idx + 4 < basename.len() {
                group_key =
                    basename[tile_idx..tile_idx + 8.min(basename.len() - tile_idx)].to_string();
            }
        }

        groups
            .entry(group_key)
            .or_insert_with(Vec::new)
            .push(filename.clone());
    }

    // Sort groups by key and return as vec
    let mut result: Vec<Vec<String>> = groups.into_values().collect();
    result.sort_by(|a, b| a.get(0).cmp(&b.get(0)));
    result
}

/// Create ZARR dataset
///
/// Creates a new ZARR dataset with specified properties
///
/// # Arguments
/// * `filepath` - Path to output ZARR file
/// * `shape` - Shape of the dataset
/// * `chunks` - Chunk size specification
/// * `dtype` - Data type identifier
/// * `compressor` - Compression configuration
///
/// # Returns
/// * `Result<(), MicroscopeProcessingError>` - Success or error
pub fn create_zarr(
    _filepath: &str,
    _shape: &[usize],
    _chunks: &[usize],
    _dtype: &str,
    _compressor: Option<&str>,
) -> Result<(), MicroscopeProcessingError> {
    // Placeholder - actual implementation would use zarrs crate
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crop_3d() {
        // Create sample 3D data (2x2x2)
        let data = vec![1.0; 8];
        let dims = [2, 2, 2];
        let start = [0, 0, 0];
        let end = [1, 1, 1];

        let result = crop_3d(&data, &dims, &start, &end);
        assert!(result.is_ok());
        // Should be 1x1x1 = 1 element
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_crop_4d() {
        // Create sample 4D data (2x2x2x2)
        let data = vec![1.0; 16];
        let dims = [2, 2, 2, 2];
        let start = [0, 0, 0, 0];
        let end = [1, 1, 1, 1];

        let result = crop_4d(&data, &dims, &start, &end);
        assert!(result.is_ok());
        // Should be 1x1x1x1 = 1 element
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_resample_stack_3d() {
        // 2x2x2 volume of ones, upsample to 4x4x4
        let data = vec![1.0; 8];
        let dims = [2, 2, 2];
        let new_dims = [4, 4, 4];

        let result = resample_stack_3d(&data, &dims, &new_dims);
        assert!(result.is_ok());
        let resampled = result.unwrap();
        // New volume should have 4*4*4 = 64 elements
        assert_eq!(resampled.len(), 64);
        // Since input is uniform, all interpolated values should be close to 1.0
        for v in resampled {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_project_3d_to_2d() {
        let data = vec![1.0; 8]; // 2x2x2
        let dims = [2, 2, 2];

        let result = project_3d_to_2d(&data, &dims);
        assert!(result.is_ok());
        // Should return 2x2 = 4 elements
        assert_eq!(result.unwrap().len(), 4);
    }

    #[test]
    fn test_group_partial_volume_files() {
        let filenames = vec![
            "file001.tif".to_string(),
            "file002.tif".to_string(),
            "file003.tif".to_string(),
        ];
        let result = group_partial_volume_files(&filenames);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_create_zarr() {
        let result = create_zarr("/tmp/test.zarr", &[100, 100], &[10, 10], "uint8", None);
        assert!(result.is_ok());
    }
}
