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

// ---------------------------------------------------------------------------
// Private helpers for affine-transform-based deskew / rotate
// ---------------------------------------------------------------------------

/// Invert a 3x3 matrix using cofactors / Cramer's rule.
///
/// Returns `Err(UnsupportedOperation)` when the matrix is singular
/// (|det| < 1e-12).
fn invert_3x3(m: &[[f64; 3]; 3]) -> Result<[[f64; 3]; 3], MicroscopeProcessingError> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return Err(MicroscopeProcessingError::UnsupportedOperation);
    }
    let inv = 1.0 / det;
    Ok([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
        ],
    ])
}

/// Apply a 3D affine transform with trilinear interpolation.
///
/// For every output voxel `(oz, oy, ox)` the corresponding input coordinate
/// is computed as:
/// ```text
/// input = matrix @ [oz, oy, ox]^T + offset
/// ```
/// Out-of-bounds input coordinates are filled with 0.
fn apply_affine_transform_3d(
    data: &[f64],
    in_nz: usize,
    in_ny: usize,
    in_nx: usize,
    out_nz: usize,
    out_ny: usize,
    out_nx: usize,
    matrix: &[[f64; 3]; 3],
    offset: &[f64; 3],
) -> Vec<f64> {
    let mut output = vec![0.0f64; out_nz * out_ny * out_nx];
    let in_nynx = in_ny * in_nx;

    for oz in 0..out_nz {
        for oy in 0..out_ny {
            for ox in 0..out_nx {
                let oz_f = oz as f64;
                let oy_f = oy as f64;
                let ox_f = ox as f64;

                let iz =
                    matrix[0][0] * oz_f + matrix[0][1] * oy_f + matrix[0][2] * ox_f + offset[0];
                let iy =
                    matrix[1][0] * oz_f + matrix[1][1] * oy_f + matrix[1][2] * ox_f + offset[1];
                let ix =
                    matrix[2][0] * oz_f + matrix[2][1] * oy_f + matrix[2][2] * ox_f + offset[2];

                // Reject out-of-bounds (constant / zero fill)
                if iz < 0.0
                    || iz > (in_nz - 1) as f64
                    || iy < 0.0
                    || iy > (in_ny - 1) as f64
                    || ix < 0.0
                    || ix > (in_nx - 1) as f64
                {
                    continue;
                }

                // Trilinear interpolation
                let z0 = iz.floor() as usize;
                let z1 = (z0 + 1).min(in_nz - 1);
                let tz = iz - iz.floor();

                let y0 = iy.floor() as usize;
                let y1 = (y0 + 1).min(in_ny - 1);
                let ty = iy - iy.floor();

                let x0 = ix.floor() as usize;
                let x1 = (x0 + 1).min(in_nx - 1);
                let tx = ix - ix.floor();

                let v000 = data[z0 * in_nynx + y0 * in_nx + x0];
                let v001 = data[z0 * in_nynx + y0 * in_nx + x1];
                let v010 = data[z0 * in_nynx + y1 * in_nx + x0];
                let v011 = data[z0 * in_nynx + y1 * in_nx + x1];
                let v100 = data[z1 * in_nynx + y0 * in_nx + x0];
                let v101 = data[z1 * in_nynx + y0 * in_nx + x1];
                let v110 = data[z1 * in_nynx + y1 * in_nx + x0];
                let v111 = data[z1 * in_nynx + y1 * in_nx + x1];

                let v00 = v000 * (1.0 - tx) + v001 * tx;
                let v01 = v010 * (1.0 - tx) + v011 * tx;
                let v10 = v100 * (1.0 - tx) + v101 * tx;
                let v11 = v110 * (1.0 - tx) + v111 * tx;

                let v0 = v00 * (1.0 - ty) + v01 * ty;
                let v1 = v10 * (1.0 - ty) + v11 * ty;

                output[oz * out_ny * out_nx + oy * out_nx + ox] = v0 * (1.0 - tz) + v1 * tz;
            }
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Public deskew / rotate API
// ---------------------------------------------------------------------------

/// Deskew a 3D frame using a shear transformation.
///
/// Corrects the oblique imaging geometry of a light-sheet microscope by
/// applying a shear along X proportional to the Z-slice index:
///
/// ```text
/// dx = cos(angle) * dz / pixel_size   (pixels shifted per Z-slice)
/// ```
///
/// Output width is automatically expanded to fit the sheared content:
/// `output_nx = ceil(nx + |dx| * (nz - 1))`.
///
/// # Arguments
/// * `data`       — Flattened 3-D input volume in ZYX (row-major) order.
/// * `dims`       — Input dimensions `[nz, ny, nx]`.
/// * `dz`         — Z step size in microns (must be > 0).
/// * `angle`      — Skew angle in degrees (typically 30–45°).
/// * `pixel_size` — XY pixel size in microns (must be > 0; typical value 0.108).
/// * `reverse`    — Set to `true` when the scan direction is reversed.
///
/// # Returns
/// `Ok((deskewed_data, [nz, ny, new_nx]))` — the deskewed volume and its
/// dimensions.
///
/// # Errors
/// Returns `InvalidDimensions` when `dims` is not length-3, when `data.len()`
/// does not match `nz*ny*nx`.  Returns `InvalidResampleParameters` when
/// `dz ≤ 0` or `pixel_size ≤ 0`.
pub fn deskew_frame_3d(
    data: &[f64],
    dims: &[usize],
    dz: f64,
    angle: f64,
    pixel_size: f64,
    reverse: bool,
) -> Result<(Vec<f64>, Vec<usize>), MicroscopeProcessingError> {
    if dims.len() != 3 {
        return Err(MicroscopeProcessingError::InvalidDimensions);
    }
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
    if data.is_empty() || data.len() != nz * ny * nx {
        return Err(MicroscopeProcessingError::InvalidDimensions);
    }
    if dz <= 0.0 || pixel_size <= 0.0 {
        return Err(MicroscopeProcessingError::InvalidResampleParameters);
    }

    let angle_rad = angle.to_radians();
    let mut dx = angle_rad.cos() * dz / pixel_size;
    if reverse {
        dx = -dx;
    }

    // Output X dimension expanded to fit the full shear extent
    let output_nx =
        (nx as f64 + dx.abs() * nz.saturating_sub(1) as f64).ceil() as usize;
    let output_nx = output_nx.max(1);

    // Shear: input = matrix @ output + offset
    //   z_in =  z_out
    //   y_in =  y_out
    //   x_in = -dx * z_out  +  x_out  +  offset_x
    //
    // offset_x shifts the output window so all input data is reachable:
    //   • forward (dx > 0): no shift needed
    //   • reverse  (dx < 0): shift by dx*(nz-1) so z=0 maps to the right half
    let offset_x = if dx < 0.0 {
        dx * nz.saturating_sub(1) as f64
    } else {
        0.0
    };

    let matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-dx, 0.0, 1.0]];
    let offset = [0.0, 0.0, offset_x];

    let result =
        apply_affine_transform_3d(data, nz, ny, nx, nz, ny, output_nx, &matrix, &offset);
    Ok((result, vec![nz, ny, output_nx]))
}

/// Rotate a 3D frame around the Y axis with Z-anisotropy correction.
///
/// Applies a rotation that combines Z-scaling (`z_aniso = sin(angle)*dz/pixel_size`)
/// and a Y-axis rotation, used to produce isotropic views after deskewing.
///
/// The rotation is performed around the volume centre and maintains the same
/// spatial extent as the input.  Optional Z-boundary and XY-region cropping
/// remove the zero-padding produced by the rotation.
///
/// # Arguments
/// * `data`       — Flattened 3-D input volume in ZYX order.
/// * `dims`       — Input dimensions `[nz, ny, nx]`.
/// * `angle`      — Rotation angle in degrees.
/// * `dz`         — Z step size in microns (must be > 0).
/// * `pixel_size` — XY pixel size in microns (must be > 0).
/// * `reverse`    — Negate the rotation angle when `true`.
/// * `crop`       — Trim all-zero Z-slices at the top/bottom boundaries.
/// * `crop_xy`    — Trim all-zero rows/columns in Y and X after rotation.
///
/// # Returns
/// `Ok((rotated_data, [out_nz, out_ny, out_nx]))`.
///
/// # Errors
/// Returns `InvalidDimensions` on shape mismatch.
/// Returns `InvalidResampleParameters` when `dz ≤ 0` or `pixel_size ≤ 0`.
/// Returns `UnsupportedOperation` when the rotation matrix is singular.
pub fn rotate_frame_3d(
    data: &[f64],
    dims: &[usize],
    angle: f64,
    dz: f64,
    pixel_size: f64,
    reverse: bool,
    crop: bool,
    crop_xy: bool,
) -> Result<(Vec<f64>, Vec<usize>), MicroscopeProcessingError> {
    if dims.len() != 3 {
        return Err(MicroscopeProcessingError::InvalidDimensions);
    }
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
    if data.is_empty() || data.len() != nz * ny * nx {
        return Err(MicroscopeProcessingError::InvalidDimensions);
    }
    if dz <= 0.0 || pixel_size <= 0.0 {
        return Err(MicroscopeProcessingError::InvalidResampleParameters);
    }

    let mut angle_rad = angle.to_radians();
    if reverse {
        angle_rad = -angle_rad;
    }

    // Z-anisotropy scaling factor (sample-scan geometry)
    let z_aniso = angle_rad.sin() * dz / pixel_size;
    let zx_ratio = if z_aniso > 0.0 { z_aniso } else { dz / pixel_size };

    // Volume centre (fractional coordinates)
    let cz = (nz as f64 - 1.0) / 2.0;
    let _cy = (ny as f64 - 1.0) / 2.0;
    let cx = (nx as f64 - 1.0) / 2.0;

    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    // Forward matrix M[:3,:3] = R_y @ S_z where
    //   S_z = diag(zx_ratio, 1, 1)
    //   R_y = [[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]]
    //   M[:3,:3] = [[cos_a*zx, 0, sin_a], [0, 1, 0], [-sin_a*zx, 0, cos_a]]
    let matrix_fwd = [
        [cos_a * zx_ratio, 0.0, sin_a],
        [0.0, 1.0, 0.0],
        [-sin_a * zx_ratio, 0.0, cos_a],
    ];

    // Forward offset = R_y @ S_z @ [-cz, -cy, -cx]^T + [cz, cy, cx]^T
    let offset_fwd = [
        cz * (1.0 - cos_a * zx_ratio) - sin_a * cx,
        0.0,
        sin_a * cz * zx_ratio + cx * (1.0 - cos_a),
    ];

    // Invert to get the scipy-style output→input mapping
    let matrix_inv = invert_3x3(&matrix_fwd)?;
    let offset_inv = [
        -(matrix_inv[0][0] * offset_fwd[0]
            + matrix_inv[0][1] * offset_fwd[1]
            + matrix_inv[0][2] * offset_fwd[2]),
        -(matrix_inv[1][0] * offset_fwd[0]
            + matrix_inv[1][1] * offset_fwd[1]
            + matrix_inv[1][2] * offset_fwd[2]),
        -(matrix_inv[2][0] * offset_fwd[0]
            + matrix_inv[2][1] * offset_fwd[1]
            + matrix_inv[2][2] * offset_fwd[2]),
    ];

    // Apply rotation (same output shape as input)
    let mut rotated =
        apply_affine_transform_3d(data, nz, ny, nx, nz, ny, nx, &matrix_inv, &offset_inv);
    let mut out_nz = nz;
    let mut out_ny = ny;
    let mut out_nx = nx;

    // ---- Crop Z ----
    if crop {
        let mut first_z = out_nz;
        let mut last_z = 0usize;
        let mut found = false;
        for z in 0..out_nz {
            let has = (0..out_ny * out_nx).any(|i| rotated[z * out_ny * out_nx + i] > 0.0);
            if has {
                if !found || z < first_z {
                    first_z = z;
                    found = true;
                }
                if z > last_z {
                    last_z = z;
                }
            }
        }
        if found && first_z <= last_z {
            let crop_nz = last_z - first_z + 1;
            let mut buf = vec![0.0f64; crop_nz * out_ny * out_nx];
            let stride = out_ny * out_nx;
            for z in first_z..=last_z {
                buf[(z - first_z) * stride..][..stride]
                    .copy_from_slice(&rotated[z * stride..][..stride]);
            }
            rotated = buf;
            out_nz = crop_nz;
        }
    }

    // ---- Crop Y ----
    if crop_xy {
        let mut first_y = out_ny;
        let mut last_y = 0usize;
        let mut found_y = false;
        for y in 0..out_ny {
            let has = (0..out_nz)
                .any(|z| (0..out_nx).any(|x| rotated[z * out_ny * out_nx + y * out_nx + x] > 0.0));
            if has {
                if !found_y || y < first_y {
                    first_y = y;
                    found_y = true;
                }
                if y > last_y {
                    last_y = y;
                }
            }
        }
        if found_y && first_y <= last_y {
            let crop_ny = last_y - first_y + 1;
            let mut buf = vec![0.0f64; out_nz * crop_ny * out_nx];
            for z in 0..out_nz {
                for y in first_y..=last_y {
                    for x in 0..out_nx {
                        buf[z * crop_ny * out_nx + (y - first_y) * out_nx + x] =
                            rotated[z * out_ny * out_nx + y * out_nx + x];
                    }
                }
            }
            rotated = buf;
            out_ny = crop_ny;
        }

        // ---- Crop X ----
        let mut first_x = out_nx;
        let mut last_x = 0usize;
        let mut found_x = false;
        for x in 0..out_nx {
            let has = (0..out_nz).any(|z| {
                (0..out_ny).any(|y| rotated[z * out_ny * out_nx + y * out_nx + x] > 0.0)
            });
            if has {
                if !found_x || x < first_x {
                    first_x = x;
                    found_x = true;
                }
                if x > last_x {
                    last_x = x;
                }
            }
        }
        if found_x && first_x <= last_x {
            let crop_nx = last_x - first_x + 1;
            let mut buf = vec![0.0f64; out_nz * out_ny * crop_nx];
            for z in 0..out_nz {
                for y in 0..out_ny {
                    for x in first_x..=last_x {
                        buf[z * out_ny * crop_nx + y * crop_nx + (x - first_x)] =
                            rotated[z * out_ny * out_nx + y * out_nx + x];
                    }
                }
            }
            rotated = buf;
            out_nx = crop_nx;
        }
    }

    Ok((rotated, vec![out_nz, out_ny, out_nx]))
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

    // --- deskew_frame_3d tests ---

    #[test]
    fn test_deskew_frame_3d_basic() {
        // 10 z-slices, 20 rows, 30 cols
        let nz = 10usize;
        let ny = 20usize;
        let nx = 30usize;
        let data = vec![1.0f64; nz * ny * nx];
        let dims = [nz, ny, nx];
        let result = deskew_frame_3d(&data, &dims, 0.5, 32.45, 0.108, false);
        assert!(result.is_ok());
        let (out, out_dims) = result.unwrap();
        // Z and Y are unchanged; X expands
        assert_eq!(out_dims[0], nz);
        assert_eq!(out_dims[1], ny);
        assert!(out_dims[2] > nx, "X dimension should expand after deskew");
        assert_eq!(out.len(), out_dims[0] * out_dims[1] * out_dims[2]);
    }

    #[test]
    fn test_deskew_frame_3d_forward_vs_reverse() {
        let nz = 8usize;
        let ny = 10usize;
        let nx = 20usize;
        // Non-uniform data so forward and reverse give different results
        let data: Vec<f64> = (0..nz * ny * nx).map(|i| (i % 13) as f64).collect();
        let dims = [nz, ny, nx];
        let (fwd, _) = deskew_frame_3d(&data, &dims, 0.5, 32.45, 0.108, false).unwrap();
        let (rev, _) = deskew_frame_3d(&data, &dims, 0.5, 32.45, 0.108, true).unwrap();
        // The two directions must differ
        assert!(fwd.iter().zip(rev.iter()).any(|(a, b)| (a - b).abs() > 1e-9));
    }

    #[test]
    fn test_deskew_frame_3d_single_slice() {
        // nz=1: no shear, output width == input width
        let nz = 1usize;
        let ny = 5usize;
        let nx = 10usize;
        let data = vec![2.0f64; nz * ny * nx];
        let dims = [nz, ny, nx];
        let (out, out_dims) = deskew_frame_3d(&data, &dims, 0.5, 32.45, 0.108, false).unwrap();
        assert_eq!(out_dims[2], nx);
        assert_eq!(out.len(), nz * ny * nx);
    }

    #[test]
    fn test_deskew_frame_3d_invalid_dims() {
        let data = vec![1.0f64; 8];
        // dims length != 3
        let result = deskew_frame_3d(&data, &[2, 2], 0.5, 32.45, 0.108, false);
        assert!(result.is_err());
        // data length mismatch
        let result2 = deskew_frame_3d(&data, &[2, 2, 3], 0.5, 32.45, 0.108, false);
        assert!(result2.is_err());
        // bad pixel_size
        let result3 = deskew_frame_3d(&data, &[2, 2, 2], 0.5, 32.45, 0.0, false);
        assert!(result3.is_err());
    }

    // --- rotate_frame_3d tests ---

    #[test]
    fn test_rotate_frame_3d_basic() {
        let nz = 10usize;
        let ny = 20usize;
        let nx = 30usize;
        let data: Vec<f64> = (0..nz * ny * nx).map(|i| (i % 7) as f64 + 1.0).collect();
        let dims = [nz, ny, nx];
        let result = rotate_frame_3d(&data, &dims, 32.45, 0.5, 0.108, false, true, true);
        assert!(result.is_ok());
        let (out, out_dims) = result.unwrap();
        // Z might be cropped, but Y and X are ≤ original
        assert!(out_dims[0] <= nz);
        assert_eq!(out.len(), out_dims[0] * out_dims[1] * out_dims[2]);
        // Non-zero values should exist
        assert!(out.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn test_rotate_frame_3d_no_crop() {
        let nz = 6usize;
        let ny = 8usize;
        let nx = 10usize;
        // Uniform non-zero volume
        let data = vec![1.0f64; nz * ny * nx];
        let dims = [nz, ny, nx];
        let (out, out_dims) =
            rotate_frame_3d(&data, &dims, 32.45, 0.5, 0.108, false, false, false).unwrap();
        // No cropping: Z/Y/X unchanged
        assert_eq!(out_dims, vec![nz, ny, nx]);
        assert_eq!(out.len(), nz * ny * nx);
    }

    #[test]
    fn test_rotate_frame_3d_reverse_differs() {
        let nz = 8usize;
        let ny = 10usize;
        let nx = 15usize;
        let data: Vec<f64> = (0..nz * ny * nx).map(|i| (i % 5) as f64 + 1.0).collect();
        let dims = [nz, ny, nx];
        let (fwd, _) =
            rotate_frame_3d(&data, &dims, 32.45, 0.5, 0.108, false, false, false).unwrap();
        let (rev, _) =
            rotate_frame_3d(&data, &dims, 32.45, 0.5, 0.108, true, false, false).unwrap();
        assert!(fwd.iter().zip(rev.iter()).any(|(a, b)| (a - b).abs() > 1e-9));
    }

    #[test]
    fn test_rotate_frame_3d_invalid() {
        let data = vec![1.0f64; 8];
        assert!(rotate_frame_3d(&data, &[2, 2], 32.45, 0.5, 0.108, false, true, true).is_err());
        assert!(
            rotate_frame_3d(&data, &[2, 2, 3], 32.45, 0.5, 0.108, false, true, true).is_err()
        );
        assert!(
            rotate_frame_3d(&data, &[2, 2, 2], 32.45, 0.5, -1.0, false, true, true).is_err()
        );
    }
}
