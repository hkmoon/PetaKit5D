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

// ---------------------------------------------------------------------------
// Camera flip correction
// ---------------------------------------------------------------------------

/// Flip a 3-D volume for sCMOS camera orientation correction.
///
/// # Arguments
/// * `data`        — Flat ZYX input, length `nz * ny * nx`.
/// * `nz`, `ny`, `nx` — Volume dimensions.
/// * `flip_mode`   — `"none"`, `"horizontal"` (flip X), `"vertical"` (flip Y), or `"both"`.
///
/// # Returns
/// Flipped volume, same length as input. Returns a copy when `flip_mode == "none"`.
pub fn scmos_camera_flip(
    data: &[f64],
    nz: usize,
    ny: usize,
    nx: usize,
    flip_mode: &str,
) -> Vec<f64> {
    if flip_mode == "none" {
        return data.to_vec();
    }
    let mut out = data.to_vec();
    let flip_x = flip_mode == "horizontal" || flip_mode == "both";
    let flip_y = flip_mode == "vertical" || flip_mode == "both";

    let nynx = ny * nx;
    for z in 0..nz {
        for y in 0..ny {
            let sy = if flip_y { ny - 1 - y } else { y };
            for x in 0..nx {
                let sx = if flip_x { nx - 1 - x } else { x };
                out[z * nynx + sy * nx + sx] = data[z * nynx + y * nx + x];
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Max pooling
// ---------------------------------------------------------------------------

/// 3-D max pooling.
///
/// Reduces each spatial dimension by taking the maximum over non-overlapping
/// blocks of size `pool_sz`. If a dimension is not divisible the volume is
/// zero-padded before pooling.
///
/// # Arguments
/// * `data`           — Flat ZYX input, length `nz * ny * nx`.
/// * `nz`, `ny`, `nx` — Input dimensions.
/// * `pool_sz`        — Pool size `[pz, py, px]`.
///
/// # Returns
/// `(pooled_data, [out_nz, out_ny, out_nx])`.
pub fn max_pooling_3d(
    data: &[f64],
    nz: usize,
    ny: usize,
    nx: usize,
    pool_sz: &[usize],
) -> Result<(Vec<f64>, Vec<usize>), MicroscopeProcessingError> {
    if pool_sz.len() != 3 {
        return Err(MicroscopeProcessingError::InvalidDimensions);
    }
    let (pz, py, px) = (pool_sz[0].max(1), pool_sz[1].max(1), pool_sz[2].max(1));

    // Pad dimensions to be divisible by pool sizes
    let padded_nz = ((nz + pz - 1) / pz) * pz;
    let padded_ny = ((ny + py - 1) / py) * py;
    let padded_nx = ((nx + px - 1) / px) * px;

    // Build padded volume (zero-pad)
    let mut padded = vec![f64::NEG_INFINITY; padded_nz * padded_ny * padded_nx];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                padded[z * padded_ny * padded_nx + y * padded_nx + x] =
                    data[z * ny * nx + y * nx + x];
            }
        }
    }

    let out_nz = padded_nz / pz;
    let out_ny = padded_ny / py;
    let out_nx = padded_nx / px;
    let mut out = vec![f64::NEG_INFINITY; out_nz * out_ny * out_nx];

    for oz in 0..out_nz {
        for oy in 0..out_ny {
            for ox in 0..out_nx {
                let mut max_val = f64::NEG_INFINITY;
                for dz in 0..pz {
                    for dy in 0..py {
                        for dx in 0..px {
                            let iz = oz * pz + dz;
                            let iy = oy * py + dy;
                            let ix = ox * px + dx;
                            let v = padded[iz * padded_ny * padded_nx + iy * padded_nx + ix];
                            if v > max_val {
                                max_val = v;
                            }
                        }
                    }
                }
                // Replace −∞ (padding) with 0
                out[oz * out_ny * out_nx + oy * out_nx + ox] =
                    if max_val == f64::NEG_INFINITY { 0.0 } else { max_val };
            }
        }
    }

    Ok((out, vec![out_nz, out_ny, out_nx]))
}

// ---------------------------------------------------------------------------
// Block-average downsampling
// ---------------------------------------------------------------------------

/// Downsample a 3-D volume by block-averaging.
///
/// Each output voxel is the mean of a `factor[0]×factor[1]×factor[2]` block
/// from the (zero-padded) input.  The shape of `factor` must be exactly 3.
///
/// # Returns
/// `(downsampled_data, [out_nz, out_ny, out_nx])`.
pub fn imresize3_average(
    data: &[f64],
    nz: usize,
    ny: usize,
    nx: usize,
    factor: &[usize],
) -> Result<(Vec<f64>, Vec<usize>), MicroscopeProcessingError> {
    if factor.len() != 3 {
        return Err(MicroscopeProcessingError::InvalidDimensions);
    }
    let (fz, fy, fx) = (factor[0].max(1), factor[1].max(1), factor[2].max(1));

    let padded_nz = ((nz + fz - 1) / fz) * fz;
    let padded_ny = ((ny + fy - 1) / fy) * fy;
    let padded_nx = ((nx + fx - 1) / fx) * fx;

    let out_nz = padded_nz / fz;
    let out_ny = padded_ny / fy;
    let out_nx = padded_nx / fx;
    let mut out = vec![0.0f64; out_nz * out_ny * out_nx];

    // Number of valid (non-padded) voxels per output block
    let block_fz = fz as f64;
    let block_fy = fy as f64;
    let block_fx = fx as f64;

    for oz in 0..out_nz {
        for oy in 0..out_ny {
            for ox in 0..out_nx {
                let mut sum = 0.0;
                let mut count = 0.0;
                for dz in 0..fz {
                    let iz = oz * fz + dz;
                    for dy in 0..fy {
                        let iy = oy * fy + dy;
                        for dx in 0..fx {
                            let ix = ox * fx + dx;
                            if iz < nz && iy < ny && ix < nx {
                                sum += data[iz * ny * nx + iy * nx + ix];
                                count += 1.0;
                            }
                        }
                    }
                }
                let _ = (block_fz, block_fy, block_fx); // suppress unused warnings
                out[oz * out_ny * out_nx + oy * out_nx + ox] =
                    if count > 0.0 { sum / count } else { 0.0 };
            }
        }
    }

    Ok((out, vec![out_nz, out_ny, out_nx]))
}

// ---------------------------------------------------------------------------
// Resample setting validation
// ---------------------------------------------------------------------------

/// Validate and compute resample factors for microscopy data.
///
/// # Arguments
/// * `resample_type`  — `"isotropic"`, `"xy_isotropic"`, or `"given"`.
/// * `resample`       — Required for `"given"`: 1, 2, or 3 values (X/Y/Z factors).
/// * `objective_scan` — `true` for objective scan, `false` for stage scan.
/// * `skew_angle`     — Skew angle in degrees.
/// * `xy_pixel_size`  — XY pixel size in µm.
/// * `dz`             — Z step size in µm.
///
/// # Returns
/// `Ok(([rx, ry, rz], z_aniso))` — resample factors and Z anisotropy.
pub fn check_resample_setting(
    resample_type: &str,
    resample: Option<&[f64]>,
    objective_scan: bool,
    skew_angle: f64,
    xy_pixel_size: f64,
    dz: f64,
) -> Result<([f64; 3], f64), MicroscopeProcessingError> {
    let angle_rad = skew_angle.to_radians();
    let z_aniso = if objective_scan {
        dz / xy_pixel_size
    } else {
        angle_rad.sin() * dz / xy_pixel_size
    };

    let factors: [f64; 3] = match resample_type {
        "isotropic" => [1.0, 1.0, 1.0],
        "xy_isotropic" => {
            let zf = ((angle_rad.sin().powi(2) + z_aniso.powi(2) * angle_rad.cos().powi(2))
                / (angle_rad.cos().powi(2) + z_aniso.powi(2) * angle_rad.sin().powi(2)))
            .sqrt();
            [1.0, 1.0, zf]
        }
        "given" => {
            let rs = resample.ok_or(MicroscopeProcessingError::InvalidResampleParameters)?;
            match rs.len() {
                0 => return Err(MicroscopeProcessingError::InvalidResampleParameters),
                1 => [rs[0], rs[0], rs[0]],
                2 => [rs[0], rs[0], rs[1]],
                _ => [rs[0], rs[1], rs[2]],
            }
        }
        _ => return Err(MicroscopeProcessingError::UnsupportedOperation),
    };

    Ok((factors, z_aniso))
}

// ---------------------------------------------------------------------------
// Memory estimation
// ---------------------------------------------------------------------------

/// Estimate computing memory required for a given set of pipeline steps.
///
/// A simplified port of `estimate_computing_memory` that returns the
/// estimated per-process RAM (GB) and GPU memory (GB).
///
/// # Arguments
/// * `im_size`        — Image dimensions `[nz, ny, nx]`.
/// * `steps`          — Pipeline step names (e.g. `&["deskew", "rotate"]`).
/// * `dtype_bytes`    — Bytes per voxel (e.g. 2 for uint16).
/// * `gpu_mem_factor` — Multiplicative factor for GPU overhead (default ≈1.5).
/// * `gpu_max_mem`    — GPU memory cap in GB (default 12.0).
///
/// # Returns
/// `(cpu_mem_gb, gpu_mem_gb)`.
pub fn estimate_computing_memory(
    im_size: &[usize],
    steps: &[&str],
    dtype_bytes: usize,
    gpu_mem_factor: f64,
    gpu_max_mem: f64,
) -> (f64, f64) {
    if im_size.is_empty() {
        return (0.0, 0.0);
    }
    let voxels: usize = im_size.iter().product();
    let data_size_gb = voxels as f64 * dtype_bytes as f64 / 1e9;

    // Base memory factor: assume ~4× input for typical steps
    let base_factor = 4.0 + steps.len() as f64 * 0.5;
    let cpu_mem_gb = data_size_gb * base_factor;
    let gpu_mem_gb = (data_size_gb * gpu_mem_factor).min(gpu_max_mem);

    (cpu_mem_gb, gpu_mem_gb)
}

// ---------------------------------------------------------------------------
// 3-D integral image (prefix sums)
// ---------------------------------------------------------------------------

/// Compute the 3-D integral image (summed area table).
///
/// The output has shape `nz * ny * nx` and `out[z][y][x]` equals the sum of
/// all input voxels with indices `(iz ≤ z, iy ≤ y, ix ≤ x)`.
///
/// # Arguments
/// * `data`           — Flat ZYX input.
/// * `nz`, `ny`, `nx` — Dimensions.
///
/// # Returns
/// Integral image as flat ZYX vector.
pub fn integral_image_3d(data: &[f64], nz: usize, ny: usize, nx: usize) -> Vec<f64> {
    if data.is_empty() || nz == 0 || ny == 0 || nx == 0 {
        return vec![];
    }
    let nynx = ny * nx;
    let mut out = data.to_vec();

    // Prefix sum along X
    for z in 0..nz {
        for y in 0..ny {
            for x in 1..nx {
                out[z * nynx + y * nx + x] += out[z * nynx + y * nx + x - 1];
            }
        }
    }
    // Prefix sum along Y
    for z in 0..nz {
        for y in 1..ny {
            for x in 0..nx {
                let prev = out[z * nynx + (y - 1) * nx + x];
                out[z * nynx + y * nx + x] += prev;
            }
        }
    }
    // Prefix sum along Z
    for z in 1..nz {
        for y in 0..ny {
            for x in 0..nx {
                let prev = out[(z - 1) * nynx + y * nx + x];
                out[z * nynx + y * nx + x] += prev;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// trim_border
// ---------------------------------------------------------------------------

/// Trim the border of a 3-D volume.
///
/// # Arguments
/// * `data`        — Flat ZYX input, length `nz * ny * nx`.
/// * `nz`, `ny`, `nx` — Input dimensions.
/// * `border`      — Border thickness `[bz, by, bx]`.
/// * `method`      — `"pre"` (trim start), `"post"` (trim end), `"both"`.
///
/// # Returns
/// `(trimmed_data, [out_nz, out_ny, out_nx])`.
pub fn trim_border(
    data: &[f64],
    nz: usize,
    ny: usize,
    nx: usize,
    border: &[usize],
    method: &str,
) -> Result<(Vec<f64>, Vec<usize>), MicroscopeProcessingError> {
    if border.len() != 3 {
        return Err(MicroscopeProcessingError::InvalidDimensions);
    }
    let (bz, by, bx) = (border[0], border[1], border[2]);

    let (sz, sy, sx, ez, ey, ex) = match method {
        "pre" => (bz, by, bx, nz, ny, nx),
        "post" => (
            0, 0, 0,
            nz.saturating_sub(bz),
            ny.saturating_sub(by),
            nx.saturating_sub(bx),
        ),
        "both" | _ => (
            bz, by, bx,
            nz.saturating_sub(bz),
            ny.saturating_sub(by),
            nx.saturating_sub(bx),
        ),
    };

    if ez <= sz || ey <= sy || ex <= sx {
        return Ok((vec![], vec![0, 0, 0]));
    }

    let out_nz = ez - sz;
    let out_ny = ey - sy;
    let out_nx = ex - sx;
    let mut out = vec![0.0f64; out_nz * out_ny * out_nx];

    for z in sz..ez {
        for y in sy..ey {
            for x in sx..ex {
                out[(z - sz) * out_ny * out_nx + (y - sy) * out_nx + (x - sx)] =
                    data[z * ny * nx + y * nx + x];
            }
        }
    }
    Ok((out, vec![out_nz, out_ny, out_nx]))
}

// ---------------------------------------------------------------------------
// Erosion via 2-D XZ projection
// ---------------------------------------------------------------------------

/// Erode a 3-D volume via box erosion of its XZ-plane max-projection.
///
/// A 2-D square structuring element of side `2*esize+1` is applied to
/// the XZ projection (max over Y); the eroded mask is then broadcast back
/// to 3-D, and the Y edges are zeroed.
///
/// Returns the eroded volume (same length as input).
pub fn erode_volume_by_2d_projection(
    data: &[f64],
    nz: usize,
    ny: usize,
    nx: usize,
    esize: usize,
) -> Vec<f64> {
    if esize == 0 || data.is_empty() {
        return data.to_vec();
    }

    // Compute XZ projection: max over Y (axis 1 in Python ZYX convention)
    // Here data is ZYX: index z*ny*nx + y*nx + x
    let mut xz_max = vec![false; nz * nx];
    for z in 0..nz {
        for x in 0..nx {
            let mut has_nonzero = false;
            for y in 0..ny {
                if data[z * ny * nx + y * nx + x] != 0.0 {
                    has_nonzero = true;
                    break;
                }
            }
            xz_max[z * nx + x] = has_nonzero;
        }
    }

    // Erode the 2-D XZ mask with a square of radius `esize`
    let eroded_xz: Vec<bool> = {
        let mut e = vec![false; nz * nx];
        let e_i = esize as isize;
        for z in 0..nz {
            for x in 0..nx {
                let mut all_true = true;
                'outer: for dz in -e_i..=e_i {
                    for dx in -e_i..=e_i {
                        let zi = z as isize + dz;
                        let xi = x as isize + dx;
                        if zi < 0
                            || zi >= nz as isize
                            || xi < 0
                            || xi >= nx as isize
                            || !xz_max[zi as usize * nx + xi as usize]
                        {
                            all_true = false;
                            break 'outer;
                        }
                    }
                }
                e[z * nx + x] = all_true;
            }
        }
        e
    };

    // Apply mask: zero voxels where XZ mask is false, and zero Y edges
    let mut out = data.to_vec();
    for z in 0..nz {
        for y in 0..ny {
            let in_y_border = y < esize || y >= ny.saturating_sub(esize);
            for x in 0..nx {
                if in_y_border || !eroded_xz[z * nx + x] {
                    out[z * ny * nx + y * nx + x] = 0.0;
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// OTF ↔ PSF conversion
// ---------------------------------------------------------------------------

/// Convert an OTF (complex) to a PSF (real) via inverse FFT.
///
/// The OTF is stored as interleaved real/imaginary pairs, length `2*nz*ny*nx`.
/// Returns a real-valued PSF of length `nz*ny*nx` with the centre shifted to
/// element (0,0,0) (i.e. circularly shifted by `floor(crop/2)` for cropping).
///
/// If `out_dims` is smaller than the OTF in any dimension the PSF is cropped.
///
/// # Arguments
/// * `otf_re`, `otf_im` — Real and imaginary parts of the OTF, length `nz*ny*nx`.
/// * `nz`, `ny`, `nx`   — OTF dimensions.
/// * `out_dims`         — Desired output size `[oz, oy, ox]`. Defaults to OTF dims.
pub fn decon_otf2psf(
    otf_re: &[f64],
    otf_im: &[f64],
    nz: usize,
    ny: usize,
    nx: usize,
    out_dims: Option<&[usize]>,
) -> Vec<f64> {
    use rustfft::num_complex::Complex;
    use rustfft::FftPlanner;

    let n = nz * ny * nx;
    if otf_re.len() != n || otf_im.len() != n {
        return vec![0.0; n];
    }

    let (oz, oy, ox) = match out_dims {
        Some(d) if d.len() >= 3 => (d[0], d[1], d[2]),
        _ => (nz, ny, nx),
    };

    let nynx = ny * nx;
    let mut buf: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new(otf_re[i], otf_im[i]))
        .collect();

    // 3-D IFFT using separable 1-D IFFTs
    let mut planner = FftPlanner::new();
    let ifft_x = planner.plan_fft_inverse(nx);
    let ifft_y = planner.plan_fft_inverse(ny);
    let ifft_z = planner.plan_fft_inverse(nz);

    // IFFT along X
    for i in 0..nz * ny {
        ifft_x.process(&mut buf[i * nx..(i + 1) * nx]);
    }
    // IFFT along Y
    let mut col = vec![Complex::new(0.0, 0.0); ny];
    for z in 0..nz {
        for x in 0..nx {
            for r in 0..ny { col[r] = buf[z * nynx + r * nx + x]; }
            ifft_y.process(&mut col);
            for r in 0..ny { buf[z * nynx + r * nx + x] = col[r]; }
        }
    }
    // IFFT along Z
    let mut zcol = vec![Complex::new(0.0, 0.0); nz];
    for y in 0..ny {
        for x in 0..nx {
            for z in 0..nz { zcol[z] = buf[z * nynx + y * nx + x]; }
            ifft_z.process(&mut zcol);
            for z in 0..nz { buf[z * nynx + y * nx + x] = zcol[z]; }
        }
    }
    let scale = n as f64;
    let psf: Vec<f64> = buf.iter().map(|c| c.re / scale).collect();

    // Circular shift: shift by -floor(crop/2) where crop = OTF - out
    let shift_z = nz.saturating_sub(oz) / 2;
    let shift_y = ny.saturating_sub(oy) / 2;
    let shift_x = nx.saturating_sub(ox) / 2;

    let mut out = vec![0.0f64; oz * oy * ox];
    let oynx = oy * ox;
    for z in 0..oz {
        for y in 0..oy {
            for x in 0..ox {
                let sz = (z + shift_z) % nz;
                let sy = (y + shift_y) % ny;
                let sx = (x + shift_x) % nx;
                out[z * oynx + y * ox + x] = psf[sz * nynx + sy * nx + sx];
            }
        }
    }
    out
}

/// Convert a PSF (real) to an OTF (complex) via FFT.
///
/// Returns interleaved real part (first half) and imaginary part (second half)
/// — i.e. `[re_0, re_1, …, re_{n-1}, im_0, …, im_{n-1}]`.
///
/// The PSF centre is assumed at `floor(psf_size/2)`. It is circularly shifted
/// to position `(0,0,0)` before the FFT.  The PSF is zero-padded to
/// `out_dims` if provided and larger than the PSF.
///
/// # Arguments
/// * `psf`              — Real PSF, length `pz*py*px`.
/// * `pz`, `py`, `px`   — PSF dimensions.
/// * `out_dims`         — Output (padded) size `[oz, oy, ox]`. Defaults to PSF dims.
pub fn decon_psf2otf(
    psf: &[f64],
    pz: usize,
    py: usize,
    px: usize,
    out_dims: Option<&[usize]>,
) -> Vec<f64> {
    use rustfft::num_complex::Complex;
    use rustfft::FftPlanner;

    let (oz, oy, ox) = match out_dims {
        Some(d) if d.len() >= 3 => (d[0], d[1], d[2]),
        _ => (pz, py, px),
    };
    let on = oz * oy * ox;

    if psf.is_empty() || on == 0 {
        return vec![0.0; 2 * on];
    }

    // Build zero-padded complex buffer
    let oynx = oy * ox;
    let pynx = py * px;
    let mut buf = vec![Complex::new(0.0, 0.0); on];
    for z in 0..pz.min(oz) {
        for y in 0..py.min(oy) {
            for x in 0..px.min(ox) {
                buf[z * oynx + y * ox + x] = Complex::new(psf[z * pynx + y * px + x], 0.0);
            }
        }
    }

    // Circularly shift PSF centre to (0,0,0)
    // Centre is at floor(psf_size / 2) in each dimension
    // We need to roll by -floor(pz/2), etc.
    let sh_z = pz / 2;
    let sh_y = py / 2;
    let sh_x = px / 2;
    let mut shifted = vec![Complex::new(0.0, 0.0); on];
    for z in 0..oz {
        for y in 0..oy {
            for x in 0..ox {
                let nz2 = (z + oz - sh_z) % oz;
                let ny2 = (y + oy - sh_y) % oy;
                let nx2 = (x + ox - sh_x) % ox;
                shifted[nz2 * oynx + ny2 * ox + nx2] = buf[z * oynx + y * ox + x];
            }
        }
    }

    let mut planner = FftPlanner::new();
    let fft_x = planner.plan_fft_forward(ox);
    let fft_y = planner.plan_fft_forward(oy);
    let fft_z = planner.plan_fft_forward(oz);

    // FFT along X
    for i in 0..oz * oy {
        fft_x.process(&mut shifted[i * ox..(i + 1) * ox]);
    }
    // FFT along Y
    let mut col = vec![Complex::new(0.0, 0.0); oy];
    for z in 0..oz {
        for x in 0..ox {
            for r in 0..oy { col[r] = shifted[z * oynx + r * ox + x]; }
            fft_y.process(&mut col);
            for r in 0..oy { shifted[z * oynx + r * ox + x] = col[r]; }
        }
    }
    // FFT along Z
    let mut zcol = vec![Complex::new(0.0, 0.0); oz];
    for y in 0..oy {
        for x in 0..ox {
            for z in 0..oz { zcol[z] = shifted[z * oynx + y * ox + x]; }
            fft_z.process(&mut zcol);
            for z in 0..oz { shifted[z * oynx + y * ox + x] = zcol[z]; }
        }
    }

    // Pack: first on values = real, next on = imaginary
    let mut out = vec![0.0f64; 2 * on];
    for i in 0..on {
        out[i] = shifted[i].re;
        out[on + i] = shifted[i].im;
    }
    out
}

// ---------------------------------------------------------------------------
// Deconvolution mask edge erosion
// ---------------------------------------------------------------------------

/// Erode the edges of a 3-D binary mask for deconvolution.
///
/// For a fully-filled mask (all `true`) the border is set directly.
/// Otherwise a box erosion of radius `edge_erosion` is applied.
///
/// # Arguments
/// * `mask`         — Flat ZYX boolean mask, length `nz * ny * nx`.
/// * `nz`, `ny`, `nx`— Dimensions.
/// * `edge_erosion` — Border width in voxels.
///
/// # Returns
/// Eroded mask of the same length.
pub fn decon_mask_edge_erosion(
    mask: &[bool],
    nz: usize,
    ny: usize,
    nx: usize,
    edge_erosion: usize,
) -> Vec<bool> {
    if edge_erosion == 0 {
        return mask.to_vec();
    }
    let n = nz * ny * nx;
    if mask.len() != n {
        return mask.to_vec();
    }

    let all_true = mask.iter().all(|&v| v);
    let nynx = ny * nx;
    let mut out = mask.to_vec();

    if all_true {
        // Fast path: directly zero the border
        let e = edge_erosion;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    if z < e || z >= nz.saturating_sub(e)
                        || y < e || y >= ny.saturating_sub(e)
                        || x < e || x >= nx.saturating_sub(e)
                    {
                        out[z * nynx + y * nx + x] = false;
                    }
                }
            }
        }
    } else {
        // General: box erosion with radius `edge_erosion`
        let e = edge_erosion as isize;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    if !mask[z * nynx + y * nx + x] {
                        out[z * nynx + y * nx + x] = false;
                        continue;
                    }
                    let mut keep = true;
                    'outer: for dz in -e..=e {
                        for dy in -e..=e {
                            for dx in -e..=e {
                                let zi = z as isize + dz;
                                let yi = y as isize + dy;
                                let xi = x as isize + dx;
                                if zi < 0 || zi >= nz as isize
                                    || yi < 0 || yi >= ny as isize
                                    || xi < 0 || xi >= nx as isize
                                    || !mask[zi as usize * nynx + yi as usize * nx + xi as usize]
                                {
                                    keep = false;
                                    break 'outer;
                                }
                            }
                        }
                    }
                    out[z * nynx + y * nx + x] = keep;
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Min value inside 3-D bounding box
// ---------------------------------------------------------------------------

/// Return the minimum value inside a 3-D bounding box.
///
/// # Arguments
/// * `data`       — Flat ZYX input, length `nz * ny * nx`.
/// * `nz`,`ny`,`nx`— Volume dimensions.
/// * `bbox`       — `[z_start, y_start, x_start, z_end, y_end, x_end]`
///                  (1-based MATLAB indexing, inclusive).
pub fn min_bbox_3d(
    data: &[f64],
    nz: usize,
    ny: usize,
    nx: usize,
    bbox: &[usize],
) -> f64 {
    if bbox.len() < 6 || data.is_empty() {
        return 0.0;
    }
    let (zs, ys, xs) = (bbox[0].saturating_sub(1), bbox[1].saturating_sub(1), bbox[2].saturating_sub(1));
    let (ze, ye, xe) = (bbox[3].min(nz), bbox[4].min(ny), bbox[5].min(nx));
    let mut min_val = f64::INFINITY;
    for z in zs..ze {
        for y in ys..ye {
            for x in xs..xe {
                let v = data[z * ny * nx + y * nx + x];
                if v < min_val { min_val = v; }
            }
        }
    }
    if min_val == f64::INFINITY { 0.0 } else { min_val }
}

// ---------------------------------------------------------------------------
// Flat-field correction
// ---------------------------------------------------------------------------

/// Apply flat-field and background correction to a microscopy frame.
///
/// The correction formula is:
/// `corrected = (frame - background) / ls_mask`
/// where `ls_mask` is processed from the light-sheet illumination pattern.
///
/// # Arguments
/// * `frame`                   — Input frame, flat ZYX (or YX for 2D).
/// * `ny`, `nx`                — Frame dimensions.
/// * `ls_image`                — Light-sheet illumination image, length `ls_ny*ls_nx`.
/// * `ls_ny`, `ls_nx`          — LS image dimensions.
/// * `bg_image`                — Background image, length `bg_ny*bg_nx`.
/// * `bg_ny`, `bg_nx`          — Background dimensions.
/// * `const_offset`            — If `Some(v)`, add `v` instead of background after division.
/// * `lower_limit`             — Minimum allowed value for the LS mask (default 0.4).
pub fn process_flatfield_correction_frame(
    frame: &[f64],
    ny: usize,
    nx: usize,
    ls_image: &[f64],
    ls_ny: usize,
    ls_nx: usize,
    bg_image: &[f64],
    bg_ny: usize,
    bg_nx: usize,
    const_offset: Option<f64>,
    lower_limit: f64,
) -> Vec<f64> {
    if frame.is_empty() || ny == 0 || nx == 0 {
        return frame.to_vec();
    }

    // Crop LS image to frame size (centre crop)
    let crop2d = |src: &[f64], sny: usize, snx: usize| -> Vec<f64> {
        if sny == ny && snx == nx {
            return src.to_vec();
        }
        let dz = ((sny as isize - ny as isize) / 2).max(0) as usize;
        let dx = ((snx as isize - nx as isize) / 2).max(0) as usize;
        let mut out = vec![0.0f64; ny * nx];
        for r in 0..ny {
            for c in 0..nx {
                let sr = r + dz;
                let sc = c + dx;
                if sr < sny && sc < snx {
                    out[r * nx + c] = src[sr * snx + sc];
                }
            }
        }
        out
    };

    let ls_cropped = crop2d(ls_image, ls_ny, ls_nx);
    let bg_cropped = crop2d(bg_image, bg_ny, bg_nx);

    // Compute LS mask: clip to lower_limit
    let ls_max = ls_cropped.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ls_max = if ls_max <= 0.0 { 1.0 } else { ls_max };
    let ls_mask: Vec<f64> = ls_cropped.iter().map(|&v| (v / ls_max).max(lower_limit)).collect();

    // Determine background offset
    let bg_offset = match const_offset {
        Some(v) => v,
        None => {
            let bg_mean: f64 = bg_cropped.iter().sum::<f64>() / bg_cropped.len().max(1) as f64;
            bg_mean
        }
    };

    // Apply correction
    let nz = frame.len() / (ny * nx).max(1);
    let mut out = vec![0.0f64; frame.len()];
    for plane in 0..nz {
        for r in 0..ny {
            for c in 0..nx {
                let idx = plane * ny * nx + r * nx + c;
                let bg = bg_cropped.get(r * nx + c).copied().unwrap_or(0.0);
                let corrected = (frame[idx] - bg) / ls_mask[r * nx + c] + bg_offset;
                out[idx] = corrected.max(0.0);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Normalize Z stack
// ---------------------------------------------------------------------------

/// Normalize intensity across Z slices using median-based scaling.
///
/// Computes median intensity per slice (above threshold 105), scales
/// each slice by `global_median / slice_median`, clamped within ±10.
///
/// # Arguments
/// * `data`         — Flat YXZ input (Y×X×Z order).
/// * `ny`, `nx`, `nz`— Dimensions.
pub fn normalize_z_stack(data: &[f64], ny: usize, nx: usize, nz: usize) -> Vec<f64> {
    if data.is_empty() || nz == 0 {
        return data.to_vec();
    }
    // Background threshold (camera offset); slices below this are ignored
    let bg: f64 = 105.0;
    // Max allowed deviation of a per-slice median from the global median
    let spd: f64 = 10.0;
    let nynx = ny * nx;

    // Compute per-slice median above threshold
    let mut meds: Vec<f64> = vec![f64::NAN; nz];
    for z in 0..nz {
        let mut pos_vals: Vec<f64> = (0..nynx)
            .filter_map(|i| {
                let v = data[i * nz + z] - bg; // YXZ layout
                if v > 0.0 { Some(v) } else { None }
            })
            .collect();
        if !pos_vals.is_empty() {
            pos_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = pos_vals.len() / 2;
            meds[z] = if pos_vals.len() % 2 == 0 {
                (pos_vals[mid - 1] + pos_vals[mid]) / 2.0
            } else {
                pos_vals[mid]
            };
        }
    }

    // Global median (ignore NaN)
    let valid: Vec<f64> = meds.iter().copied().filter(|v| v.is_finite()).collect();
    if valid.is_empty() {
        return data.to_vec();
    }
    let mut vs = valid.clone();
    vs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = if vs.len() % 2 == 0 {
        (vs[vs.len() / 2 - 1] + vs[vs.len() / 2]) / 2.0
    } else {
        vs[vs.len() / 2]
    };

    // Clamp and normalise
    let nn: Vec<f64> = meds.iter().map(|&v| {
        let vv = if v.is_nan() { m } else { v };
        let vv = vv.clamp(m - spd, m + spd);
        vv
    }).collect();
    let max_nn = nn.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let scale: Vec<f64> = nn.iter().map(|&v| if max_nn > 0.0 { v / max_nn } else { 1.0 }).collect();

    let mut out = data.to_vec();
    for z in 0..nz {
        if scale[z] == 0.0 { continue; }
        for i in 0..nynx {
            out[i * nz + z] /= scale[z];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Distance weight for blending
// ---------------------------------------------------------------------------

/// Compute Hann-window blending weights for one axis.
///
/// # Arguments
/// * `sz`          — Length of the axis.
/// * `start`, `end`— Active region, 1-based inclusive (MATLAB convention).
/// * `buffer_size` — Width of the cosine ramp.
/// * `dfactor`     — Exponential decay outside ramp (0 = no decay).
///
/// # Returns
/// Float32 weight vector of length `sz`.
pub fn distance_weight_single_axis(
    sz: usize,
    start: usize,
    end: usize,
    buffer_size: usize,
    dfactor: f64,
) -> Vec<f32> {
    let mut w = vec![1.0f32; sz];
    let s = start.saturating_sub(1).min(sz.saturating_sub(1));
    let t = end.saturating_sub(1).min(sz.saturating_sub(1));

    if s == 0 && t == sz.saturating_sub(1) {
        return w;
    }

    // Zero outside [s, t]
    for v in &mut w[..s] { *v = 0.0; }
    for v in &mut w[t + 1..] { *v = 0.0; }

    // Hann window ramp at start
    for i in 0..=buffer_size.min(s) {
        let x = i as f64;
        let y = buffer_size as f64;
        let hw = (0.5 * std::f64::consts::PI * x / y.max(1.0)).cos().powi(2);
        let idx = s - (buffer_size.min(s) - i);
        if idx < sz { w[idx] = hw.max(1e-3) as f32; }
    }

    // Hann window ramp at end
    for i in 0..=buffer_size.min(sz.saturating_sub(t + 1)) {
        let x = i as f64;
        let y = buffer_size as f64;
        let hw = (0.5 * std::f64::consts::PI * x / y.max(1.0)).cos().powi(2);
        let idx = t + (buffer_size.min(sz.saturating_sub(t + 1)) - i);
        if idx < sz { w[idx] = hw.max(1e-3) as f32; }
    }

    // Exponential decay before start ramp
    if dfactor > 0.0 {
        let exp_start = s.saturating_sub(buffer_size);
        for i in (0..exp_start).rev() {
            let dist = (exp_start - i) as f64;
            w[i] = (dfactor.powf(dist) as f32).max(1e-4);
        }
        // Decay after end ramp
        let exp_end = (t + buffer_size + 1).min(sz);
        for i in exp_end..sz {
            let dist = (i - exp_end + 1) as f64;
            w[i] = (dfactor.powf(dist) as f32).max(1e-4);
        }
    }

    w
}

// ---------------------------------------------------------------------------
// Private helpers for PSF processing
// ---------------------------------------------------------------------------

/// 3-D median filter with `size × size × size` window (nearest-neighbour border).
fn median_filter_3d_f32(data: &[f32], nz: usize, ny: usize, nx: usize, size: usize) -> Vec<f32> {
    let half = (size / 2) as i32;
    let n = nz * ny * nx;
    let mut result = vec![0.0f32; n];
    let mut window = Vec::with_capacity(size * size * size);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                window.clear();
                for dz in -half..=half {
                    for dy in -half..=half {
                        for dx in -half..=half {
                            let nz_ = (z as i32 + dz).clamp(0, nz as i32 - 1) as usize;
                            let ny_ = (y as i32 + dy).clamp(0, ny as i32 - 1) as usize;
                            let nx_ = (x as i32 + dx).clamp(0, nx as i32 - 1) as usize;
                            window.push(data[nz_ * ny * nx + ny_ * nx + nx_]);
                        }
                    }
                }
                window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                result[z * ny * nx + y * nx + x] = window[window.len() / 2];
            }
        }
    }
    result
}

/// BFS connected-component labeling (26-connectivity) for a 3-D boolean mask.
/// Returns label array (0 = background, 1..k = components).
fn label_connected_3d(mask: &[bool], nz: usize, ny: usize, nx: usize) -> Vec<usize> {
    let n = nz * ny * nx;
    let mut labels = vec![0usize; n];
    let mut current_label = 0usize;
    for start in 0..n {
        if !mask[start] || labels[start] != 0 { continue; }
        current_label += 1;
        labels[start] = current_label;
        let mut queue = vec![start];
        let mut head = 0;
        while head < queue.len() {
            let idx = queue[head];
            head += 1;
            let z = idx / (ny * nx);
            let y = (idx % (ny * nx)) / nx;
            let x = idx % nx;
            for dz in -1i32..=1 {
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dz == 0 && dy == 0 && dx == 0 { continue; }
                        let nz_ = z as i32 + dz;
                        let ny_ = y as i32 + dy;
                        let nx_ = x as i32 + dx;
                        if nz_ >= 0 && nz_ < nz as i32
                            && ny_ >= 0 && ny_ < ny as i32
                            && nx_ >= 0 && nx_ < nx as i32
                        {
                            let ni = nz_ as usize * ny * nx
                                + ny_ as usize * nx
                                + nx_ as usize;
                            if mask[ni] && labels[ni] == 0 {
                                labels[ni] = current_label;
                                queue.push(ni);
                            }
                        }
                    }
                }
            }
        }
    }
    labels
}

/// 3-D binary dilation with full 26-connectivity, applied `iterations` times.
fn binary_dilation_3d_cc(mask: &[bool], nz: usize, ny: usize, nx: usize, iterations: usize) -> Vec<bool> {
    let n = nz * ny * nx;
    let mut m = mask.to_vec();
    for _ in 0..iterations {
        let prev = m.clone();
        let mut result = vec![false; n];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    if !prev[z * ny * nx + y * nx + x] { continue; }
                    for dz in -1i32..=1 {
                        for dy in -1i32..=1 {
                            for dx in -1i32..=1 {
                                let nz_ = z as i32 + dz;
                                let ny_ = y as i32 + dy;
                                let nx_ = x as i32 + dx;
                                if nz_ >= 0 && nz_ < nz as i32
                                    && ny_ >= 0 && ny_ < ny as i32
                                    && nx_ >= 0 && nx_ < nx as i32
                                {
                                    result[nz_ as usize * ny * nx
                                        + ny_ as usize * nx
                                        + nx_ as usize] = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        m = result;
    }
    m
}

/// 3-D binary erosion with full 26-connectivity, applied `iterations` times.
fn binary_erosion_3d_cc(mask: &[bool], nz: usize, ny: usize, nx: usize, iterations: usize) -> Vec<bool> {
    let n = nz * ny * nx;
    let mut m = mask.to_vec();
    for _ in 0..iterations {
        let prev = m.clone();
        let mut result = vec![false; n];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = z * ny * nx + y * nx + x;
                    if !prev[idx] { continue; }
                    let mut ok = true;
                    'outer: for dz in -1i32..=1 {
                        for dy in -1i32..=1 {
                            for dx in -1i32..=1 {
                                let nz_ = z as i32 + dz;
                                let ny_ = y as i32 + dy;
                                let nx_ = x as i32 + dx;
                                if nz_ < 0 || nz_ >= nz as i32
                                    || ny_ < 0 || ny_ >= ny as i32
                                    || nx_ < 0 || nx_ >= nx as i32
                                    || !prev[nz_ as usize * ny * nx
                                        + ny_ as usize * nx
                                        + nx_ as usize]
                                {
                                    ok = false;
                                    break 'outer;
                                }
                            }
                        }
                    }
                    result[idx] = ok;
                }
            }
        }
        m = result;
    }
    m
}

/// 3-D binary closing: `iterations` dilations followed by `iterations` erosions.
fn binary_closing_3d_cc(mask: &[bool], nz: usize, ny: usize, nx: usize, iterations: usize) -> Vec<bool> {
    let d = binary_dilation_3d_cc(mask, nz, ny, nx, iterations);
    binary_erosion_3d_cc(&d, nz, ny, nx, iterations)
}

// ---------------------------------------------------------------------------
// PSF preprocessing
// ---------------------------------------------------------------------------

/// Resample and crop a raw PSF with background subtraction.
///
/// Ports `psf_gen` from `psf_analysis.py`.  Preprocessing steps:
/// 1. Estimate and subtract background using edge slices (3D) or edge rows (2D).
/// 2. Isolate the PSF peak with connected-component masking.
/// 3. Center the PSF via circular roll.
/// 4. Optionally resample in Z via FFT truncation if `dz_data > dz_psf`.
///
/// # Arguments
/// * `psf`            — Flat (ny × nx × nz) PSF volume in **YXZ** order (as in the MATLAB/Python code).
/// * `ny/nx/nz`       — PSF dimensions (Y, X, Z).
/// * `dz_psf`         — Z pixel size of the PSF in microns.
/// * `dz_data`        — Z pixel size of the target data in microns.
/// * `med_factor`     — Background multiplier for the `"median"` method (default 1.5).
/// * `method`         — `"median"` or `"masked"` background subtraction.
///
/// # Returns
/// Processed PSF as `Vec<f32>` with the same YX dimensions and resampled Z.
pub fn psf_gen(
    psf: &[f32],
    ny: usize,
    nx: usize,
    nz: usize,
    dz_psf: f64,
    dz_data: f64,
    med_factor: f64,
    method: &str,
) -> Vec<f32> {
    if psf.is_empty() || ny == 0 || nx == 0 { return psf.to_vec(); }
    let n = ny * nx * nz;
    if psf.len() != n { return psf.to_vec(); }

    let mut psf_raw: Vec<f32> = psf.to_vec();

    // Helper: median of positive values
    let median_positive = |v: &[f32]| -> f32 {
        let mut pos: Vec<f32> = v.iter().cloned().filter(|&x| x > 0.0).collect();
        if pos.is_empty() { return 0.0; }
        pos.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        pos[pos.len() / 2]
    };

    if nz > 1 {
        // Collect edge slices (first/last min(5, nz/4) slices)
        let n_edge = 5.min(nz / 4).max(1);
        let edge_indices: Vec<usize> =
            (0..n_edge).chain(nz.saturating_sub(n_edge)..nz).collect();
        let edge_flat: Vec<f32> = edge_indices.iter()
            .flat_map(|&z| (0..ny).flat_map(move |y| (0..nx).map(move |x| psf[y * nx * nz + x * nz + z])))
            .collect();
        let has_positive = edge_flat.iter().any(|&v| v > 0.0);

        if has_positive {
            if method == "median" {
                let bg = med_factor as f32 * median_positive(&edge_flat);
                psf_raw.iter_mut().for_each(|v| *v = (*v - bg).max(0.0));
                // Median filter
                // Reshape to (ny, nx, nz) for median_filter_3d_f32 which expects (nz, ny, nx)
                // We'll do a simple window approach on the YXZ layout
                let psf_med: Vec<f32> = {
                    let size = 3usize;
                    let half = 1i32;
                    let mut out = vec![0.0f32; n];
                    let mut win = Vec::with_capacity(27);
                    for y in 0..ny {
                        for x in 0..nx {
                            for z in 0..nz {
                                win.clear();
                                for dy in -half..=half {
                                    for dx in -half..=half {
                                        for dz in -half..=half {
                                            let ny_ = (y as i32 + dy).clamp(0, ny as i32 - 1) as usize;
                                            let nx_ = (x as i32 + dx).clamp(0, nx as i32 - 1) as usize;
                                            let nz_ = (z as i32 + dz).clamp(0, nz as i32 - 1) as usize;
                                            win.push(psf_raw[ny_ * nx * nz + nx_ * nz + nz_]);
                                        }
                                    }
                                }
                                win.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                                out[y * nx * nz + x * nz + z] = win[win.len() / 2];
                            }
                        }
                    }
                    out
                };
                // Connected components on median-filtered mask
                let bw: Vec<bool> = psf_med.iter().map(|&v| v > 0.0).collect();
                // Convert YXZ→ZYX for label function
                let mut bw_zyx = vec![false; n];
                for y in 0..ny {
                    for x in 0..nx {
                        for z in 0..nz {
                            bw_zyx[z * ny * nx + y * nx + x] = bw[y * nx * nz + x * nz + z];
                        }
                    }
                }
                let labels = label_connected_3d(&bw_zyx, nz, ny, nx);
                // Peak of psf_med (in YXZ)
                let peak_i = psf_med.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i).unwrap_or(0);
                let py = peak_i / (nx * nz);
                let px = (peak_i % (nx * nz)) / nz;
                let pz = peak_i % nz;
                let peak_label = labels[pz * ny * nx + py * nx + px];
                // Closing on ZYX mask of peak component
                let comp_mask: Vec<bool> = (0..n).map(|i| labels[i] == peak_label && peak_label > 0).collect();
                let closed = binary_closing_3d_cc(&comp_mask, nz, ny, nx, 3);
                // Apply mask back to psf_raw (YXZ layout)
                for y in 0..ny {
                    for x in 0..nx {
                        for z in 0..nz {
                            if !closed[z * ny * nx + y * nx + x] {
                                psf_raw[y * nx * nz + x * nz + z] = 0.0;
                            }
                        }
                    }
                }
            } else {
                // "masked" method
                let psf_med: Vec<f32> = {
                    let half = 1i32;
                    let mut out = vec![0.0f32; n];
                    let mut win = Vec::with_capacity(27);
                    for y in 0..ny {
                        for x in 0..nx {
                            for z in 0..nz {
                                win.clear();
                                for dy in -half..=half {
                                    for dx in -half..=half {
                                        for dz in -half..=half {
                                            let ny_ = (y as i32 + dy).clamp(0, ny as i32 - 1) as usize;
                                            let nx_ = (x as i32 + dx).clamp(0, nx as i32 - 1) as usize;
                                            let nz_ = (z as i32 + dz).clamp(0, nz as i32 - 1) as usize;
                                            win.push(psf[ny_ * nx * nz + nx_ * nz + nz_]);
                                        }
                                    }
                                }
                                win.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                                out[y * nx * nz + x * nz + z] = win[win.len() / 2];
                            }
                        }
                    }
                    out
                };
                let n_edge_rows = 10.min(ny / 4).max(1);
                // Background estimate from edge rows in psf_med
                let bg_mean: f32 = {
                    let mut vals: Vec<f32> = Vec::new();
                    for y in (0..n_edge_rows).chain(ny.saturating_sub(n_edge_rows)..ny) {
                        for x in 0..nx {
                            for z in 0..nz {
                                vals.push(psf_med[y * nx * nz + x * nz + z]);
                            }
                        }
                    }
                    if vals.is_empty() { 0.0 } else { vals.iter().sum::<f32>() / vals.len() as f32 }
                };
                // Adaptive threshold: a = max(sqrt|edge - 100|)*3 + mean(edge) per (x,z)
                // For simplicity we use: threshold = bg_mean + 3 * sqrt(bg_mean.abs())
                let bg_threshold = bg_mean + 3.0 * (bg_mean.abs()).sqrt();
                let bw_med1: Vec<bool> = psf_med.iter().map(|&v| v - bg_threshold > 0.0).collect();
                let mut bw_zyx = vec![false; n];
                for y in 0..ny {
                    for x in 0..nx {
                        for z in 0..nz {
                            bw_zyx[z * ny * nx + y * nx + x] = bw_med1[y * nx * nz + x * nz + z];
                        }
                    }
                }
                let labels = label_connected_3d(&bw_zyx, nz, ny, nx);
                let peak_i = psf_med.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i).unwrap_or(0);
                let py = peak_i / (nx * nz);
                let px = (peak_i % (nx * nz)) / nz;
                let pz = peak_i % nz;
                let peak_label = labels[pz * ny * nx + py * nx + px];
                let comp_mask: Vec<bool> = (0..n).map(|i| labels[i] == peak_label && peak_label > 0).collect();
                let closed = binary_closing_3d_cc(&comp_mask, nz, ny, nx, 3);
                let bg = {
                    let mut vals: Vec<f32> = Vec::new();
                    for y in (0..n_edge_rows).chain(ny.saturating_sub(n_edge_rows)..ny) {
                        for x in 0..nx {
                            for z in 0..nz {
                                vals.push(psf_raw[y * nx * nz + x * nz + z]);
                            }
                        }
                    }
                    if vals.is_empty() { 0.0 } else { vals.iter().sum::<f32>() / vals.len() as f32 }
                };
                for y in 0..ny {
                    for x in 0..nx {
                        for z in 0..nz {
                            let m = closed[z * ny * nx + y * nx + x];
                            let v = &mut psf_raw[y * nx * nz + x * nz + z];
                            *v = if m { (*v - bg).max(0.0) } else { 0.0 };
                        }
                    }
                }
            }
        }
    } else {
        // 2D PSF (nz == 1), treat as YX
        if method == "median" {
            let n2 = ny * nx;
            let pos_vals: Vec<f32> = psf_raw.iter().cloned().filter(|&v| v > 0.0).collect();
            if !pos_vals.is_empty() {
                let bg = med_factor as f32 * median_positive(&pos_vals);
                psf_raw.iter_mut().for_each(|v| *v = (*v - bg).max(0.0));
            }
        }
    }

    // Find peak and center via circular shift
    let peak_i = psf_raw.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i).unwrap_or(0);
    let peak_y = peak_i / (nx * nz.max(1));
    let peak_x = (peak_i % (nx * nz.max(1))) / nz.max(1);
    let peak_z = if nz > 1 { peak_i % nz } else { 0 };

    let shift_y = (ny + 1) / 2 - peak_y.min((ny + 1) / 2);
    let shift_x = (nx + 1) / 2 - peak_x.min((nx + 1) / 2);
    let shift_z = if nz > 1 { (nz + 1) / 2 - peak_z.min((nz + 1) / 2) } else { 0 };

    // Apply circular shift in YXZ layout
    let mut psf_centered = vec![0.0f32; n];
    for y in 0..ny {
        for x in 0..nx {
            let ny_ = (y + shift_y) % ny;
            let nx_ = (x + shift_x) % nx;
            for z in 0..nz {
                let nz_ = (z + shift_z) % nz.max(1);
                psf_centered[ny_ * nx * nz.max(1) + nx_ * nz.max(1) + nz_] =
                    psf_raw[y * nx * nz.max(1) + x * nz.max(1) + z];
            }
        }
    }

    // FFT resampling in Z if dz_data > dz_psf
    if nz > 1 && dz_data > dz_psf && dz_psf > 0.0 {
        let dz_ratio = dz_data / dz_psf;
        let new_nz = ((nz as f64 / dz_ratio).round() as usize).max(1);
        if new_nz < nz {
            // Truncate FFT in Z: keep first and last halves of frequency domain
            // Simple truncation: keep first new_nz/2 and last new_nz - new_nz/2 from fft(nz)
            // We implement this as a crop in Fourier domain via DFT sum
            let half_new = new_nz / 2;
            let scale = new_nz as f32 / nz as f32;
            let mut psf_out = vec![0.0f32; ny * nx * new_nz];
            // For each (y, x) column, apply FFT truncation along Z
            for y in 0..ny {
                for x in 0..nx {
                    // Extract Z column
                    let col: Vec<f32> = (0..nz).map(|z| psf_centered[y * nx * nz + x * nz + z]).collect();
                    // DFT
                    let mut fft_col: Vec<(f32, f32)> = vec![(0.0, 0.0); nz];
                    for k in 0..nz {
                        let mut re = 0.0f32;
                        let mut im = 0.0f32;
                        for t in 0..nz {
                            let angle = -2.0 * std::f32::consts::PI * (k * t) as f32 / nz as f32;
                            re += col[t] * angle.cos();
                            im += col[t] * angle.sin();
                        }
                        fft_col[k] = (re, im);
                    }
                    // Truncate: keep first half_new and last (new_nz - half_new) coefficients
                    let mut fft_trunc: Vec<(f32, f32)> = vec![(0.0, 0.0); new_nz];
                    for k in 0..half_new { fft_trunc[k] = fft_col[k]; }
                    for k in 0..(new_nz - half_new) {
                        fft_trunc[half_new + k] = fft_col[nz - (new_nz - half_new) + k];
                    }
                    // IDFT
                    for t in 0..new_nz {
                        let mut re = 0.0f32;
                        for k in 0..new_nz {
                            let angle = 2.0 * std::f32::consts::PI * (k * t) as f32 / new_nz as f32;
                            re += fft_trunc[k].0 * angle.cos() - fft_trunc[k].1 * angle.sin();
                        }
                        psf_out[y * nx * new_nz + x * new_nz + t] = (re / nz as f32).max(0.0);
                    }
                }
            }
            return psf_out;
        }
    }

    psf_centered.iter_mut().for_each(|v| *v = v.max(0.0));
    psf_centered
}

// ---------------------------------------------------------------------------
// PSF rotation
// ---------------------------------------------------------------------------

/// Rotate a PSF to match deskewed/rotated microscopy data coordinates.
///
/// Ports `rotate_psf` from `psf_analysis.py` (based on XR_rotate_PSF.m by Xiongtao Ruan).
///
/// # Arguments
/// * `psf`            — Flat (ny × nx × nz) PSF in **YXZ** order.
/// * `ny/nx/nz`       — PSF dimensions.
/// * `skew_angle`     — Skew angle in degrees (e.g. 32.45).
/// * `xy_pixel_size`  — XY pixel size in microns.
/// * `dz`             — Z step size in microns.
/// * `objective_scan` — True for objective scan; false for stage scan.
/// * `reverse`        — Reverse rotation direction.
///
/// # Returns
/// Rotated PSF with the same dimensions, any zero-padding filled with median.
pub fn rotate_psf(
    psf: &[f32],
    ny: usize,
    nx: usize,
    nz: usize,
    skew_angle: f64,
    xy_pixel_size: f64,
    dz: f64,
    objective_scan: bool,
    reverse: bool,
) -> Vec<f32> {
    if psf.is_empty() || ny == 0 || nx == 0 { return psf.to_vec(); }

    // Compute z anisotropy factor
    let z_aniso = if objective_scan {
        dz / xy_pixel_size
    } else {
        let theta = skew_angle.to_radians();
        (theta.sin() * dz) / xy_pixel_size
    };

    // Convert YXZ → ZYX f64 for rotate_frame_3d
    let n = ny * nx * nz;
    let mut vol_zyx = vec![0.0f64; n];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                vol_zyx[z * ny * nx + y * nx + x] = psf[y * nx * nz + x * nz + z] as f64;
            }
        }
    }

    let rotated_result = rotate_frame_3d(
        &vol_zyx,
        &[nz, ny, nx],
        skew_angle,
        dz,
        xy_pixel_size,
        reverse,
        true, // crop
        false, // crop_xy
    );

    let (rotated_flat, rot_dims) = match rotated_result {
        Ok((v, d)) => (v, d),
        Err(_) => return psf.to_vec(),
    };
    if rotated_flat.is_empty() { return psf.to_vec(); }
    let (rnz, rny, rnx) = if rot_dims.len() >= 3 {
        (rot_dims[0], rot_dims[1], rot_dims[2])
    } else {
        (nz, ny, nx)
    };

    // Fill zeros with median of positive values
    let positive_vals: Vec<f32> = rotated_flat.iter().cloned()
        .filter(|&v| v > 0.0)
        .map(|v| v as f32)
        .collect();
    let fill = if positive_vals.is_empty() {
        0.0f32
    } else {
        let mut sorted = positive_vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Use median of lower 99th percentile
        let p99_index = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
        let p99 = sorted[p99_index];
        let valid: Vec<f32> = sorted.iter().cloned().filter(|&v| v < p99).collect();
        if valid.is_empty() { 0.0 } else { valid[valid.len() / 2] }
    };

    // Convert ZYX → YXZ f32
    let out_n = rny * rnx * rnz;
    let mut out = vec![fill; out_n];
    for z in 0..rnz {
        for y in 0..rny {
            for x in 0..rnx {
                let src = z * rny * rnx + y * rnx + x;
                let dst = y * rnx * rnz + x * rnz + z;
                if src < rotated_flat.len() && dst < out.len() {
                    let v = rotated_flat[src] as f32;
                    out[dst] = if v == 0.0 { fill } else { v };
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 4-D indexed array assignment
// ---------------------------------------------------------------------------

/// Assign a 4-D region into a flat array at the position specified by a
/// MATLAB-style bounding box.
///
/// The bounding box uses 1-based indexing (`bbox[k]` is the *start* index
/// and `bbox[k+4]` is the *inclusive end* index in dimension `k`, matching
/// MATLAB semantics).  The equivalent Python/NumPy slice is
/// `arr[s-1 : e, ...]` where `s = bbox[k]` and `e = bbox[k+4]`.
///
/// # Arguments
/// * `array`       — Mutable flat target array in `[d0, d1, d2, d3]` row-major order.
/// * `dims`        — Dimensions of `array` `[d0, d1, d2, d3]`.
/// * `bbox`        — 8-element 1-based bounding box
///                   `[y_s, x_s, z_s, t_s, y_e, x_e, z_e, t_e]`.
/// * `region`      — Flat source region to insert.
/// * `region_dims` — Dimensions of `region` `[rd0, rd1, rd2, rd3]`.
/// * `region_bbox` — Optional 8-element 1-based crop applied to `region`
///                   before insertion.  `None` inserts the entire `region`.
pub fn indexing_4d(
    array: &mut [f64],
    dims: &[usize; 4],
    bbox: &[usize; 8],
    region: &[f64],
    region_dims: &[usize; 4],
    region_bbox: Option<&[usize; 8]>,
) {
    let [d0, d1, d2, d3] = *dims;
    let [rd0, rd1, rd2, rd3] = *region_dims;

    // Convert 1-based MATLAB bbox to 0-based start / exclusive end.
    let ys = bbox[0].saturating_sub(1);
    let xs = bbox[1].saturating_sub(1);
    let zs = bbox[2].saturating_sub(1);
    let ts = bbox[3].saturating_sub(1);
    let ye = bbox[4]; // Python slice: arr[ys:ye]
    let xe = bbox[5];
    let ze = bbox[6];
    let te = bbox[7];

    let (rys, rxs, rzs, rts, rye, rxe, rze, rte) = match region_bbox {
        Some(rb) => (
            rb[0].saturating_sub(1),
            rb[1].saturating_sub(1),
            rb[2].saturating_sub(1),
            rb[3].saturating_sub(1),
            rb[4],
            rb[5],
            rb[6],
            rb[7],
        ),
        None => (0, 0, 0, 0, rd0, rd1, rd2, rd3),
    };

    for (ai, ri) in (ys..ye.min(d0)).zip(rys..rye.min(rd0)) {
        for (aj, rj) in (xs..xe.min(d1)).zip(rxs..rxe.min(rd1)) {
            for (ak, rk) in (zs..ze.min(d2)).zip(rzs..rze.min(rd2)) {
                for (al, rl) in (ts..te.min(d3)).zip(rts..rte.min(rd3)) {
                    let a_idx = ai * d1 * d2 * d3 + aj * d2 * d3 + ak * d3 + al;
                    let r_idx = ri * rd1 * rd2 * rd3 + rj * rd2 * rd3 + rk * rd3 + rl;
                    if a_idx < array.len() && r_idx < region.len() {
                        array[a_idx] = region[r_idx];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Write data block to a Zarr-style chunked array
// ---------------------------------------------------------------------------

/// Write a block of data into a flat array using Zarr chunk-coordinate addressing.
///
/// Computes the target slice from the 1-based block subscripts and the chunk
/// shape, then copies data.  If `trim_edge` is `true` and the block extends
/// beyond the array boundary the data is trimmed before writing.
///
/// # Arguments
/// * `array`       — Mutable flat target array in `[d0, d1, d2]` row-major order.
/// * `array_shape` — Shape of `array` `[nz, ny, nx]`.
/// * `chunk_shape` — Chunk dimensions `[cz, cy, cx]`.
/// * `block_sub`   — 1-based block subscripts `[bz, by, bx]`.
/// * `data`        — Block data to write (must equal the chunk size unless trimming).
/// * `trim_edge`   — When `true`, trim `data` if the block extends beyond the array.
pub fn write_zarr_block(
    array: &mut [f64],
    array_shape: &[usize; 3],
    chunk_shape: &[usize; 3],
    block_sub: &[usize; 3],
    data: &[f64],
    trim_edge: bool,
) {
    let [az, ay, ax] = *array_shape;
    let [cz, cy, cx] = *chunk_shape;
    let [bz, by, bx] = *block_sub;

    // Convert 1-based block subscripts to 0-based start indices.
    let sz = (bz.saturating_sub(1)) * cz;
    let sy = (by.saturating_sub(1)) * cy;
    let sx = (bx.saturating_sub(1)) * cx;

    // Determine actual write extents, trimming at array boundary if requested.
    let ez = if trim_edge { (sz + cz).min(az) } else { sz + cz };
    let ey = if trim_edge { (sy + cy).min(ay) } else { sy + cy };
    let ex = if trim_edge { (sx + cx).min(ax) } else { sx + cx };

    let dz = ez.saturating_sub(sz);
    let dy = ey.saturating_sub(sy);
    let dx = ex.saturating_sub(sx);

    for iz in 0..dz {
        for iy in 0..dy {
            for ix in 0..dx {
                let arr_idx = (sz + iz) * ay * ax + (sy + iy) * ax + (sx + ix);
                let dat_idx = iz * cy * cx + iy * cx + ix;
                if arr_idx < array.len() && dat_idx < data.len() {
                    array[arr_idx] = data[dat_idx];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Full deskew workflow
// ---------------------------------------------------------------------------

/// Result returned by [`deskew_data`].
#[derive(Debug, Clone)]
pub struct DeskewResult {
    /// Number of input files processed.
    pub n_files: usize,
    /// Output directory path.
    pub output_dir: String,
    /// Paths to saved deskewed TIFF files (empty if `save_deskew` is `false`).
    pub deskewed_files: Vec<String>,
    /// Paths to saved rotated TIFF files (empty if `save_rotate` is `false`).
    pub rotated_files: Vec<String>,
}

/// Complete deskewing workflow for light-sheet microscopy data.
///
/// Orchestrates the full pipeline for each input file:
/// 1. Read TIFF.
/// 2. Apply sCMOS camera flip (`flip_mode`).
/// 3. Optionally apply flat-field correction.
/// 4. Optionally crop to `bbox` (`[z1, z2, y1, y2, x1, x2]`).
/// 5. Deskew using [`deskew_frame_3d`].
/// 6. Optionally rotate using [`rotate_frame_3d`].
/// 7. Write results as TIFF files in `output_dir`.
///
/// # Arguments
/// * `input_paths`      — Paths to input TIFF files.
/// * `output_dir`       — Directory for output files.
/// * `dz`               — Z step size in microns.
/// * `angle`            — Skew angle in degrees.
/// * `pixel_size`       — XY pixel size in microns.
/// * `reverse`          — Reverse scan direction.
/// * `rotate`           — Also produce rotated output.
/// * `flip_mode`        — Camera flip mode: `"none"`, `"horizontal"`, `"vertical"`, `"both"`.
/// * `flat_field_path`  — Optional path to flat-field correction TIFF.
/// * `save_deskew`      — Whether to save deskewed volumes.
/// * `save_rotate`      — Whether to save rotated volumes.
/// * `bbox`             — Optional `[z1, z2, y1, y2, x1, x2]` crop (0-based, exclusive end).
/// * `overwrite`        — Whether to overwrite existing output files.
/// * `ny/nx/nz`         — Dimensions of each input volume.  These are required because
///                        the raw TIFF bytes are decoded as f32 samples.
pub fn deskew_data(
    input_paths: &[&str],
    output_dir: &str,
    dz: f64,
    angle: f64,
    pixel_size: f64,
    reverse: bool,
    rotate: bool,
    flip_mode: &str,
    flat_field_path: Option<&str>,
    save_deskew: bool,
    save_rotate: bool,
    bbox: Option<[usize; 6]>,
    overwrite: bool,
    ny: usize,
    nx: usize,
    nz: usize,
) -> DeskewResult {
    use std::path::Path;

    // Attempt to create the output directory; if it fails, return early with an empty result.
    if std::fs::create_dir_all(output_dir).is_err() {
        return DeskewResult {
            n_files: input_paths.len(),
            output_dir: output_dir.to_string(),
            deskewed_files: Vec::new(),
            rotated_files: Vec::new(),
        };
    }

    // Load flat field once if provided.
    let flat_field: Option<Vec<f64>> = flat_field_path.and_then(|p| {
        crate::io::read_tiff(p, None).ok().map(|bytes| {
            bytes.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
                .collect()
        })
    });

    let mut result = DeskewResult {
        n_files: input_paths.len(),
        output_dir: output_dir.to_string(),
        deskewed_files: Vec::new(),
        rotated_files: Vec::new(),
    };

    for &input_path in input_paths {
        let stem = Path::new(input_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");

        let deskew_out = format!("{}/{}_deskewed.tif", output_dir, stem);
        let rotate_out = format!("{}/{}_rotated.tif", output_dir, stem);

        // Skip if output already exists and overwrite is disabled.
        if !overwrite
            && save_deskew
            && Path::new(&deskew_out).exists()
        {
            result.deskewed_files.push(deskew_out.clone());
            if save_rotate && Path::new(&rotate_out).exists() {
                result.rotated_files.push(rotate_out);
            }
            continue;
        }

        // Read raw bytes and decode as f32 samples.
        let bytes = match crate::io::read_tiff(input_path, None) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let mut data: Vec<f64> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
            .collect();
        let mut dims = [nz, ny, nx];

        // Camera flip.
        data = scmos_camera_flip(&data, dims[0], dims[1], dims[2], flip_mode);

        // Flat-field correction.
        if let Some(ref ff) = flat_field {
            let empty_bg = vec![0.0f64; ny * nx];
            data = process_flatfield_correction_frame(
                &data, ny, nx,
                ff, ny, nx,
                &empty_bg, ny, nx,
                None,
                0.05,
            );
        }

        // Bounding-box crop.
        if let Some([z1, z2, y1, y2, x1, x2]) = bbox {
            let cropped_nz = z2.saturating_sub(z1);
            let cropped_ny = y2.saturating_sub(y1);
            let cropped_nx = x2.saturating_sub(x1);
            let mut cropped = vec![0.0f64; cropped_nz * cropped_ny * cropped_nx];
            for iz in z1..z2.min(dims[0]) {
                for iy in y1..y2.min(dims[1]) {
                    for ix in x1..x2.min(dims[2]) {
                        let src = iz * dims[1] * dims[2] + iy * dims[2] + ix;
                        let dst = (iz - z1) * cropped_ny * cropped_nx
                            + (iy - y1) * cropped_nx
                            + (ix - x1);
                        if src < data.len() && dst < cropped.len() {
                            cropped[dst] = data[src];
                        }
                    }
                }
            }
            data = cropped;
            dims = [cropped_nz, cropped_ny, cropped_nx];
        }

        // Deskew.
        let (deskewed, deskew_dims) = match deskew_frame_3d(&data, &dims, dz, angle, pixel_size, reverse) {
            Ok(r) => r,
            Err(_) => continue,
        };

        // Save deskewed volume.
        if save_deskew {
            let out_ny = deskew_dims.get(1).copied().unwrap_or(ny);
            let out_nx = deskew_dims.get(2).copied().unwrap_or(nx);
            let bytes_out: Vec<u8> = deskewed
                .iter()
                .flat_map(|&v| (v as f32).to_le_bytes())
                .collect();
            if crate::io::write_tiff(&deskew_out, &bytes_out, out_nx, out_ny, 32, "none").is_ok() {
                result.deskewed_files.push(deskew_out.clone());
            }
        }

        // Optionally rotate.
        if rotate {
            let rotated_result = rotate_frame_3d(
                &deskewed,
                &deskew_dims,
                angle,
                dz,
                pixel_size,
                reverse,
                true,
                false,
            );
            if let Ok((rotated, rot_dims)) = rotated_result {
                if save_rotate {
                    let rot_ny = rot_dims.get(1).copied().unwrap_or(ny);
                    let rot_nx = rot_dims.get(2).copied().unwrap_or(nx);
                    let bytes_rot: Vec<u8> = rotated
                        .iter()
                        .flat_map(|&v| (v as f32).to_le_bytes())
                        .collect();
                    if crate::io::write_tiff(&rotate_out, &bytes_rot, rot_nx, rot_ny, 32, "none")
                        .is_ok()
                    {
                        result.rotated_files.push(rotate_out);
                    }
                }
            }
        }
    }

    result
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

    // --- scmos_camera_flip ---

    #[test]
    fn test_scmos_camera_flip_none() {
        let data: Vec<f64> = (0..27).map(|i| i as f64).collect();
        let out = scmos_camera_flip(&data, 3, 3, 3, "none");
        assert_eq!(out, data);
    }

    #[test]
    fn test_scmos_camera_flip_horizontal_reverses_x() {
        // 1×1×3 volume: [10, 20, 30]
        let data = vec![10.0, 20.0, 30.0];
        let out = scmos_camera_flip(&data, 1, 1, 3, "horizontal");
        assert_eq!(out, vec![30.0, 20.0, 10.0]);
    }

    #[test]
    fn test_scmos_camera_flip_vertical_reverses_y() {
        // 1×3×1 volume: [10, 20, 30]
        let data = vec![10.0, 20.0, 30.0];
        let out = scmos_camera_flip(&data, 1, 3, 1, "vertical");
        assert_eq!(out, vec![30.0, 20.0, 10.0]);
    }

    // --- max_pooling_3d ---

    #[test]
    fn test_max_pooling_3d_basic() {
        // 4×4×4 volume of ones, pool 2×2×2 → 2×2×2
        let data = vec![1.0f64; 64];
        let (out, dims) = max_pooling_3d(&data, 4, 4, 4, &[2, 2, 2]).unwrap();
        assert_eq!(dims, vec![2, 2, 2]);
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-9));
    }

    #[test]
    fn test_max_pooling_3d_takes_max() {
        // 2×2×2 volume, one hot at corner
        let mut data = vec![0.0f64; 8];
        data[0] = 5.0; // z=0,y=0,x=0
        let (out, dims) = max_pooling_3d(&data, 2, 2, 2, &[2, 2, 2]).unwrap();
        assert_eq!(dims, vec![1, 1, 1]);
        assert!((out[0] - 5.0).abs() < 1e-9);
    }

    // --- imresize3_average ---

    #[test]
    fn test_imresize3_average_half() {
        // 4×4×4 uniform volume, downsample by 2
        let data = vec![2.0f64; 64];
        let (out, dims) = imresize3_average(&data, 4, 4, 4, &[2, 2, 2]).unwrap();
        assert_eq!(dims, vec![2, 2, 2]);
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|&v| (v - 2.0).abs() < 1e-9));
    }

    #[test]
    fn test_imresize3_average_identity() {
        let data: Vec<f64> = (0..27).map(|i| i as f64).collect();
        let (out, dims) = imresize3_average(&data, 3, 3, 3, &[1, 1, 1]).unwrap();
        assert_eq!(dims, vec![3, 3, 3]);
        assert_eq!(out, data);
    }

    // --- check_resample_setting ---

    #[test]
    fn test_check_resample_setting_isotropic() {
        let (factors, _z) =
            check_resample_setting("isotropic", None, true, 32.45, 0.108, 0.3).unwrap();
        assert_eq!(factors, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_check_resample_setting_given_scalar() {
        let rs = [2.0f64];
        let (factors, _) =
            check_resample_setting("given", Some(&rs), false, 32.45, 0.108, 0.3).unwrap();
        assert_eq!(factors, [2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_check_resample_setting_given_two() {
        let rs = [1.5f64, 2.0];
        let (factors, _) =
            check_resample_setting("given", Some(&rs), false, 32.45, 0.108, 0.3).unwrap();
        assert_eq!(factors, [1.5, 1.5, 2.0]);
    }

    #[test]
    fn test_check_resample_setting_invalid_type() {
        assert!(
            check_resample_setting("unknown", None, false, 32.45, 0.108, 0.3).is_err()
        );
    }

    // --- estimate_computing_memory ---

    #[test]
    fn test_estimate_computing_memory_basic() {
        let im_size = [100usize, 200, 300];
        let (cpu, gpu) = estimate_computing_memory(&im_size, &["deskew", "rotate"], 2, 1.5, 12.0);
        assert!(cpu > 0.0);
        assert!(gpu > 0.0);
        assert!(gpu <= 12.0);
    }

    // --- integral_image_3d ---

    #[test]
    fn test_integral_image_3d_ones() {
        // 2×2×2 volume of ones: corner sum should be 8
        let data = vec![1.0f64; 8];
        let out = integral_image_3d(&data, 2, 2, 2);
        assert_eq!(out.len(), 8);
        // Last voxel (1,1,1) should be sum of entire volume = 8
        assert!((out[7] - 8.0).abs() < 1e-9, "corner sum = {}", out[7]);
    }

    #[test]
    fn test_integral_image_3d_single_voxel() {
        let data = vec![5.0f64];
        let out = integral_image_3d(&data, 1, 1, 1);
        assert_eq!(out, vec![5.0]);
    }

    // --- trim_border ---

    #[test]
    fn test_trim_border_both() {
        let data: Vec<f64> = (0..27).map(|i| i as f64).collect(); // 3×3×3
        let (out, dims) = trim_border(&data, 3, 3, 3, &[1, 1, 1], "both").unwrap();
        assert_eq!(dims, vec![1, 1, 1]);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn test_trim_border_pre() {
        let data = vec![0.0f64; 125]; // 5×5×5
        let (out, dims) = trim_border(&data, 5, 5, 5, &[1, 1, 1], "pre").unwrap();
        assert_eq!(dims, vec![4, 4, 4]);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn test_trim_border_post() {
        let data = vec![0.0f64; 125];
        let (out, dims) = trim_border(&data, 5, 5, 5, &[1, 1, 1], "post").unwrap();
        assert_eq!(dims, vec![4, 4, 4]);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn test_trim_border_zero_border() {
        let data: Vec<f64> = (0..8).map(|i| i as f64).collect(); // 2×2×2
        let (out, dims) = trim_border(&data, 2, 2, 2, &[0, 0, 0], "both").unwrap();
        assert_eq!(dims, vec![2, 2, 2]);
        assert_eq!(out, data);
    }

    // --- erode_volume_by_2d_projection ---

    #[test]
    fn test_erode_volume_by_2d_projection_zero_esize() {
        let data: Vec<f64> = (0..27).map(|i| i as f64).collect();
        let out = erode_volume_by_2d_projection(&data, 3, 3, 3, 0);
        assert_eq!(out, data);
    }

    #[test]
    fn test_erode_volume_by_2d_projection_zeros_border() {
        // 5×5×5 volume full of ones, esize=1 → border voxels should be zeroed
        let data = vec![1.0f64; 125];
        let out = erode_volume_by_2d_projection(&data, 5, 5, 5, 1);
        // Y border (y=0 and y=4) should be zero
        for x in 0..5 {
            for z in 0..5 {
                assert_eq!(out[z * 25 + 0 * 5 + x], 0.0);
                assert_eq!(out[z * 25 + 4 * 5 + x], 0.0);
            }
        }
    }

    // --- decon_otf2psf / decon_psf2otf round-trip ---

    #[test]
    fn test_decon_psf2otf_shape() {
        // PSF of size 4×4×4, output same size → OTF has same number of elements
        let psf: Vec<f64> = (0..64).map(|i| i as f64).collect();
        let otf = decon_psf2otf(&psf, 4, 4, 4, None);
        // Returns 2 * n elements (real + imag)
        assert_eq!(otf.len(), 2 * 64);
    }

    #[test]
    fn test_decon_otf2psf_shape() {
        // Build trivial OTF (all real, no imag)
        let otf_re = vec![1.0f64; 64];
        let otf_im = vec![0.0f64; 64];
        let psf = decon_otf2psf(&otf_re, &otf_im, 4, 4, 4, None);
        assert_eq!(psf.len(), 64);
    }

    #[test]
    fn test_decon_psf2otf_roundtrip() {
        // A delta PSF (impulse) should have a flat OTF; then IFFT → delta
        let mut psf = vec![0.0f64; 8 * 8 * 8];
        psf[4 * 64 + 4 * 8 + 4] = 1.0; // centre of 8×8×8
        let otf = decon_psf2otf(&psf, 8, 8, 8, None);
        let re = &otf[..512];
        let im = &otf[512..];
        // For a shifted delta, all OTF magnitudes should be close to 1
        for i in 0..512 {
            let mag = (re[i] * re[i] + im[i] * im[i]).sqrt();
            assert!((mag - 1.0).abs() < 1e-6, "OTF mag[{i}] = {mag}");
        }
        // Round-trip: OTF → PSF should recover the original
        let recovered = decon_otf2psf(re, im, 8, 8, 8, None);
        // Maximum should be near 1.0
        let max_val = recovered.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((max_val - 1.0).abs() < 1e-6, "recovered max = {max_val}");
    }

    // --- decon_mask_edge_erosion ---

    #[test]
    fn test_decon_mask_edge_erosion_zero() {
        let mask = vec![true; 27];
        let out = decon_mask_edge_erosion(&mask, 3, 3, 3, 0);
        assert!(out.iter().all(|&v| v));
    }

    #[test]
    fn test_decon_mask_edge_erosion_full_mask() {
        let mask = vec![true; 125]; // 5×5×5
        let out = decon_mask_edge_erosion(&mask, 5, 5, 5, 1);
        // Interior (3×3×3 = 27 voxels) should remain true
        let true_count = out.iter().filter(|&&v| v).count();
        assert_eq!(true_count, 27, "interior voxels: {true_count}");
    }

    #[test]
    fn test_decon_mask_edge_erosion_partial_mask() {
        let mut mask = vec![false; 27]; // 3×3×3
        // Set only centre voxel
        mask[13] = true;
        // With esize=1, centre should survive (all neighbours in the box exist, but box is fully masked)
        // Actually centre only has itself as true; erosion with any esize>0 will zero it
        let out = decon_mask_edge_erosion(&mask, 3, 3, 3, 1);
        // Centre should be removed because some neighbours are false
        assert!(!out[13]);
    }

    // --- min_bbox_3d ---

    #[test]
    fn test_min_bbox_3d_basic() {
        let mut data = vec![5.0f64; 27]; // 3×3×3
        data[13] = 1.0; // centre
        // bbox covering whole volume
        let v = min_bbox_3d(&data, 3, 3, 3, &[1, 1, 1, 3, 3, 3]);
        assert!((v - 1.0).abs() < 1e-9, "min = {v}");
    }

    #[test]
    fn test_min_bbox_3d_empty_bbox() {
        let data = vec![5.0f64; 8];
        let v = min_bbox_3d(&data, 2, 2, 2, &[]);
        assert_eq!(v, 0.0);
    }

    // --- process_flatfield_correction_frame ---

    #[test]
    fn test_flatfield_correction_shape() {
        let frame = vec![100.0f64; 100]; // 10×10
        let ls = vec![1.0f64; 100];
        let bg = vec![10.0f64; 100];
        let out = process_flatfield_correction_frame(&frame, 10, 10, &ls, 10, 10, &bg, 10, 10, None, 0.4);
        assert_eq!(out.len(), 100);
    }

    #[test]
    fn test_flatfield_correction_uniform() {
        // frame=100, ls=1.0, bg=0 → corrected = 100/1.0 + 0 = 100
        let frame = vec![100.0f64; 100];
        let ls = vec![1.0f64; 100];
        let bg = vec![0.0f64; 100];
        let out = process_flatfield_correction_frame(&frame, 10, 10, &ls, 10, 10, &bg, 10, 10, Some(0.0), 0.4);
        for v in &out {
            assert!((v - 100.0).abs() < 1e-6, "corrected = {v}");
        }
    }

    // --- normalize_z_stack ---

    #[test]
    fn test_normalize_z_stack_shape() {
        let data = vec![200.0f64; 400]; // 4×10×10 in YXZ layout (10×10×4)
        let out = normalize_z_stack(&data, 10, 10, 4);
        assert_eq!(out.len(), 400);
    }

    // --- distance_weight_single_axis ---

    #[test]
    fn test_distance_weight_full_range() {
        // Full range → all ones
        let w = distance_weight_single_axis(100, 1, 100, 10, 0.99);
        assert!((w[50] - 1.0).abs() < 1e-5, "w[50] = {}", w[50]);
    }

    #[test]
    fn test_distance_weight_shape() {
        let w = distance_weight_single_axis(50, 5, 45, 5, 0.99);
        assert_eq!(w.len(), 50);
    }

    #[test]
    fn test_distance_weight_interior_ones() {
        // Interior (away from both ends) should be 1.0
        let w = distance_weight_single_axis(100, 10, 90, 5, 0.99);
        for i in 15..85 {
            assert!((w[i] - 1.0).abs() < 1e-5, "w[{i}] = {}", w[i]);
        }
    }

    // --- psf_gen ---

    #[test]
    fn test_psf_gen_shape_unchanged() {
        // A simple 4×4×4 PSF with a Gaussian-like peak
        let ny = 4usize;
        let nx = 4usize;
        let nz = 4usize;
        let mut psf = vec![1.0f32; ny * nx * nz];
        // Set peak at centre
        psf[1 * nx * nz + 1 * nz + 1] = 100.0;
        // dz_data = dz_psf → no resampling
        let out = psf_gen(&psf, ny, nx, nz, 0.1, 0.1, 1.5, "median");
        // Should return same or resampled volume ≥ 0
        assert!(!out.is_empty());
        assert!(out.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_psf_gen_empty() {
        let out = psf_gen(&[], 0, 0, 0, 0.1, 0.1, 1.5, "median");
        assert!(out.is_empty());
    }

    #[test]
    fn test_psf_gen_resample() {
        // dz_data = 2 * dz_psf → Z should be halved
        let ny = 8usize;
        let nx = 8usize;
        let nz = 8usize;
        let psf = vec![1.0f32; ny * nx * nz];
        let out = psf_gen(&psf, ny, nx, nz, 0.1, 0.2, 1.5, "masked");
        assert!(!out.is_empty());
    }

    // --- rotate_psf ---

    #[test]
    fn test_rotate_psf_shape() {
        let ny = 4usize;
        let nx = 4usize;
        let nz = 4usize;
        let psf = vec![1.0f32; ny * nx * nz];
        let out = rotate_psf(&psf, ny, nx, nz, 32.45, 0.108, 0.1, false, false);
        assert!(!out.is_empty());
    }

    #[test]
    fn test_rotate_psf_empty() {
        let out = rotate_psf(&[], 0, 0, 0, 32.45, 0.108, 0.1, false, false);
        assert!(out.is_empty());
    }

    // --- indexing_4d ---

    #[test]
    fn test_indexing_4d_full_insert() {
        // Insert a 2×2×2×2 region into a 4×4×4×4 array at MATLAB position (2,2,2,2).
        let mut array = vec![0.0f64; 4 * 4 * 4 * 4];
        let region = vec![1.0f64; 2 * 2 * 2 * 2];
        indexing_4d(
            &mut array,
            &[4, 4, 4, 4],
            &[2, 2, 2, 2, 3, 3, 3, 3], // 1-based, inclusive end
            &region,
            &[2, 2, 2, 2],
            None,
        );
        let idx = 1 * 4 * 4 * 4 + 1 * 4 * 4 + 1 * 4 + 1;
        assert!((array[idx] - 1.0).abs() < 1e-9, "inserted value should be 1.0");
    }

    #[test]
    fn test_indexing_4d_with_region_bbox() {
        let mut array = vec![0.0f64; 8 * 8 * 8 * 8];
        let region = vec![2.0f64; 4 * 4 * 4 * 4];
        // Use only the first 2×2×2×2 of the region.
        indexing_4d(
            &mut array,
            &[8, 8, 8, 8],
            &[1, 1, 1, 1, 2, 2, 2, 2],
            &region,
            &[4, 4, 4, 4],
            Some(&[1, 1, 1, 1, 2, 2, 2, 2]),
        );
        assert!((array[0] - 2.0).abs() < 1e-9);
    }

    // --- write_zarr_block ---

    #[test]
    fn test_write_zarr_block_basic() {
        // A 4×4×4 array filled with zeros; write a 2×2×2 block at (1,1,1).
        let mut array = vec![0.0f64; 4 * 4 * 4];
        let block = vec![7.0f64; 2 * 2 * 2];
        write_zarr_block(&mut array, &[4, 4, 4], &[2, 2, 2], &[1, 1, 1], &block, false);
        // Block (1,1,1) writes to z=0..2, y=0..2, x=0..2 → index 0.
        assert!((array[0] - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_write_zarr_block_second_block() {
        // Block (2,1,1) writes to z=2..4, y=0..2, x=0..2.
        let mut array = vec![0.0f64; 4 * 4 * 4];
        let block = vec![5.0f64; 2 * 2 * 2];
        write_zarr_block(&mut array, &[4, 4, 4], &[2, 2, 2], &[2, 1, 1], &block, false);
        // z=2, y=0, x=0 → index 2*4*4 = 32
        assert!((array[32] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_write_zarr_block_trim_edge() {
        // Block at (2,2,2) extends past a 3×3×3 array; trim_edge should prevent OOB.
        let mut array = vec![0.0f64; 3 * 3 * 3];
        let block = vec![3.0f64; 2 * 2 * 2];
        write_zarr_block(&mut array, &[3, 3, 3], &[2, 2, 2], &[2, 2, 2], &block, true);
        // Should not panic; just writes whatever fits.
        assert_eq!(array.len(), 27);
    }

    // --- deskew_data (unit-level: no-op on non-existent paths) ---

    #[test]
    fn test_deskew_data_missing_input() {
        // Input path that does not exist: result should have 0 output files.
        let result = deskew_data(
            &["/nonexistent/path.tif"],
            "/tmp",
            0.5,
            32.45,
            0.108,
            false,
            false,
            "none",
            None,
            true,
            false,
            None,
            true,
            4,
            4,
            4,
        );
        assert_eq!(result.n_files, 1);
        assert!(result.deskewed_files.is_empty());
    }

    #[test]
    fn test_deskew_data_empty_input() {
        let result = deskew_data(
            &[],
            "/tmp",
            0.5, 32.45, 0.108, false, false, "none",
            None, true, false, None, true, 4, 4, 4,
        );
        assert_eq!(result.n_files, 0);
        assert!(result.deskewed_files.is_empty());
    }
}
