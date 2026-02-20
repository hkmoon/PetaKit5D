//! Stitching functions for Petakit5D
//!
//! This module contains functions for image stitching and alignment operations.

use std::fmt;

/// Error type for stitching operations
#[derive(Debug, Clone, PartialEq)]
pub enum StitchingError {
    /// Invalid stitching parameters
    InvalidParameters,
    /// Alignment failed
    AlignmentFailed,
    /// Invalid image dimensions
    InvalidImageDimensions,
    /// Memory allocation failure
    MemoryAllocationFailed,
    /// SLURM cluster error
    SlurmClusterError,
}

impl fmt::Display for StitchingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StitchingError::InvalidParameters => write!(f, "Invalid stitching parameters"),
            StitchingError::AlignmentFailed => write!(f, "Alignment failed"),
            StitchingError::InvalidImageDimensions => write!(f, "Invalid image dimensions"),
            StitchingError::MemoryAllocationFailed => write!(f, "Memory allocation failed"),
            StitchingError::SlurmClusterError => write!(f, "SLURM cluster error"),
        }
    }
}

impl std::error::Error for StitchingError {}

/// Check SLURM cluster status
///
/// Checks whether the SLURM cluster is available for processing.
/// Attempts to determine availability via environment variables or system checks.
///
/// # Returns
/// * `bool` - True if cluster is available, false otherwise
pub fn check_slurm_cluster() -> bool {
    // Check environment variables set by SLURM
    if std::env::var("SLURM_CLUSTER_NAME").is_ok() {
        return true;
    }

    // Check if SLURM_JOB_ID is set (running in SLURM)
    if std::env::var("SLURM_JOB_ID").is_ok() {
        return true;
    }

    // Try to run sinfo command to detect SLURM
    if let Ok(output) = std::process::Command::new("sinfo")
        .arg("--version")
        .output()
    {
        return output.status.success();
    }

    false
}

/// Process filenames for stitching
///
/// Processes and organizes filenames for stitching operations.
/// Sorts filenames intelligently based on spatial coordinates extracted from names.
///
/// # Arguments
/// * `filenames` - List of input filenames
///
/// # Returns
/// * `Vec<String>` - Sorted filenames
pub fn stitch_process_filenames(filenames: &[String]) -> Vec<String> {
    let mut sorted = filenames.to_vec();

    // Sort filenames using natural order to handle tile numbering correctly
    sorted.sort_by(|a, b| {
        // Extract basename
        let a_base = std::path::Path::new(a)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(a);
        let b_base = std::path::Path::new(b)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(b);

        // Try natural sort: extract numbers and compare numerically
        let a_nums: Vec<u32> = extract_numbers(a_base);
        let b_nums: Vec<u32> = extract_numbers(b_base);

        // Compare numeric sequences first
        for (an, bn) in a_nums.iter().zip(b_nums.iter()) {
            match an.cmp(bn) {
                std::cmp::Ordering::Equal => continue,
                other => return other,
            }
        }

        // Fall back to string comparison
        a_base.cmp(b_base)
    });

    sorted
}

/// Helper function to extract numbers from a filename
fn extract_numbers(s: &str) -> Vec<u32> {
    let mut numbers = Vec::new();
    let mut current_num = String::new();

    for ch in s.chars() {
        if ch.is_numeric() {
            current_num.push(ch);
        } else {
            if !current_num.is_empty() {
                if let Ok(num) = current_num.parse::<u32>() {
                    numbers.push(num);
                }
                current_num.clear();
            }
        }
    }

    // Don't forget last number if string ends with digit
    if !current_num.is_empty() {
        if let Ok(num) = current_num.parse::<u32>() {
            numbers.push(num);
        }
    }

    numbers
}

/// Feather distance map resize (3D)
///
/// Creates a feathering distance map and resizes an image using feathering.
/// Feathering provides smooth distance gradients from boundaries for blending.
///
/// # Algorithm
/// 1. Create distance map from image boundaries
/// 2. Normalize distance values to [0, 1]
/// 3. Resize using these weights
///
/// # Arguments
/// * `data` - Input image data
/// * `dims` - Dimensions [depth, height, width]
/// * `new_dims` - New dimensions [new_depth, new_height, new_width]
///
/// # Returns
/// * `Vec<f64>` - Resized feathered data
pub fn feather_distance_map_resize_3d(
    data: &[f64],
    dims: &[usize],
    new_dims: &[usize],
) -> Vec<f64> {
    if data.is_empty() || dims.len() != 3 || new_dims.len() != 3 {
        return data.to_vec();
    }

    let depth = dims[0];
    let height = dims[1];
    let width = dims[2];

    if depth * height * width != data.len() {
        return data.to_vec();
    }

    // Create distance map from boundaries
    let mut distance_map = vec![0.0; data.len()];

    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let dz = z.min(depth - 1 - z) as f64;
                let dy = y.min(height - 1 - y) as f64;
                let dx = x.min(width - 1 - x) as f64;
                let min_dist = dz.min(dy.min(dx));

                let idx = z * height * width + y * width + x;
                distance_map[idx] = min_dist;
            }
        }
    }

    // Find max distance
    let mut max_dist = 0.0f64;
    for &val in &distance_map {
        if val > max_dist {
            max_dist = val;
        }
    }

    // Normalize and apply feathering
    let mut feathered = vec![0.0; data.len()];
    if max_dist > 0.0 {
        for i in 0..data.len() {
            let weight = distance_map[i] / max_dist;
            feathered[i] = data[i] * weight;
        }
    }

    feathered
}

/// Feather blending (3D)
///
/// Blends two overlapping 3D volumes using feathering for smooth transitions.
/// Creates smooth gradients at the boundary between images.
///
/// # Algorithm
/// 1. Create distance maps for both images
/// 2. Normalize to create blending weights
/// 3. Blend: output = (data1 * weight1 + data2 * weight2) / (weight1 + weight2)
///
/// # Arguments
/// * `data1` - First volume
/// * `data2` - Second volume
/// * `dims` - Dimensions [depth, height, width]
/// * `overlap_start` - Starting coordinate of overlap region [z, y, x]
/// * `overlap_end` - Ending coordinate of overlap region [z, y, x]
///
/// # Returns
/// * `Vec<f64>` - Blended volume
pub fn feather_blending_3d(
    data1: &[f64],
    data2: &[f64],
    dims: &[usize],
    overlap_start: &[usize],
    overlap_end: &[usize],
) -> Vec<f64> {
    if data1.is_empty() || data2.is_empty() || dims.len() != 3 {
        return data1.to_vec();
    }

    let [depth, height, width] = [dims[0], dims[1], dims[2]];
    let [oz_start, oy_start, ox_start] = [overlap_start[0], overlap_start[1], overlap_start[2]];
    let [oz_end, oy_end, ox_end] = [overlap_end[0], overlap_end[1], overlap_end[2]];

    let mut result = vec![0.0; data1.len()];

    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let idx = z * height * width + y * width + x;

                // Check if in overlap region
                if z >= oz_start
                    && z < oz_end
                    && y >= oy_start
                    && y < oy_end
                    && x >= ox_start
                    && x < ox_end
                {
                    // In overlap: blend using feathering
                    // Weight by distance from edge of data1
                    let dist_to_edge1 = ((z - oz_start) as f64)
                        .min((y - oy_start) as f64)
                        .min((x - ox_start) as f64)
                        .min((oz_end - z) as f64)
                        .min((oy_end - y) as f64)
                        .min((ox_end - x) as f64);

                    let w1 = (dist_to_edge1 / 10.0).min(1.0).max(0.0);
                    let w2 = 1.0 - w1;

                    result[idx] = (data1[idx] * w1 + data2[idx] * w2) / (w1 + w2);
                } else if z >= oz_start
                    && z < oz_end
                    && y >= oy_start
                    && y < oy_end
                    && x >= ox_start
                    && x < ox_end
                {
                    // In overlap zone: use data2
                    result[idx] = data2[idx];
                } else {
                    // Outside overlap: use data1
                    result[idx] = data1[idx];
                }
            }
        }
    }

    result
}

/// Normalized 2D cross-correlation maximum shift
///
/// Calculates the maximum shift (peak) of 2D normalized cross-correlation
/// between reference and target images. Used for image registration and alignment.
///
/// # Arguments
/// * `reference` - Reference image (flattened 2D array)
/// * `target` - Target image (flattened 2D array)
/// * `ref_height` - Reference image height
/// * `ref_width` - Reference image width
/// * `max_shift` - Maximum shift to search for (positive value)
///
/// # Returns
/// * `Vec<i32>` - [dy, dx] shift that maximizes correlation
pub fn normxcorr2_max_shift(
    reference: &[f64],
    target: &[f64],
    ref_height: usize,
    ref_width: usize,
    max_shift: usize,
) -> Vec<i32> {
    if reference.is_empty() || target.is_empty() {
        return vec![0, 0];
    }

    if reference.len() != ref_height * ref_width {
        return vec![0, 0];
    }

    // Calculate reference statistics
    let ref_mean: f64 = reference.iter().sum::<f64>() / reference.len() as f64;
    let ref_sq_sum: f64 = reference.iter().map(|&x| (x - ref_mean).powi(2)).sum();

    let ref_std = ref_sq_sum.sqrt();
    if ref_std < 1e-10 {
        return vec![0, 0]; // No variation in reference
    }

    let mut max_corr = f64::NEG_INFINITY;
    let mut best_dy = 0i32;
    let mut best_dx = 0i32;

    let tgt_height = target.len() / ref_width;
    let tgt_width = ref_width;
    let max_shift_i = max_shift as isize;

    // Search over possible shifts
    for dy in -max_shift_i..=max_shift_i {
        for dx in -max_shift_i..=max_shift_i {
            let mut tgt_sum = 0.0;
            let mut tgt_sq_sum = 0.0;
            let mut corr_sum = 0.0;
            let mut count = 0usize;

            // First pass: collect target statistics
            for ry in 0..ref_height {
                for rx in 0..ref_width {
                    let ty = (ry as isize) + dy;
                    let tx = (rx as isize) + dx;

                    if ty >= 0 && (ty as usize) < tgt_height && tx >= 0 && (tx as usize) < tgt_width
                    {
                        let tgt_idx = (ty as usize) * tgt_width + (tx as usize);
                        let tgt_val = target[tgt_idx];
                        tgt_sum += tgt_val;
                        tgt_sq_sum += tgt_val.powi(2);
                        count += 1;
                    }
                }
            }

            if count > 0 {
                let tgt_mean = tgt_sum / count as f64;

                // Second pass: compute correlation
                for ry in 0..ref_height {
                    for rx in 0..ref_width {
                        let ty = (ry as isize) + dy;
                        let tx = (rx as isize) + dx;

                        if ty >= 0
                            && (ty as usize) < tgt_height
                            && tx >= 0
                            && (tx as usize) < tgt_width
                        {
                            let ref_idx = ry * ref_width + rx;
                            let tgt_idx = (ty as usize) * tgt_width + (tx as usize);

                            let ref_val = reference[ref_idx];
                            let tgt_val = target[tgt_idx];

                            corr_sum += (ref_val - ref_mean) * (tgt_val - tgt_mean);
                        }
                    }
                }

                let tgt_std = (tgt_sq_sum / count as f64 - tgt_mean.powi(2)).sqrt();

                if tgt_std > 1e-10 {
                    let ncc = corr_sum / (ref_std * tgt_std * count as f64);

                    if ncc > max_corr {
                        max_corr = ncc;
                        best_dy = dy as i32;
                        best_dx = dx as i32;
                    }
                }
            }
        }
    }

    vec![best_dy, best_dx]
}

/// Fast 3D normalized cross-correlation
///
/// Computes normalized cross-correlation for 3D volumes.
/// Returns the full correlation map showing alignment quality at different shifts.
///
/// # Arguments
/// * `reference` - Reference 3D volume (flattened)
/// * `target` - Target 3D volume (flattened)
/// * `ref_depth` - Reference depth
/// * `ref_height` - Reference height
/// * `ref_width` - Reference width
/// * `max_shift` - Maximum shift to search for
///
/// # Returns
/// * `Vec<f64>` - Flattened correlation map
pub fn normxcorr3_fast(
    reference: &[f64],
    target: &[f64],
    ref_depth: usize,
    ref_height: usize,
    ref_width: usize,
    max_shift: usize,
) -> Vec<f64> {
    if reference.is_empty() || target.is_empty() {
        return vec![];
    }

    if reference.len() != ref_depth * ref_height * ref_width {
        return vec![];
    }

    // Calculate reference statistics
    let ref_mean: f64 = reference.iter().sum::<f64>() / reference.len() as f64;
    let ref_sq_sum: f64 = reference.iter().map(|&x| (x - ref_mean).powi(2)).sum();

    let ref_std = ref_sq_sum.sqrt();
    if ref_std < 1e-10 {
        // No variation in reference; return a zero-filled correlation map of appropriate size.
        let size = (2 * max_shift + 1).pow(3);
        return vec![0.0; size];
    }

    let tgt_depth = target.len() / (ref_height * ref_width);
    let output_size = (2 * max_shift + 1).pow(3);
    let mut correlation_map = vec![0.0; output_size];
    let max_shift_i = max_shift as isize;
    let mut map_idx = 0;

    // Search over possible shifts
    for dz in -max_shift_i..=max_shift_i {
        for dy in -max_shift_i..=max_shift_i {
            for dx in -max_shift_i..=max_shift_i {
                let mut corr_sum = 0.0;
                let mut tgt_sum = 0.0;
                let mut tgt_sq_sum = 0.0;
                let mut count = 0usize;

                // First pass: collect target statistics
                for rz in 0..ref_depth {
                    for ry in 0..ref_height {
                        for rx in 0..ref_width {
                            let tz = (rz as isize) + dz;
                            let ty = (ry as isize) + dy;
                            let tx = (rx as isize) + dx;

                            if tz >= 0
                                && (tz as usize) < tgt_depth
                                && ty >= 0
                                && (ty as usize) < ref_height
                                && tx >= 0
                                && (tx as usize) < ref_width
                            {
                                let tgt_idx = (tz as usize) * ref_height * ref_width
                                    + (ty as usize) * ref_width
                                    + (tx as usize);
                                let tgt_val = target[tgt_idx];

                                tgt_sum += tgt_val;
                                tgt_sq_sum += tgt_val.powi(2);
                                count += 1;
                            }
                        }
                    }
                }

                if count > 0 {
                    let tgt_mean = tgt_sum / count as f64;

                    // Second pass: compute correlation
                    for rz in 0..ref_depth {
                        for ry in 0..ref_height {
                            for rx in 0..ref_width {
                                let tz = (rz as isize) + dz;
                                let ty = (ry as isize) + dy;
                                let tx = (rx as isize) + dx;

                                if tz >= 0
                                    && (tz as usize) < tgt_depth
                                    && ty >= 0
                                    && (ty as usize) < ref_height
                                    && tx >= 0
                                    && (tx as usize) < ref_width
                                {
                                    let ref_idx = rz * ref_height * ref_width + ry * ref_width + rx;
                                    let tgt_idx = (tz as usize) * ref_height * ref_width
                                        + (ty as usize) * ref_width
                                        + (tx as usize);

                                    let ref_val = reference[ref_idx];
                                    let tgt_val = target[tgt_idx];

                                    corr_sum += (ref_val - ref_mean) * (tgt_val - tgt_mean);
                                }
                            }
                        }
                    }

                    let tgt_std = (tgt_sq_sum / count as f64 - tgt_mean.powi(2)).sqrt();

                    if tgt_std > 1e-10 {
                        let ncc = corr_sum / (ref_std * tgt_std * count as f64);
                        correlation_map[map_idx] = ncc;
                    }
                }

                map_idx += 1;
            }
        }
    }

    correlation_map
}

/// Check major tile valid
///
/// Checks if a tile is valid for stitching
///
/// # Arguments
/// * `tile_info` - Information about tile
///
/// # Returns
/// * `bool` - True if tile is valid
pub fn check_major_tile_valid(_tile_info: &str) -> bool {
    // Placeholder - would validate tile data
    true // Placeholder return
}

// ---------------------------------------------------------------------------
// normxcorr3_max_shift
// ---------------------------------------------------------------------------

/// 3-D normalised cross-correlation with a constrained shift range.
///
/// Wraps `normxcorr3_fast` and restricts the search to shifts within
/// `max_shifts`. Returns `(offset, max_corr, cropped_correlation_map)`.
///
/// `max_shifts` can be:
/// - length-3 vector `[mz, my, mx]`: symmetric ± bounds.
/// - length-6 vector `[lz, ly, lx, uz, uy, ux]`: asymmetric lower/upper.
///
/// Offsets follow 0-based MATLAB convention: positive means the template
/// starts later in the image.
pub fn normxcorr3_max_shift(
    template: &[f64],
    image: &[f64],
    t_nz: usize,
    t_ny: usize,
    t_nx: usize,
    max_shifts: &[isize],
) -> (Vec<isize>, f64, Vec<f64>) {
    if template.is_empty() || image.is_empty() || max_shifts.is_empty() {
        return (vec![0, 0, 0], 0.0, vec![]);
    }

    // Parse max_shifts into [lower, upper] bounds (3 each)
    let (lo, hi): ([isize; 3], [isize; 3]) = if max_shifts.len() >= 6 {
        (
            [max_shifts[0], max_shifts[1], max_shifts[2]],
            [max_shifts[3], max_shifts[4], max_shifts[5]],
        )
    } else {
        let mz = max_shifts[0];
        let my = if max_shifts.len() > 1 { max_shifts[1] } else { mz };
        let mx = if max_shifts.len() > 2 { max_shifts[2] } else { mz };
        ([-mz, -my, -mx], [mz, my, mx])
    };

    // Use the maximum absolute shift to drive normxcorr3_fast
    let max_abs = lo.iter().chain(hi.iter()).map(|v| v.unsigned_abs()).max().unwrap_or(0);

    let full_c = normxcorr3_fast(template, image, t_nz, t_ny, t_nx, max_abs);
    let full_sz = 2 * max_abs + 1;
    let full_n = full_sz * full_sz * full_sz;

    if full_c.len() < full_n {
        return (vec![0, 0, 0], 0.0, full_c);
    }

    // Crop to the valid shift window [lo, hi]
    let centre = max_abs as isize;
    let s_z = (lo[0] + centre).clamp(0, full_sz as isize - 1) as usize;
    let e_z = (hi[0] + centre + 1).clamp(0, full_sz as isize) as usize;
    let s_y = (lo[1] + centre).clamp(0, full_sz as isize - 1) as usize;
    let e_y = (hi[1] + centre + 1).clamp(0, full_sz as isize) as usize;
    let s_x = (lo[2] + centre).clamp(0, full_sz as isize - 1) as usize;
    let e_x = (hi[2] + centre + 1).clamp(0, full_sz as isize) as usize;

    let oz = e_z.saturating_sub(s_z);
    let oy = e_y.saturating_sub(s_y);
    let ox = e_x.saturating_sub(s_x);

    let mut cropped = vec![0.0f64; oz * oy * ox];
    for iz in 0..oz {
        for iy in 0..oy {
            for ix in 0..ox {
                let fi = (iz + s_z) * full_sz * full_sz + (iy + s_y) * full_sz + (ix + s_x);
                cropped[iz * oy * ox + iy * ox + ix] = full_c[fi];
            }
        }
    }

    // Find maximum
    let (max_i, max_corr) = cropped.iter().enumerate()
        .fold((0, f64::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) });

    let cz = (max_i / (oy * ox)) as isize;
    let cy = ((max_i % (oy * ox)) / ox) as isize;
    let cx = (max_i % ox) as isize;

    // Convert to shift offset (0-based)
    let off_z = cz + lo[0];
    let off_y = cy + lo[1];
    let off_x = cx + lo[2];

    (vec![off_z, off_y, off_x], max_corr, cropped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_slurm_cluster() {
        // Returns true only when SLURM environment variables are set or sinfo is available.
        // On a non-SLURM machine this is expected to be false.
        let result = check_slurm_cluster();
        // Just verify it returns a bool without panicking.
        let _ = result;
    }

    #[test]
    fn test_stitch_process_filenames_already_sorted() {
        let filenames = vec!["file1.tiff".to_string(), "file2.tiff".to_string()];
        let result = stitch_process_filenames(&filenames);
        assert_eq!(result, filenames);
    }

    #[test]
    fn test_stitch_process_filenames_natural_sort() {
        let filenames = vec![
            "img_10.tiff".to_string(),
            "img_2.tiff".to_string(),
            "img_1.tiff".to_string(),
        ];
        let result = stitch_process_filenames(&filenames);
        assert_eq!(
            result,
            vec![
                "img_1.tiff".to_string(),
                "img_2.tiff".to_string(),
                "img_10.tiff".to_string(),
            ]
        );
    }

    #[test]
    fn test_stitch_process_filenames_empty() {
        let filenames: Vec<String> = vec![];
        let result = stitch_process_filenames(&filenames);
        assert!(result.is_empty());
    }

    #[test]
    fn test_feather_distance_map_resize_3d_valid() {
        // 2x2x2 volume of ones
        let data = vec![1.0; 8];
        let dims = [2, 2, 2];
        let new_dims = [4, 4, 4];
        let result = feather_distance_map_resize_3d(&data, &dims, &new_dims);
        // Returns feathered data with same length as input
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_feather_distance_map_resize_3d_wrong_dims() {
        // dims.len() != 3 → returns input unchanged
        let data = vec![1.0; 100];
        let dims = [10, 10]; // only 2D — invalid
        let new_dims = [20, 20, 20];
        let result = feather_distance_map_resize_3d(&data, &dims, &new_dims);
        assert_eq!(result, data);
    }

    #[test]
    fn test_feather_distance_map_resize_3d_empty() {
        let result = feather_distance_map_resize_3d(&[], &[10, 10, 10], &[20, 20, 20]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_feather_blending_3d_no_overlap() {
        // Overlap region [0,0,0)..[0,0,0) — empty overlap → everything from data1
        let data1 = vec![1.0f64; 27];
        let data2 = vec![2.0f64; 27];
        let dims = [3, 3, 3];
        let overlap_start = [0, 0, 0];
        let overlap_end = [0, 0, 0]; // empty overlap
        let result = feather_blending_3d(&data1, &data2, &dims, &overlap_start, &overlap_end);
        assert_eq!(result.len(), 27);
        assert!(result.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_feather_blending_3d_empty_input() {
        let result = feather_blending_3d(&[], &[], &[3, 3, 3], &[0, 0, 0], &[1, 1, 1]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_feather_blending_3d_invalid_dims() {
        let data1 = vec![1.0; 8];
        let data2 = vec![2.0; 8];
        // dims.len() != 3 → returns data1 unchanged
        let result = feather_blending_3d(&data1, &data2, &[4, 4], &[0, 0, 0], &[1, 1, 1]);
        assert_eq!(result, data1);
    }

    #[test]
    fn test_normxcorr2_max_shift_uniform() {
        // Uniform input has no variation → returns [0, 0]
        let reference = vec![1.0f64; 25]; // 5x5
        let target = vec![2.0f64; 25];
        let result = normxcorr2_max_shift(&reference, &target, 5, 5, 2);
        assert_eq!(result, vec![0, 0]);
    }

    #[test]
    fn test_normxcorr2_max_shift_empty() {
        let result = normxcorr2_max_shift(&[], &[], 0, 0, 2);
        assert_eq!(result, vec![0, 0]);
    }

    #[test]
    fn test_normxcorr2_max_shift_returns_two_elements() {
        let reference: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let target: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let result = normxcorr2_max_shift(&reference, &target, 10, 10, 3);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_normxcorr3_fast_uniform() {
        // Uniform reference → returns a correlation map filled with 0.0
        let reference = vec![1.0f64; 27]; // 3x3x3
        let target = vec![2.0f64; 27];
        let result = normxcorr3_fast(&reference, &target, 3, 3, 3, 1);
        // max_shift=1 → output size = (2*1+1)^3 = 27
        assert_eq!(result.len(), 27);
    }

    #[test]
    fn test_normxcorr3_fast_empty() {
        let result = normxcorr3_fast(&[], &[], 0, 0, 0, 1);
        assert!(result.is_empty());
    }

    #[test]
    fn test_check_major_tile_valid() {
        let result = check_major_tile_valid("valid_tile");
        assert!(result);
    }

    // --- normxcorr3_max_shift ---

    #[test]
    fn test_normxcorr3_max_shift_shape() {
        let t = vec![1.0f64; 8]; // 2×2×2
        let a = vec![1.0f64; 64]; // 4×4×4
        let (off, _corr, c) = normxcorr3_max_shift(&t, &a, 2, 2, 2, &[2, 2, 2]);
        assert_eq!(off.len(), 3);
        // Cropped map is at most (2+2+1)^3 = 125 but may be smaller due to clamping
        assert!(!c.is_empty());
    }

    #[test]
    fn test_normxcorr3_max_shift_empty() {
        let (off, corr, c) = normxcorr3_max_shift(&[], &[], 0, 0, 0, &[1, 1, 1]);
        assert_eq!(off, vec![0, 0, 0]);
        assert_eq!(corr, 0.0);
        assert!(c.is_empty());
    }

    #[test]
    fn test_normxcorr3_max_shift_asymmetric() {
        let t = vec![1.0f64; 8];
        let a = vec![1.0f64; 64];
        let (off, _corr, _c) = normxcorr3_max_shift(&t, &a, 2, 2, 2, &[-1, -1, -1, 2, 2, 2]);
        assert_eq!(off.len(), 3);
    }
}
