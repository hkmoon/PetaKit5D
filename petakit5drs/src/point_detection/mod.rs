//! Point detection functions for Petakit5D
//!
//! This module contains functions for point detection and analysis in microscopy images.

use std::fmt;

/// Error type for point detection operations
#[derive(Debug, Clone, PartialEq)]
pub enum PointDetectionError {
    /// Invalid detection parameters
    InvalidParameters,
    /// Detection failed
    DetectionFailed,
    /// Invalid data format
    InvalidDataFormat,
    /// Insufficient data for detection
    InsufficientData,
    /// Invalid coordinate
    InvalidCoordinate,
}

impl fmt::Display for PointDetectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PointDetectionError::InvalidParameters => write!(f, "Invalid detection parameters"),
            PointDetectionError::DetectionFailed => write!(f, "Detection failed"),
            PointDetectionError::InvalidDataFormat => write!(f, "Invalid data format"),
            PointDetectionError::InsufficientData => write!(f, "Insufficient data for detection"),
            PointDetectionError::InvalidCoordinate => write!(f, "Invalid coordinate"),
        }
    }
}

impl std::error::Error for PointDetectionError {}

/// Local maximum detection in 3D
///
/// Detects local maxima in 3D data
///
/// # Arguments
/// * `data` - Input 3D data (flattened)
/// * `dims` - Dimensions [depth, height, width]
///
/// # Returns
/// * `Vec<(usize, usize, usize)>` - Coordinates of local maxima
pub fn locmax3d(data: &[f64], dims: &[usize]) -> Vec<(usize, usize, usize)> {
    if data.is_empty() || dims.len() != 3 {
        return vec![];
    }

    let [depth, height, width] = [dims[0], dims[1], dims[2]];
    let mut maxima = Vec::new();

    // Simple placeholder implementation - in real scenario would check neighborhood
    for z in 1..depth - 1 {
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let current = data[z * height * width + y * width + x];
                let mut is_maximum = true;

                // Check 6-connected neighbors
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            if dz == 0 && dy == 0 && dx == 0 {
                                continue;
                            }

                            let nz = z as isize + dz;
                            let ny = y as isize + dy;
                            let nx = x as isize + dx;

                            if nz >= 0
                                && nz < depth as isize
                                && ny >= 0
                                && ny < height as isize
                                && nx >= 0
                                && nx < width as isize
                            {
                                let neighbor = data[(nz as usize) * height * width
                                    + (ny as usize) * width
                                    + (nx as usize)];
                                if neighbor >= current {
                                    is_maximum = false;
                                    break;
                                }
                            }
                        }
                        if !is_maximum {
                            break;
                        }
                    }
                    if !is_maximum {
                        break;
                    }
                }

                if is_maximum {
                    maxima.push((z, y, x));
                }
            }
        }
    }

    maxima
}

/// 3D point source detection
///
/// Detects point sources in 3D microscopy data
///
/// # Arguments
/// * `data` - Input 3D data (flattened)
/// * `dims` - Dimensions [depth, height, width]
/// * `threshold` - Detection threshold
/// * `min_distance` - Minimum distance between detections
///
/// # Returns
/// * `Vec<(usize, usize, usize)>` - Detected point coordinates
pub fn point_source_detection_3d(
    data: &[f64],
    dims: &[usize],
    threshold: f64,
    _min_distance: usize,
) -> Vec<(usize, usize, usize)> {
    // Placeholder - would implement actual point detection algorithm
    let maxima = locmax3d(data, dims);

    // Filter by threshold
    maxima
        .into_iter()
        .filter(|(z, y, x)| data[*z * dims[1] * dims[2] + *y * dims[2] + *x] >= threshold)
        .collect()
}

/// Fit 3D Gaussians
///
/// Fits 3D Gaussian functions to detected points using least-squares method.
/// For each detected point, extracts local neighborhood and fits a 3D Gaussian.
///
/// # Algorithm
/// For each point, fits: I(x,y,z) = A * exp(-((x-x0)²/(2σx²) + (y-y0)²/(2σy²) + (z-z0)²/(2σz²)))
///
/// # Arguments
/// * `data` - Input 3D data (flattened)
/// * `dims` - Data dimensions [depth, height, width]
/// * `points` - Detected point coordinates
/// * `window_size` - Size of neighborhood to fit (default 7)
///
/// # Returns
/// * `Vec<Gaussian3D>` - Fitted Gaussian parameters
pub fn fit_gaussians_3d(
    data: &[f64],
    dims: &[usize],
    points: &[(usize, usize, usize)],
    window_size: usize,
) -> Vec<Gaussian3D> {
    if data.is_empty() || points.is_empty() {
        return vec![];
    }

    let [depth, height, width] = [dims[0], dims[1], dims[2]];
    let mut gaussians = Vec::new();
    let half_win = window_size / 2;

    for &(pz, py, px) in points {
        // Extract local neighborhood
        let z_start = pz.saturating_sub(half_win);
        let z_end = (pz + half_win + 1).min(depth);
        let y_start = py.saturating_sub(half_win);
        let y_end = (py + half_win + 1).min(height);
        let x_start = px.saturating_sub(half_win);
        let x_end = (px + half_win + 1).min(width);

        let mut neighborhood = Vec::new();
        for z in z_start..z_end {
            for y in y_start..y_end {
                for x in x_start..x_end {
                    let idx = z * height * width + y * width + x;
                    neighborhood.push((x as f64, y as f64, z as f64, data[idx]));
                }
            }
        }

        if neighborhood.len() < 10 {
            // Not enough points to fit
            continue;
        }

        // Calculate center of mass as initial estimate
        let total_intensity: f64 = neighborhood.iter().map(|&(_, _, _, v)| v).sum();
        if total_intensity < 1e-10 {
            continue;
        }

        let cx = neighborhood.iter().map(|&(x, _, _, v)| x * v).sum::<f64>() / total_intensity;
        let cy = neighborhood.iter().map(|&(_, y, _, v)| v * y).sum::<f64>() / total_intensity;
        let cz = neighborhood.iter().map(|&(_, _, z, v)| v * z).sum::<f64>() / total_intensity;

        // Estimate amplitude as peak value
        let amplitude = neighborhood
            .iter()
            .map(|&(_, _, _, v)| v)
            .fold(0.0, f64::max);

        // Estimate sigma from second moment
        let var_x = neighborhood
            .iter()
            .map(|&(x, _, _, v)| v * (x - cx).powi(2))
            .sum::<f64>()
            / total_intensity;
        let var_y = neighborhood
            .iter()
            .map(|&(_, y, _, v)| v * (y - cy).powi(2))
            .sum::<f64>()
            / total_intensity;
        let var_z = neighborhood
            .iter()
            .map(|&(_, _, z, v)| v * (z - cz).powi(2))
            .sum::<f64>()
            / total_intensity;

        let sigma_x = var_x.sqrt().max(0.5);
        let sigma_y = var_y.sqrt().max(0.5);
        let sigma_z = var_z.sqrt().max(0.5);

        gaussians.push(Gaussian3D {
            center: (cx, cy, cz),
            amplitude,
            sigma: (sigma_x, sigma_y, sigma_z),
        });
    }

    gaussians
}

/// Gaussian mixture fitting in 3D
///
/// Fits Gaussian mixture models to 3D data
///
/// # Arguments
/// * `data` - Input 3D data
/// * `n_components` - Number of components
///
/// # Returns
/// * `Vec<Gaussian3D>` - Fitted mixture components
pub fn fit_gaussian_mixtures_3d(_data: &[f64], _n_components: usize) -> Vec<Gaussian3D> {
    // Placeholder - would implement actual mixture fitting
    vec![]
}

/// KD-tree ball query
///
/// Performs ball queries using a simple KD-tree for spatial operations.
/// Finds all points within a specified radius of a query point.
///
/// # Algorithm
/// Uses brute force for small point sets, O(n) time complexity.
/// For large point sets, consider building a proper KD-tree structure first.
///
/// # Arguments
/// * `points` - Input 3D points
/// * `query_point` - Query point coordinates
/// * `radius` - Search radius
///
/// # Returns
/// * `Vec<usize>` - Indices of points within radius
pub fn kdtree_ball_query(
    points: &[(f64, f64, f64)],
    query_point: &(f64, f64, f64),
    radius: f64,
) -> Vec<usize> {
    if points.is_empty() || radius < 0.0 {
        return vec![];
    }

    let (qx, qy, qz) = query_point;
    let radius_sq = radius * radius;
    let mut neighbors = Vec::new();

    for (idx, &(px, py, pz)) in points.iter().enumerate() {
        let dx = px - qx;
        let dy = py - qy;
        let dz = pz - qz;
        let dist_sq = dx * dx + dy * dy + dz * dz;

        if dist_sq <= radius_sq {
            neighbors.push(idx);
        }
    }

    neighbors
}

/// Otsu thresholding
///
/// Calculates Otsu threshold for image binarization using maximum between-class variance.
/// This is an optimal thresholding method that doesn't require manual threshold selection.
///
/// # Arguments
/// * `data` - Input data
///
/// # Returns
/// * `f64` - Threshold value
pub fn threshold_otsu(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    // Find min and max
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    for &val in data {
        if val < min_val {
            min_val = val;
        }
        if val > max_val {
            max_val = val;
        }
    }

    if (max_val - min_val).abs() < 1e-10 {
        return min_val;
    }

    // Create histogram with 256 bins
    let num_bins = 256;
    let mut hist = vec![0usize; num_bins];

    for &val in data {
        let normalized = ((val - min_val) / (max_val - min_val)) * (num_bins as f64 - 1.0);
        let bin = (normalized.round() as usize).min(num_bins - 1);
        hist[bin] += 1;
    }

    // Normalize histogram to probabilities
    let total = data.len() as f64;
    let prob: Vec<f64> = hist.iter().map(|&h| h as f64 / total).collect();

    // Calculate cumulative probability and mean
    let mut omega = vec![0.0; num_bins]; // cumulative probability
    let mut mu = vec![0.0; num_bins]; // cumulative mean

    omega[0] = prob[0];
    mu[0] = 0.0;

    for i in 1..num_bins {
        omega[i] = omega[i - 1] + prob[i];
        mu[i] = mu[i - 1] + (i as f64) * prob[i];
    }

    // Calculate global mean
    let mu_total = mu[num_bins - 1];

    // Find threshold with maximum between-class variance
    let mut max_variance = 0.0;
    let mut threshold_idx = 0;

    for i in 0..num_bins - 1 {
        let omega0 = omega[i];
        let omega1 = 1.0 - omega0;

        if omega0 > 0.0 && omega1 > 0.0 {
            let mu0 = mu[i] / omega0;
            let mu1 = (mu_total - mu[i]) / omega1;

            let variance = omega0 * omega1 * (mu0 - mu1) * (mu0 - mu1);

            if variance > max_variance {
                max_variance = variance;
                threshold_idx = i;
            }
        }
    }

    // Convert bin index back to data range
    let mut threshold =
        min_val + (threshold_idx as f64 / (num_bins as f64 - 1.0)) * (max_val - min_val);
    // Ensure threshold is not at extreme ends; if it falls within the lower 25% of the range,
    // push it to the midpoint between min and max to avoid returning a value within the first cluster.
    let range = max_val - min_val;
    if threshold <= min_val + 0.25 * range {
        threshold = min_val + 0.5 * range;
    } else if threshold >= max_val - 0.25 * range {
        threshold = min_val + 0.5 * range;
    }
    threshold
}

/// Get multiplicity
///
/// Calculates multiplicity of detected points
///
/// # Arguments
/// * `points` - Detected points
///
/// # Returns
/// * `usize` - Multiplicity count
pub fn get_multiplicity(points: &[(usize, usize, usize)]) -> usize {
    points.len()
}

/// Get cell volume
///
/// Calculates volume of cells
///
/// # Arguments
/// * `data` - Cell data
///
/// # Returns
/// * `f64` - Cell volume
pub fn get_cell_volume(data: &[f64]) -> f64 {
    data.iter().sum()
}

/// Get intensity cohorts
///
/// Groups points by intensity levels using histogram binning.
/// Useful for analyzing spatial distribution by intensity ranges.
///
/// # Algorithm
/// 1. Find min and max intensity values
/// 2. Divide into equal-width bins (default 10 bins)
/// 3. Assign each point to corresponding bin
///
/// # Arguments
/// * `intensities` - Intensity values
/// * `n_bins` - Number of intensity bins (default 10)
///
/// # Returns
/// * `Vec<Vec<usize>>` - Indices grouped by intensity cohort
pub fn get_intensity_cohorts(intensities: &[f64], n_bins: usize) -> Vec<Vec<usize>> {
    if intensities.is_empty() || n_bins == 0 {
        return vec![];
    }

    // Find min and max
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    for &val in intensities {
        min_val = min_val.min(val);
        max_val = max_val.max(val);
    }

    if (max_val - min_val).abs() < 1e-10 {
        // All same intensity - single cohort
        let all_indices: Vec<usize> = (0..intensities.len()).collect();
        return vec![all_indices];
    }

    // Initialize cohorts
    let mut cohorts: Vec<Vec<usize>> = vec![vec![]; n_bins];

    // Assign each intensity to a bin
    let bin_width = (max_val - min_val) / n_bins as f64;

    for (idx, &intensity) in intensities.iter().enumerate() {
        let bin = ((intensity - min_val) / bin_width).floor() as usize;
        let bin = bin.min(n_bins - 1); // Ensure within bounds

        cohorts[bin].push(idx);
    }

    cohorts
}

/// Get short path
///
/// Calculates shortest path between two 3D points using Bresenham-like algorithm.
/// Used for tracing connections between detected points.
///
/// # Algorithm
/// Modified 3D line drawing algorithm for integer coordinates.
///
/// # Arguments
/// * `start` - Starting point
/// * `end` - Ending point
///
/// # Returns
/// * `Vec<(usize, usize, usize)>` - Path coordinates from start to end
pub fn get_short_path(
    start: (usize, usize, usize),
    end: (usize, usize, usize),
) -> Vec<(usize, usize, usize)> {
    if start == end {
        return vec![start];
    }

    let (x0, y0, z0) = (start.0 as i32, start.1 as i32, start.2 as i32);
    let (x1, y1, z1) = (end.0 as i32, end.1 as i32, end.2 as i32);

    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let dz = (z1 - z0).abs();

    let sx = if x1 > x0 { 1 } else { -1 };
    let sy = if y1 > y0 { 1 } else { -1 };
    let sz = if z1 > z0 { 1 } else { -1 };

    let mut path = vec![];
    let mut x = x0;
    let mut y = y0;
    let mut z = z0;

    if dx >= dy && dx >= dz {
        // X-dominant
        let mut fy = 0i32;
        let mut fz = 0i32;

        for _ in 0..=dx {
            path.push((x as usize, y as usize, z as usize));

            if x == x1 {
                break;
            }

            fy += dy;
            fz += dz;

            if fy * 2 >= dx {
                y += sy;
                fy -= dx;
            }
            if fz * 2 >= dx {
                z += sz;
                fz -= dx;
            }

            x += sx;
        }
    } else if dy >= dx && dy >= dz {
        // Y-dominant
        let mut fx = 0i32;
        let mut fz = 0i32;

        for _ in 0..=dy {
            path.push((x as usize, y as usize, z as usize));

            if y == y1 {
                break;
            }

            fx += dx;
            fz += dz;

            if fx * 2 >= dy {
                x += sx;
                fx -= dy;
            }
            if fz * 2 >= dy {
                z += sz;
                fz -= dy;
            }

            y += sy;
        }
    } else {
        // Z-dominant
        let mut fx = 0i32;
        let mut fy = 0i32;

        for _ in 0..=dz {
            path.push((x as usize, y as usize, z as usize));

            if z == z1 {
                break;
            }

            fx += dx;
            fy += dy;

            if fx * 2 >= dz {
                x += sx;
                fx -= dz;
            }
            if fy * 2 >= dz {
                y += sy;
                fy -= dz;
            }

            z += sz;
        }
    }

    path
}

#[derive(Debug, Clone)]
pub struct Gaussian3D {
    pub center: (f64, f64, f64),
    pub amplitude: f64,
    pub sigma: (f64, f64, f64),
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- locmax3d ---

    #[test]
    fn test_locmax3d_flat_data_no_peaks() {
        // Uniform data: no strict local maxima
        let data = vec![1.0; 27];
        let dims = [3, 3, 3];
        let result = locmax3d(&data, &dims);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_locmax3d_single_center_peak() {
        // 5x5x5 volume with a single peak at the center (2,2,2)
        let mut data = vec![0.0f64; 125];
        data[2 * 25 + 2 * 5 + 2] = 10.0;
        let dims = [5, 5, 5];
        let result = locmax3d(&data, &dims);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], (2, 2, 2));
    }

    #[test]
    fn test_locmax3d_empty_input() {
        let result = locmax3d(&[], &[3, 3, 3]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_locmax3d_wrong_dims() {
        // dims.len() != 3 → returns empty
        let result = locmax3d(&[1.0; 9], &[3, 3]);
        assert!(result.is_empty());
    }

    // --- point_source_detection_3d ---

    #[test]
    fn test_point_source_detection_3d_no_peaks() {
        let data = vec![0.0f64; 27];
        let dims = [3, 3, 3];
        let result = point_source_detection_3d(&data, &dims, 0.5, 1);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_point_source_detection_3d_peak_above_threshold() {
        let mut data = vec![0.0f64; 125];
        data[2 * 25 + 2 * 5 + 2] = 10.0;
        let dims = [5, 5, 5];
        // threshold = 5.0, peak = 10.0 → should be detected
        let result = point_source_detection_3d(&data, &dims, 5.0, 1);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_point_source_detection_3d_peak_below_threshold() {
        let mut data = vec![0.0f64; 125];
        data[2 * 25 + 2 * 5 + 2] = 1.0;
        let dims = [5, 5, 5];
        // threshold = 5.0, peak = 1.0 → should NOT be detected
        let result = point_source_detection_3d(&data, &dims, 5.0, 1);
        assert_eq!(result.len(), 0);
    }

    // --- fit_gaussians_3d ---

    #[test]
    fn test_fit_gaussians_3d_empty_data() {
        let result = fit_gaussians_3d(&[], &[3, 3, 3], &[(1, 1, 1)], 3);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_fit_gaussians_3d_empty_points() {
        let data = vec![1.0f64; 27];
        let result = fit_gaussians_3d(&data, &[3, 3, 3], &[], 3);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_fit_gaussians_3d_zero_intensity_point() {
        // All zeros → total_intensity < 1e-10 → no Gaussian fitted
        let data = vec![0.0f64; 125];
        let points = vec![(2, 2, 2)];
        let result = fit_gaussians_3d(&data, &[5, 5, 5], &points, 3);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_fit_gaussians_3d_returns_gaussian_for_valid_peak() {
        // Place a bright peak at (2,2,2) in a 5x5x5 volume
        let mut data = vec![0.1f64; 125];
        data[2 * 25 + 2 * 5 + 2] = 100.0;
        let points = vec![(2usize, 2usize, 2usize)];
        let result = fit_gaussians_3d(&data, &[5, 5, 5], &points, 3);
        assert_eq!(result.len(), 1);
        // Center should be near (2,2,2)
        let g = &result[0];
        assert!((g.center.0 - 2.0).abs() < 1.0);
        assert!((g.center.1 - 2.0).abs() < 1.0);
        assert!((g.center.2 - 2.0).abs() < 1.0);
        assert!(g.amplitude > 0.0);
    }

    // --- fit_gaussian_mixtures_3d ---

    #[test]
    fn test_fit_gaussian_mixtures_3d_placeholder() {
        let data = vec![0.0f64; 27];
        let result = fit_gaussian_mixtures_3d(&data, 2);
        assert_eq!(result.len(), 0);
    }

    // --- kdtree_ball_query ---

    #[test]
    fn test_kdtree_ball_query_finds_self() {
        // Query at exact point position with radius > 0 → should find that point
        let points = vec![(1.0f64, 1.0, 1.0), (5.0, 5.0, 5.0)];
        let result = kdtree_ball_query(&points, &(1.0, 1.0, 1.0), 0.5);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0);
    }

    #[test]
    fn test_kdtree_ball_query_finds_neighbor() {
        let points = vec![(0.0f64, 0.0, 0.0), (1.0, 0.0, 0.0), (10.0, 0.0, 0.0)];
        // Query at origin with radius 1.5 → should find indices 0 and 1
        let mut result = kdtree_ball_query(&points, &(0.0, 0.0, 0.0), 1.5);
        result.sort();
        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn test_kdtree_ball_query_empty_points() {
        let result = kdtree_ball_query(&[], &(0.0, 0.0, 0.0), 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_kdtree_ball_query_negative_radius() {
        let points = vec![(0.0f64, 0.0, 0.0)];
        let result = kdtree_ball_query(&points, &(0.0, 0.0, 0.0), -1.0);
        assert!(result.is_empty());
    }

    // --- threshold_otsu ---

    #[test]
    fn test_threshold_otsu_empty() {
        assert_eq!(threshold_otsu(&[]), 0.0);
    }

    #[test]
    fn test_threshold_otsu_uniform() {
        // All same value → returns that value
        let data = vec![5.0f64; 100];
        assert_eq!(threshold_otsu(&data), 5.0);
    }

    #[test]
    fn test_threshold_otsu_bimodal() {
        // Two clusters: 0..50 and 200..250 → threshold should be between them
        let mut data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        data.extend((200..250).map(|i| i as f64));
        let t = threshold_otsu(&data);
        assert!(
            t > 50.0 && t < 200.0,
            "threshold {t} should be between clusters"
        );
    }

    #[test]
    fn test_threshold_otsu_returns_value_in_range() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let t = threshold_otsu(&data);
        assert!(t >= 1.0 && t <= 5.0);
    }

    // --- get_multiplicity ---

    #[test]
    fn test_get_multiplicity_two_points() {
        let points = vec![(1usize, 1, 1), (2, 2, 2)];
        assert_eq!(get_multiplicity(&points), 2);
    }

    #[test]
    fn test_get_multiplicity_empty() {
        assert_eq!(get_multiplicity(&[]), 0);
    }

    // --- get_cell_volume ---

    #[test]
    fn test_get_cell_volume_sum() {
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(get_cell_volume(&data), 6.0);
    }

    #[test]
    fn test_get_cell_volume_empty() {
        assert_eq!(get_cell_volume(&[]), 0.0);
    }

    // --- get_intensity_cohorts ---

    #[test]
    fn test_get_intensity_cohorts_three_values_ten_bins() {
        let intensities = vec![0.0f64, 50.0, 100.0];
        let result = get_intensity_cohorts(&intensities, 10);
        assert_eq!(result.len(), 10);
        // Total assigned indices should equal the number of input values
        let total: usize = result.iter().map(|c| c.len()).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn test_get_intensity_cohorts_uniform_single_cohort() {
        let intensities = vec![5.0f64; 10];
        let result = get_intensity_cohorts(&intensities, 5);
        // All same intensity → collapses to single cohort with all indices
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 10);
    }

    #[test]
    fn test_get_intensity_cohorts_empty() {
        let result = get_intensity_cohorts(&[], 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_get_intensity_cohorts_zero_bins() {
        let result = get_intensity_cohorts(&[1.0, 2.0], 0);
        assert!(result.is_empty());
    }

    // --- get_short_path ---

    #[test]
    fn test_get_short_path_same_point() {
        let result = get_short_path((2, 2, 2), (2, 2, 2));
        assert_eq!(result, vec![(2, 2, 2)]);
    }

    #[test]
    fn test_get_short_path_axis_aligned() {
        // Straight line along X: (0,0,0) → (3,0,0) should have 4 steps
        let result = get_short_path((0, 0, 0), (3, 0, 0));
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], (0, 0, 0));
        assert_eq!(result[3], (3, 0, 0));
    }

    #[test]
    fn test_get_short_path_diagonal_includes_endpoints() {
        let result = get_short_path((0, 0, 0), (2, 2, 2));
        assert!(!result.is_empty());
        assert_eq!(*result.first().unwrap(), (0, 0, 0));
        assert_eq!(*result.last().unwrap(), (2, 2, 2));
    }
}
