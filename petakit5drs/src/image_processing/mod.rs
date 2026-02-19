//! Image processing functions for Petakit5D
//!
//! This module contains core image processing operations for microscopy data.

use rustfft::{num_complex::Complex, FftPlanner};
use std::fmt;

/// Error type for image processing operations
#[derive(Debug, Clone, PartialEq)]
pub enum ImageProcessingError {
    /// Invalid array dimensions
    InvalidDimensions,
    /// Unsupported degree for spline computation
    UnsupportedDegree,
    /// Unsupported boundary condition
    UnsupportedBoundary,
    /// Unsupported mode for computation
    UnsupportedMode,
    /// Invalid kernel size
    InvalidKernelSize,
    /// Invalid filter parameters
    InvalidFilterParameters,
}

impl fmt::Display for ImageProcessingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImageProcessingError::InvalidDimensions => write!(f, "Invalid array dimensions"),
            ImageProcessingError::UnsupportedDegree => write!(f, "Unsupported spline degree"),
            ImageProcessingError::UnsupportedBoundary => {
                write!(f, "Unsupported boundary condition")
            }
            ImageProcessingError::UnsupportedMode => write!(f, "Unsupported computation mode"),
            ImageProcessingError::InvalidKernelSize => write!(f, "Invalid kernel size"),
            ImageProcessingError::InvalidFilterParameters => write!(f, "Invalid filter parameters"),
        }
    }
}

impl std::error::Error for ImageProcessingError {}

/// Compute B-spline coefficients for given data
///
/// This function computes B-spline coefficients using the Fourier method,
/// which is suitable for periodic or symmetric boundary conditions.
///
/// # Arguments
/// * `s` - Input data array (1D)
/// * `lambda_` - Regularization parameter (default: 0)
/// * `degree` - Spline degree (only degree 3 supported currently)
/// * `mode` - Computation mode (currently only "fourier" supported)
/// * `boundary` - Boundary condition ("symmetric" or "periodic")
///
/// # Returns
/// * `Ok(Vec<f64>)` - Computed coefficients
/// * `Err(ImageProcessingError)` - Error during computation
///
/// # Example
/// ```
/// use petakit5drs::image_processing::compute_bspline_coefficients;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let coeffs = compute_bspline_coefficients(&data, 0.0, 3, "fourier", "symmetric").unwrap();
/// ```
pub fn compute_bspline_coefficients(
    s: &[f64],
    lambda_: f64,
    degree: usize,
    mode: &str,
    boundary: &str,
) -> Result<Vec<f64>, ImageProcessingError> {
    // Validate inputs
    if degree != 3 {
        return Err(ImageProcessingError::UnsupportedDegree);
    }

    if boundary != "symmetric" && boundary != "periodic" {
        return Err(ImageProcessingError::UnsupportedBoundary);
    }

    if mode != "fourier" {
        return Err(ImageProcessingError::UnsupportedMode);
    }

    // Handle scalar case
    if s.len() == 1 {
        return Ok(s.to_vec());
    }

    // Handle 1D case
    if s.len() >= 1 {
        compute_bspline_1d(s, lambda_, boundary)
    } else {
        Err(ImageProcessingError::InvalidDimensions)
    }
}

/// Compute B-spline coefficients for 1D input
fn compute_bspline_1d(
    arr: &[f64],
    lambda_: f64,
    boundary: &str,
) -> Result<Vec<f64>, ImageProcessingError> {
    let n = arr.len();

    // Handle special case of single element
    if n == 1 {
        return Ok(arr.to_vec());
    }

    // Create extended array with boundary handling
    let s_m: Vec<f64>;
    let m: usize;

    if boundary == "symmetric" && n > 1 {
        // Mirror the array for symmetric boundary conditions
        let mirror: Vec<f64> = arr[1..n - 1].iter().rev().copied().collect();
        s_m = [arr, &mirror].concat();
        m = s_m.len();
    } else {
        s_m = arr.to_vec();
        m = n;
    }

    // Compute frequency grid
    let w: Vec<f64> = (0..m)
        .map(|i| 2.0 * std::f64::consts::PI * i as f64 / m as f64)
        .collect();

    // Compute filter coefficients
    let h: Vec<f64> = w
        .iter()
        .map(|&w_val| {
            3.0 / (2.0
                + w_val.cos()
                + 6.0 * lambda_ * (2.0 * w_val.cos() - 4.0 * w_val.cos() + 3.0))
        })
        .collect();

    // Perform FFT and apply filter
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(m);
    let mut data: Vec<Complex<f64>> = s_m.iter().map(|&x| Complex::new(x, 0.0)).collect();

    fft.process(&mut data);

    // Apply filter
    let filtered: Vec<Complex<f64>> = data
        .iter()
        .zip(h.iter())
        .map(|(x, &h_val)| *x * h_val)
        .collect();

    // Inverse FFT
    let ifft = planner.plan_fft_inverse(m);
    let mut result = filtered;
    ifft.process(&mut result);

    // Extract real part and return first n elements
    let real_result: Vec<f64> = result.into_iter().take(n).map(|c| c.re).collect();

    Ok(real_result)
}

/// 1D Gaussian filtering
///
/// Applies a Gaussian filter along the first axis of the input array
///
/// # Arguments
/// * `input` - Input data array
/// * `sigma` - Standard deviation of the Gaussian kernel
///
/// # Returns
/// * `Vec<f64>` - Filtered data
pub fn filter_gauss_1d(input: &[f64], sigma: f64) -> Vec<f64> {
    if input.is_empty() {
        return vec![];
    }

    if sigma <= 0.0 {
        return input.to_vec();
    }

    // Create Gaussian kernel (simplified implementation)
    let kernel_size = (sigma * 6.0).ceil() as usize;
    let kernel_size = kernel_size.max(1);
    let kernel_size = kernel_size + (kernel_size + 1) % 2; // Make odd

    let center = kernel_size / 2;
    let mut kernel = vec![0.0; kernel_size];

    // Create Gaussian kernel
    let sigma_squared = sigma * sigma;
    for i in 0..kernel_size {
        let x = (i as f64 - center as f64).abs();
        kernel[i] = (-0.5 * x * x / sigma_squared).exp();
    }

    // Normalize kernel
    let sum: f64 = kernel.iter().sum();
    if sum != 0.0 {
        for k in kernel.iter_mut() {
            *k /= sum;
        }
    }

    // Apply convolution
    let mut output = vec![0.0; input.len()];
    for i in 0..input.len() {
        let mut sum = 0.0;
        for j in 0..kernel_size {
            // Ensure we don't go out of bounds
            let idx = i as isize + j as isize - center as isize;
            if idx >= 0 && idx < input.len() as isize {
                sum += input[idx as usize] * kernel[j];
            }
        }
        output[i] = sum;
    }

    output
}

/// 2D Gaussian filtering
///
/// Applies a 2D Gaussian filter to an image
///
/// # Arguments
/// * `input` - Input 2D array (row-major)
/// * `rows` - Number of rows
/// * `cols` - Number of columns
/// * `sigma` - Standard deviation of the Gaussian kernel
///
/// # Returns
/// * `Vec<f64>` - Filtered data in same format
pub fn filter_gauss_2d(input: &[f64], rows: usize, cols: usize, sigma: f64) -> Vec<f64> {
    if input.is_empty() || rows == 0 || cols == 0 {
        return vec![];
    }

    if sigma <= 0.0 {
        return input.to_vec();
    }

    // Create 2D Gaussian kernel
    let kernel_size = (sigma * 6.0).ceil() as usize;
    let kernel_size = kernel_size.max(1);
    let kernel_size = kernel_size + (kernel_size + 1) % 2; // Make odd

    let center = kernel_size / 2;
    let mut kernel = vec![vec![0.0; kernel_size]; kernel_size];

    // Create Gaussian kernel
    let sigma_squared = sigma * sigma;
    for i in 0..kernel_size {
        for j in 0..kernel_size {
            let x = (i as f64 - center as f64).abs();
            let y = (j as f64 - center as f64).abs();
            kernel[i][j] = (-0.5 * (x * x + y * y) / sigma_squared).exp();
        }
    }

    // Normalize kernel
    let sum: f64 = kernel.iter().flatten().sum();
    for row in kernel.iter_mut() {
        for k in row.iter_mut() {
            *k /= sum;
        }
    }

    // Apply convolution
    let mut output = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0;
            for ki in 0..kernel_size {
                for kj in 0..kernel_size {
                    let ni = i as isize + ki as isize - center as isize;
                    let nj = j as isize + kj as isize - center as isize;

                    if ni >= 0 && ni < rows as isize && nj >= 0 && nj < cols as isize {
                        let idx = (ni as usize) * cols + (nj as usize);
                        sum += input[idx] * kernel[ki][kj];
                    }
                }
            }
            output[i * cols + j] = sum;
        }
    }

    output
}

/// Fast 3D convolution with separable kernels
///
/// Applies separable 1D convolutions along each axis for efficient 3D filtering.
/// This is much faster than full 3D convolution while maintaining quality.
///
/// # Arguments
/// * `input` - Input 3D array (flattened, row-major order)
/// * `dims` - Dimensions [depth, height, width]
/// * `kernel` - 1D separable kernel (same kernel applied to all three axes)
///
/// # Returns
/// * `Vec<f64>` - Convolved data
pub fn conv3_fast(input: &[f64], dims: &[usize], kernel: &[f64]) -> Vec<f64> {
    if input.is_empty() || dims.len() != 3 || kernel.is_empty() {
        return input.to_vec();
    }

    let [depth, height, width] = [dims[0], dims[1], dims[2]];

    // Handle edge case of single element
    if depth * height * width != input.len() {
        return input.to_vec();
    }

    let kernel_radius = kernel.len() / 2;

    // Step 1: Convolve along X axis (width dimension)
    let mut result_x = vec![0.0; depth * height * width];
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                for k in 0..kernel.len() {
                    let ki = x as isize + k as isize - kernel_radius as isize;
                    if ki >= 0 && ki < width as isize {
                        let idx = z * height * width + y * width + ki as usize;
                        sum += input[idx] * kernel[k];
                        weight_sum += kernel[k];
                    }
                }

                let out_idx = z * height * width + y * width + x;
                result_x[out_idx] = if weight_sum != 0.0 {
                    sum / weight_sum
                } else {
                    0.0
                };
            }
        }
    }

    // Step 2: Convolve along Y axis (height dimension)
    let mut result_y = vec![0.0; depth * height * width];
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                for k in 0..kernel.len() {
                    let ki = y as isize + k as isize - kernel_radius as isize;
                    if ki >= 0 && ki < height as isize {
                        let idx = z * height * width + ki as usize * width + x;
                        sum += result_x[idx] * kernel[k];
                        weight_sum += kernel[k];
                    }
                }

                let out_idx = z * height * width + y * width + x;
                result_y[out_idx] = if weight_sum != 0.0 {
                    sum / weight_sum
                } else {
                    0.0
                };
            }
        }
    }

    // Step 3: Convolve along Z axis (depth dimension)
    let mut result_z = vec![0.0; depth * height * width];
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                for k in 0..kernel.len() {
                    let ki = z as isize + k as isize - kernel_radius as isize;
                    if ki >= 0 && ki < depth as isize {
                        let idx = ki as usize * height * width + y * width + x;
                        sum += result_y[idx] * kernel[k];
                        weight_sum += kernel[k];
                    }
                }

                let out_idx = z * height * width + y * width + x;
                result_z[out_idx] = if weight_sum != 0.0 {
                    sum / weight_sum
                } else {
                    0.0
                };
            }
        }
    }

    result_z
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_bspline_coefficients_single_element() {
        let data = vec![5.0];
        let result = compute_bspline_coefficients(&data, 0.0, 3, "fourier", "symmetric");
        assert!(result.is_ok());
    }

    #[test]
    fn test_compute_bspline_coefficients_invalid_degree() {
        let data = vec![1.0, 2.0, 3.0];
        let result = compute_bspline_coefficients(&data, 0.0, 2, "fourier", "symmetric");
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_bspline_coefficients_unsupported_boundary() {
        let data = vec![1.0, 2.0, 3.0];
        let result = compute_bspline_coefficients(&data, 0.0, 3, "fourier", "unsupported");
        assert!(result.is_err());
    }

    #[test]
    fn test_filter_gauss_1d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = filter_gauss_1d(&data, 1.0);
        assert_eq!(result.len(), data.len());
        // Should be similar to original but smoothed
    }

    #[test]
    fn test_filter_gauss_2d() {
        let data = vec![1.0; 25]; // 5x5 array
        let result = filter_gauss_2d(&data, 5, 5, 1.0);
        assert_eq!(result.len(), 25);
        // Should be similar to original but smoothed
    }
}
