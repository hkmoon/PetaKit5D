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

/// 3D Gaussian filtering
///
/// Applies a separable 3D Gaussian filter to a volume.
/// Supports isotropic (single sigma) or anisotropic (sigma_xy, sigma_z) filtering.
///
/// # Arguments
/// * `input` - Input 3D array (flattened, row-major order)
/// * `depth` - Number of Z slices
/// * `height` - Number of rows (Y)
/// * `width` - Number of columns (X)
/// * `sigma_xy` - Standard deviation for X and Y axes
/// * `sigma_z` - Standard deviation for Z axis
///
/// # Returns
/// * `Vec<f64>` - Filtered volume
pub fn filter_gauss_3d(
    input: &[f64],
    depth: usize,
    height: usize,
    width: usize,
    sigma_xy: f64,
    sigma_z: f64,
) -> Vec<f64> {
    if input.is_empty() || depth == 0 || height == 0 || width == 0 {
        return vec![];
    }

    if sigma_xy <= 0.0 && sigma_z <= 0.0 {
        return input.to_vec();
    }

    let make_kernel = |sigma: f64| -> Vec<f64> {
        if sigma <= 0.0 {
            return vec![1.0];
        }
        let w = (3.0 * sigma).ceil() as usize;
        let size = 2 * w + 1;
        let sigma_sq = sigma * sigma;
        let mut k: Vec<f64> = (0..size)
            .map(|i| {
                let x = i as f64 - w as f64;
                (-0.5 * x * x / sigma_sq).exp()
            })
            .collect();
        let sum: f64 = k.iter().sum();
        if sum != 0.0 {
            k.iter_mut().for_each(|v| *v /= sum);
        }
        k
    };

    let k_xy = make_kernel(sigma_xy);
    let k_z = make_kernel(sigma_z);
    let r_xy = k_xy.len() / 2;
    let r_z = k_z.len() / 2;

    // Convolve along X
    let mut buf = vec![0.0f64; depth * height * width];
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                for (ki, &kv) in k_xy.iter().enumerate() {
                    let xi = x as isize + ki as isize - r_xy as isize;
                    if xi >= 0 && xi < width as isize {
                        sum += input[z * height * width + y * width + xi as usize] * kv;
                    }
                }
                buf[z * height * width + y * width + x] = sum;
            }
        }
    }

    // Convolve along Y
    let buf_x = buf.clone();
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                for (ki, &kv) in k_xy.iter().enumerate() {
                    let yi = y as isize + ki as isize - r_xy as isize;
                    if yi >= 0 && yi < height as isize {
                        sum += buf_x[z * height * width + yi as usize * width + x] * kv;
                    }
                }
                buf[z * height * width + y * width + x] = sum;
            }
        }
    }

    // Convolve along Z
    let buf_y = buf.clone();
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                for (ki, &kv) in k_z.iter().enumerate() {
                    let zi = z as isize + ki as isize - r_z as isize;
                    if zi >= 0 && zi < depth as isize {
                        sum += buf_y[zi as usize * height * width + y * width + x] * kv;
                    }
                }
                buf[z * height * width + y * width + x] = sum;
            }
        }
    }

    buf
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

// ---------------------------------------------------------------------------
// B-spline coefficient computation (forward transform)
// ---------------------------------------------------------------------------

/// Compute forward 1D cubic B-spline coefficients along axis 1 (columns).
///
/// Equivalent to `bspline_coeffs.b3spline_1d` in Python.
///
/// # Arguments
/// * `data`     — Input, row-major order, length `ny * nx`.
/// * `ny`, `nx` — Row and column counts.
/// * `boundary` — `"mirror"` (symmetric) or `"periodic"`.
///
/// # Returns
/// `Ok(Vec<f64>)` — B-spline coefficients, same length as `data`.
pub fn b3spline_1d(
    data: &[f64],
    ny: usize,
    nx: usize,
    boundary: &str,
) -> Result<Vec<f64>, ImageProcessingError> {
    if boundary != "mirror" && boundary != "periodic" {
        return Err(ImageProcessingError::UnsupportedBoundary);
    }
    if data.len() != ny * nx {
        return Err(ImageProcessingError::InvalidDimensions);
    }

    // Cubic B-spline parameters
    let z1: f64 = -2.0 + 3.0_f64.sqrt();
    let c0: f64 = 6.0;

    let mut cp = data.to_vec(); // causal pass
    let mut cn = vec![0.0f64; ny * nx]; // anti-causal pass

    for row in 0..ny {
        let base = row * nx;

        // ------- causal init -------
        if boundary == "mirror" {
            // sum_{k=0}^{inf} z1^k * s[k]  (mirror at 0)
            let mut acc = 0.0f64;
            let mut zk = 1.0f64;
            let sum_limit = ((-7.0 / z1.ln()) as usize).min(nx - 1);
            for k in 0..=sum_limit {
                acc += data[base + k] * zk;
                zk *= z1;
            }
            cp[base] = acc;
        } else {
            // periodic: init by solving linear system  (approximation)
            let zn = z1.powi(nx as i32);
            let mut num = data[base];
            let mut zk = z1;
            for k in 1..nx {
                num += data[base + k] * zk;
                zk *= z1;
            }
            cp[base] = num / (1.0 - zn);
        }
        for k in 1..nx {
            cp[base + k] = data[base + k] + z1 * cp[base + k - 1];
        }

        // ------- anti-causal init -------
        if boundary == "mirror" {
            cn[base + nx - 1] =
                (z1 / (z1 * z1 - 1.0)) * (cp[base + nx - 1] + z1 * cp[base + nx - 2]);
        } else {
            // periodic
            let zn = z1.powi(nx as i32);
            let z2 = z1 * z1;
            let zn1 = z1.powi((nx - 1) as i32);
            cn[base + nx - 1] =
                z1 / (z2 - 1.0) * (cp[base + nx - 1] * z1 + cp[base] * zn1 / (1.0 - zn));
        }
        for k in (0..nx - 1).rev() {
            cn[base + k] = z1 * (cn[base + k + 1] - cp[base + k]);
        }

        // scale
        for k in 0..nx {
            cn[base + k] *= c0;
        }
    }

    Ok(cn)
}

/// Compute forward 2D cubic B-spline coefficients.
///
/// Applies [`b3spline_1d`] along columns then along rows (separable).
///
/// # Arguments
/// * `data`     — Row-major input, length `ny * nx`.
/// * `ny`, `nx` — Dimensions.
/// * `boundary` — `"mirror"` or `"periodic"`.
pub fn b3spline_2d(
    data: &[f64],
    ny: usize,
    nx: usize,
    boundary: &str,
) -> Result<Vec<f64>, ImageProcessingError> {
    // First pass: along columns (axis 1)
    let after_x = b3spline_1d(data, ny, nx, boundary)?;
    // Transpose → (nx, ny) then apply along the new "columns" (original rows)
    let mut transposed = vec![0.0f64; ny * nx];
    for r in 0..ny {
        for c in 0..nx {
            transposed[c * ny + r] = after_x[r * nx + c];
        }
    }
    let after_y = b3spline_1d(&transposed, nx, ny, boundary)?;
    // Transpose back
    let mut result = vec![0.0f64; ny * nx];
    for r in 0..ny {
        for c in 0..nx {
            result[r * nx + c] = after_y[c * ny + r];
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Contrast utilities
// ---------------------------------------------------------------------------

/// Scale contrast: linearly maps `[in_min, in_max]` → `[out_min, out_max]`.
///
/// When `in_max == in_min` the output is all zeros.
pub fn scale_contrast(
    data: &[f64],
    in_min: f64,
    in_max: f64,
    out_min: f64,
    out_max: f64,
) -> Vec<f64> {
    let range_in = in_max - in_min;
    if range_in == 0.0 {
        return vec![0.0; data.len()];
    }
    let scale = (out_max - out_min) / range_in;
    data.iter()
        .map(|&v| (v - in_min) * scale + out_min)
        .collect()
}

/// Invert contrast: maps `v → (in_min + in_max) - v`, clipped to `[in_min, in_max]`.
pub fn invert_contrast(data: &[f64], in_min: f64, in_max: f64) -> Vec<f64> {
    let sum = in_min + in_max;
    data.iter()
        .map(|&v| sum - v.clamp(in_min, in_max))
        .collect()
}

// ---------------------------------------------------------------------------
// Local statistics (2D sliding window)
// ---------------------------------------------------------------------------

/// Compute local mean and standard deviation over a square window.
///
/// Window uses constant (zero) padding beyond image borders.
/// Returns `(mean, std_dev)` as flat row-major vectors of length `rows * cols`.
///
/// # Arguments
/// * `data`        — Flat 2-D input (row-major), length `rows * cols`.
/// * `rows`, `cols`— Image dimensions.
/// * `window_size` — Side length of square window (must be ≥ 1).
pub fn local_avg_std_2d(
    data: &[f64],
    rows: usize,
    cols: usize,
    window_size: usize,
) -> (Vec<f64>, Vec<f64>) {
    if data.is_empty() || rows == 0 || cols == 0 || window_size == 0 {
        return (vec![], vec![]);
    }
    let half = (window_size / 2) as isize;
    let n_total = rows * cols;
    let mut avg = vec![0.0f64; n_total];
    let mut std = vec![0.0f64; n_total];

    for r in 0..rows {
        for c in 0..cols {
            let mut sum = 0.0;
            let mut sum2 = 0.0;
            let mut count = 0usize;
            let r0 = r as isize - half;
            let c0 = c as isize - half;
            for wr in r0..r0 + window_size as isize {
                for wc in c0..c0 + window_size as isize {
                    if wr >= 0 && wr < rows as isize && wc >= 0 && wc < cols as isize {
                        let v = data[wr as usize * cols + wc as usize];
                        sum += v;
                        sum2 += v * v;
                        count += 1;
                    }
                }
            }
            if count > 0 {
                let mean = sum / count as f64;
                avg[r * cols + c] = mean;
                if count > 1 {
                    let var = (sum2 - sum * sum / count as f64) / (count - 1) as f64;
                    std[r * cols + c] = var.max(0.0).sqrt();
                }
            }
        }
    }
    (avg, std)
}

// ---------------------------------------------------------------------------
// Gaussian gradient
// ---------------------------------------------------------------------------

/// Compute Gaussian gradient in X and Y directions for a 2-D image.
///
/// Applies separable derivative-of-Gaussian filters.
///
/// # Arguments
/// * `data`        — Flat 2-D input (row-major), length `rows * cols`.
/// * `rows`, `cols`— Image dimensions.
/// * `sigma`       — Gaussian standard deviation.
///
/// # Returns
/// `(dX, dY)` — gradient in X (columns) and Y (rows) directions.
pub fn gradient_filter_gauss_2d(
    data: &[f64],
    rows: usize,
    cols: usize,
    sigma: f64,
) -> (Vec<f64>, Vec<f64>) {
    if data.is_empty() || sigma <= 0.0 {
        return (data.to_vec(), data.to_vec());
    }
    let w = (3.0 * sigma).ceil() as usize;
    let size = 2 * w + 1;
    let sigma2 = sigma * sigma;

    let mut g = vec![0.0f64; size];
    let mut dg = vec![0.0f64; size];
    for i in 0..size {
        let x = i as f64 - w as f64;
        let gv = (-0.5 * x * x / sigma2).exp();
        g[i] = gv;
        dg[i] = -x / sigma2 * gv;
    }
    let gsum: f64 = g.iter().sum();
    g.iter_mut().for_each(|v| *v /= gsum);
    dg.iter_mut().for_each(|v| *v /= gsum);

    // Convolve row-major 2D data with 1D kernel along given axis
    let conv1d = |src: &[f64], kernel: &[f64], axis: usize| -> Vec<f64> {
        let klen = kernel.len();
        let khalf = klen / 2;
        let mut out = vec![0.0f64; rows * cols];
        if axis == 0 {
            // along rows (Y)
            for r in 0..rows {
                for c in 0..cols {
                    let mut s = 0.0;
                    for k in 0..klen {
                        let ri = r as isize + k as isize - khalf as isize;
                        if ri >= 0 && ri < rows as isize {
                            s += src[ri as usize * cols + c] * kernel[k];
                        }
                    }
                    out[r * cols + c] = s;
                }
            }
        } else {
            // along cols (X)
            for r in 0..rows {
                for c in 0..cols {
                    let mut s = 0.0;
                    for k in 0..klen {
                        let ci = c as isize + k as isize - khalf as isize;
                        if ci >= 0 && ci < cols as isize {
                            s += src[r * cols + ci as usize] * kernel[k];
                        }
                    }
                    out[r * cols + c] = s;
                }
            }
        }
        out
    };

    // dX: derivative along X (axis 1), smoothed along Y (axis 0)
    let tmp = conv1d(data, &dg, 1);
    let dx = conv1d(&tmp, &g, 0);

    // dY: derivative along Y (axis 0), smoothed along X (axis 1)
    let tmp = conv1d(data, &g, 1);
    let dy = conv1d(&tmp, &dg, 0);

    (dx, dy)
}

/// Compute Gaussian gradient in X, Y, Z directions for a 3-D volume.
///
/// # Arguments
/// * `data`          — Flat ZYX (row-major) input, length `nz * ny * nx`.
/// * `nz`, `ny`, `nx`— Volume dimensions.
/// * `sigma`         — Gaussian standard deviation (isotropic).
///
/// # Returns
/// `(dX, dY, dZ)` — gradient volumes in X, Y, Z directions.
pub fn gradient_filter_gauss_3d(
    data: &[f64],
    nz: usize,
    ny: usize,
    nx: usize,
    sigma: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if data.is_empty() || sigma <= 0.0 {
        let d = data.to_vec();
        return (d.clone(), d.clone(), d);
    }
    let w = (3.0 * sigma).ceil() as usize;
    let size = 2 * w + 1;
    let sigma2 = sigma * sigma;

    let mut g = vec![0.0f64; size];
    let mut dg = vec![0.0f64; size];
    for i in 0..size {
        let x = i as f64 - w as f64;
        let gv = (-0.5 * x * x / sigma2).exp();
        g[i] = gv;
        dg[i] = -x / sigma2 * gv;
    }
    let gsum: f64 = g.iter().sum();
    g.iter_mut().for_each(|v| *v /= gsum);
    dg.iter_mut().for_each(|v| *v /= gsum);

    let nynx = ny * nx;
    // conv along axis: 0=Z, 1=Y, 2=X
    let conv3d = |src: &[f64], kernel: &[f64], axis: usize| -> Vec<f64> {
        let klen = kernel.len();
        let khalf = klen / 2;
        let mut out = vec![0.0f64; nz * nynx];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let mut s = 0.0;
                    for k in 0..klen {
                        let offset = k as isize - khalf as isize;
                        let (zi, yi, xi) = match axis {
                            0 => (z as isize + offset, y as isize, x as isize),
                            1 => (z as isize, y as isize + offset, x as isize),
                            _ => (z as isize, y as isize, x as isize + offset),
                        };
                        if zi >= 0
                            && zi < nz as isize
                            && yi >= 0
                            && yi < ny as isize
                            && xi >= 0
                            && xi < nx as isize
                        {
                            s += src[zi as usize * nynx + yi as usize * nx + xi as usize]
                                * kernel[k];
                        }
                    }
                    out[z * nynx + y * nx + x] = s;
                }
            }
        }
        out
    };

    // dX (derivative along X=axis2, smooth Y and Z)
    let tmp = conv3d(data, &dg, 2);
    let tmp = conv3d(&tmp, &g, 1);
    let dx = conv3d(&tmp, &g, 0);

    // dY
    let tmp = conv3d(data, &g, 2);
    let tmp = conv3d(&tmp, &dg, 1);
    let dy = conv3d(&tmp, &g, 0);

    // dZ
    let tmp = conv3d(data, &g, 2);
    let tmp = conv3d(&tmp, &g, 1);
    let dz = conv3d(&tmp, &dg, 0);

    (dx, dy, dz)
}

// ---------------------------------------------------------------------------
// Morphology and neighborhood utilities
// ---------------------------------------------------------------------------

/// Create a 3×3×3 boolean neighborhood for 3-D morphological operations.
///
/// * `conn = 6`  — face neighbors only (6-connected)
/// * `conn = 18` — face + edge neighbors
/// * `conn = 26` — all neighbors except centre (26-connected, default)
///
/// Returns a flat ZYX array of length 27.
pub fn bwn_hood_3d(conn: u8) -> Result<Vec<bool>, ImageProcessingError> {
    let mut hood = vec![false; 27];
    match conn {
        6 => {
            for i in [4, 10, 12, 14, 16, 22usize] {
                hood[i] = true;
            }
        }
        18 => {
            let corners = [0, 2, 6, 8, 18, 20, 24, 26usize];
            for i in 0..27usize {
                if i != 13 && !corners.contains(&i) {
                    hood[i] = true;
                }
            }
        }
        26 => {
            for i in 0..27usize {
                if i != 13 {
                    hood[i] = true;
                }
            }
        }
        _ => return Err(ImageProcessingError::InvalidFilterParameters),
    }
    Ok(hood)
}

/// Create a 3-D spherical structuring element.
///
/// Returns a flat ZYX array of length `(2w+1)^3` where `w = floor(radius)`.
/// Each element is `true` when the voxel lies within `radius` of the centre.
pub fn binary_sphere(radius: f64) -> Result<(Vec<bool>, usize), ImageProcessingError> {
    if radius <= 0.0 {
        return Err(ImageProcessingError::InvalidFilterParameters);
    }
    let w = radius.floor() as usize;
    let side = 2 * w + 1;
    let mut sphere = vec![false; side * side * side];
    let r2 = radius * radius;
    for z in 0..side {
        for y in 0..side {
            for x in 0..side {
                let dz = (z as f64 - w as f64).powi(2);
                let dy = (y as f64 - w as f64).powi(2);
                let dx = (x as f64 - w as f64).powi(2);
                if dz + dy + dx <= r2 {
                    sphere[z * side * side + y * side + x] = true;
                }
            }
        }
    }
    Ok((sphere, side))
}

/// Keep only the largest connected component in a 3-D binary mask.
///
/// Uses 6-connectivity (face neighbours). Returns a flat boolean vector
/// with the same length as `mask`, where only the largest connected
/// component is `true`.
pub fn bw_largest_obj(mask: &[bool], nz: usize, ny: usize, nx: usize) -> Vec<bool> {
    if mask.len() != nz * ny * nx {
        return mask.to_vec();
    }
    let nynx = ny * nx;
    let n = nz * nynx;

    // BFS / flood-fill with 6-connectivity
    let mut label = vec![0u32; n];
    let mut current_label = 0u32;
    let mut component_size: Vec<usize> = vec![0]; // index 0 is background

    let neighbours = |idx: usize| -> Vec<usize> {
        let z = (idx / nynx) as isize;
        let y = ((idx % nynx) / nx) as isize;
        let x = (idx % nx) as isize;
        let mut nb = Vec::with_capacity(6);
        for (dz, dy, dx) in [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ] {
            let nz2 = z + dz;
            let ny2 = y + dy;
            let nx2 = x + dx;
            if nz2 >= 0
                && nz2 < nz as isize
                && ny2 >= 0
                && ny2 < ny as isize
                && nx2 >= 0
                && nx2 < nx as isize
            {
                nb.push(nz2 as usize * nynx + ny2 as usize * nx + nx2 as usize);
            }
        }
        nb
    };

    for start in 0..n {
        if !mask[start] || label[start] != 0 {
            continue;
        }
        current_label += 1;
        label[start] = current_label;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        let mut size = 0usize;
        while let Some(idx) = queue.pop_front() {
            size += 1;
            for nb in neighbours(idx) {
                if mask[nb] && label[nb] == 0 {
                    label[nb] = current_label;
                    queue.push_back(nb);
                }
            }
        }
        component_size.push(size);
    }

    if current_label == 0 {
        return vec![false; n];
    }

    let largest = component_size[1..]
        .iter()
        .enumerate()
        .max_by_key(|(_, &s)| s)
        .map(|(i, _)| i as u32 + 1)
        .unwrap_or(0);

    label.iter().map(|&l| l == largest).collect()
}

// ---------------------------------------------------------------------------
// Fast 3-D Gaussian with border correction
// ---------------------------------------------------------------------------

/// Fast 3-D Gaussian filter with border correction.
///
/// Applies a separable Gaussian with `sigma_xy` along X and Y, and `sigma_z` along Z.
/// Border correction divides by the local kernel weight sum, reducing edge darkening.
///
/// # Arguments
/// * `data`           — Flat ZYX input, length `nz * ny * nx`.
/// * `nz`, `ny`, `nx` — Volume dimensions.
/// * `sigma_xy`       — Standard deviation in X and Y.
/// * `sigma_z`        — Standard deviation in Z.
/// * `correct_border` — When `true`, normalize by local weight sum at borders.
pub fn fast_gauss_3d(
    data: &[f64],
    nz: usize,
    ny: usize,
    nx: usize,
    sigma_xy: f64,
    sigma_z: f64,
    correct_border: bool,
) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }

    let make_kernel = |sigma: f64| -> Vec<f64> {
        if sigma <= 0.0 {
            return vec![1.0];
        }
        let w = (4.0 * sigma).ceil() as usize; // 4-sigma support
        let size = 2 * w + 1;
        let sigma2 = sigma * sigma;
        let mut k: Vec<f64> = (0..size)
            .map(|i| {
                let x = i as f64 - w as f64;
                (-0.5 * x * x / sigma2).exp()
            })
            .collect();
        let s: f64 = k.iter().sum();
        if s != 0.0 {
            k.iter_mut().for_each(|v| *v /= s);
        }
        k
    };

    let k_xy = make_kernel(sigma_xy);
    let k_z = make_kernel(sigma_z);

    let nynx = ny * nx;

    let conv1d_border =
        |src: &[f64], kernel: &[f64], axis: usize, correct: bool| -> Vec<f64> {
            let klen = kernel.len();
            let khalf = klen / 2;
            let mut out = vec![0.0f64; nz * nynx];
            let (dim_nz, dim_ny, dim_nx) = (nz as isize, ny as isize, nx as isize);
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let mut s = 0.0;
                        let mut ws = 0.0;
                        for k in 0..klen {
                            let offset = k as isize - khalf as isize;
                            let (zi, yi, xi) = match axis {
                                0 => (z as isize + offset, y as isize, x as isize),
                                1 => (z as isize, y as isize + offset, x as isize),
                                _ => (z as isize, y as isize, x as isize + offset),
                            };
                            if zi >= 0
                                && zi < dim_nz
                                && yi >= 0
                                && yi < dim_ny
                                && xi >= 0
                                && xi < dim_nx
                            {
                                let kv = kernel[k];
                                s += src[zi as usize * nynx + yi as usize * nx + xi as usize] * kv;
                                ws += kv;
                            }
                        }
                        let v = if correct && ws > 0.0 { s / ws } else { s };
                        out[z * nynx + y * nx + x] = v;
                    }
                }
            }
            out
        };

    let tmp = conv1d_border(data, &k_xy, 2, correct_border); // X
    let tmp = conv1d_border(&tmp, &k_xy, 1, correct_border); // Y
    conv1d_border(&tmp, &k_z, 0, correct_border) // Z
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

    #[test]
    fn test_filter_gauss_3d_isotropic() {
        let data = vec![1.0; 125]; // 5x5x5 volume
        let result = filter_gauss_3d(&data, 5, 5, 5, 1.0, 1.0);
        assert_eq!(result.len(), 125);
    }

    #[test]
    fn test_filter_gauss_3d_anisotropic() {
        let data = vec![1.0; 150]; // 5x5x6 volume
        let result = filter_gauss_3d(&data, 5, 5, 6, 2.0, 1.0);
        assert_eq!(result.len(), 150);
    }

    #[test]
    fn test_filter_gauss_3d_passthrough() {
        let data = vec![1.0; 27]; // 3x3x3 volume, sigma=0
        let result = filter_gauss_3d(&data, 3, 3, 3, 0.0, 0.0);
        assert_eq!(result, data);
    }

    #[test]
    fn test_filter_gauss_3d_empty() {
        let result = filter_gauss_3d(&[], 5, 5, 5, 1.0, 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_b3spline_1d_shape() {
        let data: Vec<f64> = (0..60).map(|i| i as f64).collect(); // 3×20
        let out = b3spline_1d(&data, 3, 20, "mirror").unwrap();
        assert_eq!(out.len(), 60);
    }

    #[test]
    fn test_b3spline_1d_bad_boundary() {
        let data = vec![1.0f64; 20];
        assert!(b3spline_1d(&data, 1, 20, "unknown").is_err());
    }

    #[test]
    fn test_b3spline_2d_shape() {
        let data = vec![1.0f64; 100]; // 10×10
        let out = b3spline_2d(&data, 10, 10, "mirror").unwrap();
        assert_eq!(out.len(), 100);
    }

    #[test]
    fn test_scale_contrast_basic() {
        let data = vec![0.0, 50.0, 100.0];
        let out = scale_contrast(&data, 0.0, 100.0, 0.0, 1.0);
        assert!((out[0] - 0.0).abs() < 1e-9);
        assert!((out[1] - 0.5).abs() < 1e-9);
        assert!((out[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_scale_contrast_flat() {
        let data = vec![5.0; 4];
        let out = scale_contrast(&data, 5.0, 5.0, 0.0, 1.0);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_invert_contrast() {
        let data = vec![0.0, 50.0, 100.0];
        let out = invert_contrast(&data, 0.0, 100.0);
        assert!((out[0] - 100.0).abs() < 1e-9);
        assert!((out[1] - 50.0).abs() < 1e-9);
        assert!((out[2] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_local_avg_std_2d_shape() {
        let data = vec![1.0f64; 100]; // 10×10
        let (avg, std) = local_avg_std_2d(&data, 10, 10, 3);
        assert_eq!(avg.len(), 100);
        assert_eq!(std.len(), 100);
    }

    #[test]
    fn test_local_avg_std_2d_uniform() {
        let data = vec![2.0f64; 25]; // 5×5 all 2.0
        let (avg, std) = local_avg_std_2d(&data, 5, 5, 3);
        for v in &avg {
            assert!((v - 2.0).abs() < 1e-9);
        }
        for v in &std {
            assert!(v.abs() < 1e-9);
        }
    }

    #[test]
    fn test_gradient_filter_gauss_2d_shape() {
        let data = vec![1.0f64; 100]; // 10×10
        let (dx, dy) = gradient_filter_gauss_2d(&data, 10, 10, 1.0);
        assert_eq!(dx.len(), 100);
        assert_eq!(dy.len(), 100);
    }

    #[test]
    fn test_gradient_filter_gauss_2d_uniform() {
        // Gradient of a uniform field should be ≈0 at interior pixels.
        // Border pixels may have non-zero gradient due to zero-padding.
        let rows = 10usize;
        let cols = 10usize;
        let data = vec![5.0f64; rows * cols];
        let (dx, dy) = gradient_filter_gauss_2d(&data, rows, cols, 1.0);
        // Check only interior (3 pixels from each border) to avoid pad artifacts
        let margin = 3;
        for r in margin..rows - margin {
            for c in margin..cols - margin {
                let vx = dx[r * cols + c];
                let vy = dy[r * cols + c];
                assert!(vx.abs() < 1e-6, "interior dX[{r},{c}] = {vx}");
                assert!(vy.abs() < 1e-6, "interior dY[{r},{c}] = {vy}");
            }
        }
    }

    #[test]
    fn test_gradient_filter_gauss_3d_shape() {
        let data = vec![1.0f64; 125]; // 5×5×5
        let (dx, dy, dz) = gradient_filter_gauss_3d(&data, 5, 5, 5, 1.0);
        assert_eq!(dx.len(), 125);
        assert_eq!(dy.len(), 125);
        assert_eq!(dz.len(), 125);
    }

    #[test]
    fn test_bwn_hood_3d_counts() {
        let h6 = bwn_hood_3d(6).unwrap();
        let h18 = bwn_hood_3d(18).unwrap();
        let h26 = bwn_hood_3d(26).unwrap();
        assert_eq!(h6.iter().filter(|&&v| v).count(), 6);
        assert_eq!(h18.iter().filter(|&&v| v).count(), 18);
        assert_eq!(h26.iter().filter(|&&v| v).count(), 26);
        // centre is always false
        assert!(!h6[13]);
        assert!(!h18[13]);
        assert!(!h26[13]);
    }

    #[test]
    fn test_bwn_hood_3d_invalid() {
        assert!(bwn_hood_3d(7).is_err());
    }

    #[test]
    fn test_binary_sphere_centre() {
        let (sphere, side) = binary_sphere(2.0).unwrap();
        let w = 2usize;
        let centre = w * side * side + w * side + w;
        assert!(sphere[centre]);
    }

    #[test]
    fn test_binary_sphere_invalid() {
        assert!(binary_sphere(0.0).is_err());
        assert!(binary_sphere(-1.0).is_err());
    }

    #[test]
    fn test_bw_largest_obj_single_component() {
        let mut mask = vec![false; 27]; // 3×3×3
        // Fill entire volume
        for v in mask.iter_mut() {
            *v = true;
        }
        let out = bw_largest_obj(&mask, 3, 3, 3);
        assert!(out.iter().all(|&v| v));
    }

    #[test]
    fn test_bw_largest_obj_two_components() {
        let mut mask = vec![false; 125]; // 5×5×5
        // Component A: voxels 0..8 (corner cube 2×2×2)
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    mask[z * 25 + y * 5 + x] = true;
                }
            }
        }
        // Component B: single isolated voxel at (4,4,4)
        mask[4 * 25 + 4 * 5 + 4] = true;
        let out = bw_largest_obj(&mask, 5, 5, 5);
        // Component A (8 voxels) should be kept, isolated voxel removed
        assert_eq!(out.iter().filter(|&&v| v).count(), 8);
        assert!(!out[4 * 25 + 4 * 5 + 4]);
    }

    #[test]
    fn test_fast_gauss_3d_shape() {
        let data = vec![1.0f64; 125]; // 5×5×5
        let out = fast_gauss_3d(&data, 5, 5, 5, 1.0, 1.5, true);
        assert_eq!(out.len(), 125);
    }

    #[test]
    fn test_fast_gauss_3d_uniform() {
        // Filtering a uniform field should return ≈same values
        let data = vec![3.0f64; 64]; // 4×4×4
        let out = fast_gauss_3d(&data, 4, 4, 4, 1.0, 1.0, true);
        for v in &out {
            assert!((v - 3.0).abs() < 1e-9, "uniform: {v}");
        }
    }
}
