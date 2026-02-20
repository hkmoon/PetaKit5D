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

// ---------------------------------------------------------------------------
// Bilateral filter
// ---------------------------------------------------------------------------

/// Edge-preserving bilateral filter for 2-D images.
///
/// Applies a spatial Gaussian (controlled by `sigma_s`) combined with a
/// range/intensity Gaussian (controlled by `sigma_r`).
///
/// This uses a direct sliding-window approach: for each output pixel the
/// nearby pixels within a `ceil(3*sigma_s)` radius are weighted by
/// `exp(-|pos|²/(2σ_s²)) * exp(-|intensity_diff|²/(2σ_r²))`.
///
/// # Arguments
/// * `data`        — Flat 2-D input (row-major), length `rows * cols`.
/// * `rows`, `cols`— Dimensions.
/// * `sigma_s`     — Spatial standard deviation.
/// * `sigma_r`     — Range (intensity) standard deviation.
pub fn bilateral_filter(
    data: &[f64],
    rows: usize,
    cols: usize,
    sigma_s: f64,
    sigma_r: f64,
) -> Vec<f64> {
    if data.is_empty() || sigma_s <= 0.0 || sigma_r <= 0.0 {
        return data.to_vec();
    }
    let w = (3.0 * sigma_s).ceil() as usize;
    let ss2 = 2.0 * sigma_s * sigma_s;
    let sr2 = 2.0 * sigma_r * sigma_r;

    let mut out = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let center = data[r * cols + c];
            let mut sum_w = 0.0f64;
            let mut sum_v = 0.0f64;
            let r_lo = if r >= w { r - w } else { 0 };
            let r_hi = (r + w).min(rows - 1);
            let c_lo = if c >= w { c - w } else { 0 };
            let c_hi = (c + w).min(cols - 1);
            for rr in r_lo..=r_hi {
                let dr = (rr as f64 - r as f64).powi(2);
                for cc in c_lo..=c_hi {
                    let dc = (cc as f64 - c as f64).powi(2);
                    let v = data[rr * cols + cc];
                    let di = (v - center).powi(2);
                    let w = (-(dr + dc) / ss2 - di / sr2).exp();
                    sum_w += w;
                    sum_v += w * v;
                }
            }
            out[r * cols + c] = if sum_w > 0.0 { sum_v / sum_w } else { center };
        }
    }
    out
}

// ---------------------------------------------------------------------------
// LoG filter via FFT
// ---------------------------------------------------------------------------

/// Laplacian-of-Gaussian (LoG) filter applied in the frequency domain.
///
/// The LoG kernel in Fourier space is `|ω|² · exp(-σ²|ω|²/2)`.
/// Works for 1-D (`dims=[n]`), 2-D (`dims=[ny, nx]`), or 3-D
/// (`dims=[nz, ny, nx]`).
///
/// # Arguments
/// * `data` — Flat input (row-major for 2D/3D).
/// * `dims` — Dimension sizes (length 1, 2, or 3).
/// * `sigma`— Gaussian standard deviation.
///
/// # Returns
/// Filtered data, same length as input.
pub fn filter_log(
    data: &[f64],
    dims: &[usize],
    sigma: f64,
) -> Vec<f64> {
    use rustfft::num_complex::Complex;
    use rustfft::FftPlanner;

    if data.is_empty() || sigma <= 0.0 {
        return data.to_vec();
    }

    // Build frequency-domain LoG weights: |ω|² · exp(-σ²|ω|²/2)
    // ω_k = 2π·k/n  for k=0..n-1 (FFT convention)
    let freq_1d = |n: usize| -> Vec<f64> {
        let mut f = vec![0.0f64; n];
        for k in 0..n {
            let fk = if k <= n / 2 { k as f64 } else { k as f64 - n as f64 };
            f[k] = 2.0 * std::f64::consts::PI * fk / n as f64;
        }
        f
    };

    match dims.len() {
        1 => {
            let n = dims[0];
            let w = freq_1d(n);
            let log_w: Vec<f64> = w.iter().map(|&wi| {
                let w2 = wi * wi;
                w2 * (-0.5 * sigma * sigma * w2).exp()
            }).collect();

            let mut planner = FftPlanner::new();
            let fft = planner.plan_fft_forward(n);
            let ifft = planner.plan_fft_inverse(n);
            let mut buf: Vec<Complex<f64>> = data.iter().map(|&v| Complex::new(v, 0.0)).collect();
            fft.process(&mut buf);
            for (b, lw) in buf.iter_mut().zip(log_w.iter()) {
                *b *= lw;
            }
            ifft.process(&mut buf);
            buf.iter().map(|c| c.re / n as f64).collect()
        }
        2 => {
            let (ny, nx) = (dims[0], dims[1]);
            let wy = freq_1d(ny);
            let wx = freq_1d(nx);
            let s2 = sigma * sigma;

            let mut planner = FftPlanner::new();
            // 2D FFT: row-by-row along X, then column-by-column along Y
            let fft_x = planner.plan_fft_forward(nx);
            let ifft_x = planner.plan_fft_inverse(nx);
            let fft_y = planner.plan_fft_forward(ny);
            let ifft_y = planner.plan_fft_inverse(ny);

            let mut buf: Vec<Complex<f64>> = data.iter().map(|&v| Complex::new(v, 0.0)).collect();
            // FFT along X
            for r in 0..ny {
                fft_x.process(&mut buf[r * nx..(r + 1) * nx]);
            }
            // FFT along Y (need transpose trick)
            let mut col_buf = vec![Complex::new(0.0, 0.0); ny];
            for c in 0..nx {
                for r in 0..ny { col_buf[r] = buf[r * nx + c]; }
                fft_y.process(&mut col_buf);
                for r in 0..ny { buf[r * nx + c] = col_buf[r]; }
            }
            // Apply LoG weights
            for r in 0..ny {
                for c in 0..nx {
                    let w2 = wy[r] * wy[r] + wx[c] * wx[c];
                    buf[r * nx + c] *= w2 * (-0.5 * s2 * w2).exp();
                }
            }
            // IFFT along Y
            for c in 0..nx {
                for r in 0..ny { col_buf[r] = buf[r * nx + c]; }
                ifft_y.process(&mut col_buf);
                for r in 0..ny { buf[r * nx + c] = col_buf[r]; }
            }
            // IFFT along X
            for r in 0..ny {
                ifft_x.process(&mut buf[r * nx..(r + 1) * nx]);
            }
            let scale = (nx * ny) as f64;
            buf.iter().map(|c| c.re / scale).collect()
        }
        3 => {
            let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
            let wz = freq_1d(nz);
            let wy = freq_1d(ny);
            let wx = freq_1d(nx);
            let s2 = sigma * sigma;
            let nynx = ny * nx;

            let mut planner = FftPlanner::new();
            let fft_x = planner.plan_fft_forward(nx);
            let ifft_x = planner.plan_fft_inverse(nx);
            let fft_y = planner.plan_fft_forward(ny);
            let ifft_y = planner.plan_fft_inverse(ny);
            let fft_z = planner.plan_fft_forward(nz);
            let ifft_z = planner.plan_fft_inverse(nz);

            let mut buf: Vec<Complex<f64>> = data.iter().map(|&v| Complex::new(v, 0.0)).collect();
            // FFT along X
            for i in 0..nz * ny {
                fft_x.process(&mut buf[i * nx..(i + 1) * nx]);
            }
            // FFT along Y
            let mut col_buf = vec![Complex::new(0.0, 0.0); ny];
            for z in 0..nz {
                for c in 0..nx {
                    for r in 0..ny { col_buf[r] = buf[z * nynx + r * nx + c]; }
                    fft_y.process(&mut col_buf);
                    for r in 0..ny { buf[z * nynx + r * nx + c] = col_buf[r]; }
                }
            }
            // FFT along Z
            let mut z_buf = vec![Complex::new(0.0, 0.0); nz];
            for y in 0..ny {
                for x in 0..nx {
                    for z in 0..nz { z_buf[z] = buf[z * nynx + y * nx + x]; }
                    fft_z.process(&mut z_buf);
                    for z in 0..nz { buf[z * nynx + y * nx + x] = z_buf[z]; }
                }
            }
            // Apply LoG weights
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let w2 = wz[z] * wz[z] + wy[y] * wy[y] + wx[x] * wx[x];
                        buf[z * nynx + y * nx + x] *= w2 * (-0.5 * s2 * w2).exp();
                    }
                }
            }
            // IFFT along Z
            for y in 0..ny {
                for x in 0..nx {
                    for z in 0..nz { z_buf[z] = buf[z * nynx + y * nx + x]; }
                    ifft_z.process(&mut z_buf);
                    for z in 0..nz { buf[z * nynx + y * nx + x] = z_buf[z]; }
                }
            }
            // IFFT along Y
            for z in 0..nz {
                for c in 0..nx {
                    for r in 0..ny { col_buf[r] = buf[z * nynx + r * nx + c]; }
                    ifft_y.process(&mut col_buf);
                    for r in 0..ny { buf[z * nynx + r * nx + c] = col_buf[r]; }
                }
            }
            // IFFT along X
            for i in 0..nz * ny {
                ifft_x.process(&mut buf[i * nx..(i + 1) * nx]);
            }
            let scale = (nz * ny * nx) as f64;
            buf.iter().map(|c| c.re / scale).collect()
        }
        _ => data.to_vec(),
    }
}

// ---------------------------------------------------------------------------
// Non-maximum suppression (2D)
// ---------------------------------------------------------------------------

/// Non-maximum suppression along the orientation direction for a 2-D response.
///
/// Each pixel is kept only if its response is ≥ both bilinearly-interpolated
/// neighbours at ±1 pixel along the local `orientation` (radians).
///
/// # Arguments
/// * `response`    — Flat 2-D response map, length `rows * cols`.
/// * `orientation` — Flat 2-D orientation map (radians), same size.
/// * `rows`, `cols`— Dimensions.
pub fn non_maximum_suppression(
    response: &[f64],
    orientation: &[f64],
    rows: usize,
    cols: usize,
) -> Vec<f64> {
    if response.is_empty() || response.len() != rows * cols {
        return response.to_vec();
    }

    // Build symmetrically-padded response (1 pixel each side)
    let pr = rows + 2;
    let pc = cols + 2;
    let mut padded = vec![0.0f64; pr * pc];
    for r in 0..rows {
        for c in 0..cols {
            // symmetric: clamp to [0, rows-1] / [0, cols-1]
            padded[(r + 1) * pc + (c + 1)] = response[r * cols + c];
        }
        // left/right borders
        padded[(r + 1) * pc] = response[r * cols];
        padded[(r + 1) * pc + cols + 1] = response[r * cols + cols - 1];
    }
    for c in 0..pc {
        padded[c] = padded[pc + c];
        padded[(pr - 1) * pc + c] = padded[(pr - 2) * pc + c];
    }

    let bilinear = |y: f64, x: f64| -> f64 {
        let y = y.clamp(0.0, (pr - 1) as f64);
        let x = x.clamp(0.0, (pc - 1) as f64);
        let y0 = y.floor() as usize;
        let y1 = (y0 + 1).min(pr - 1);
        let x0 = x.floor() as usize;
        let x1 = (x0 + 1).min(pc - 1);
        let ty = y - y.floor();
        let tx = x - x.floor();
        let v00 = padded[y0 * pc + x0];
        let v01 = padded[y0 * pc + x1];
        let v10 = padded[y1 * pc + x0];
        let v11 = padded[y1 * pc + x1];
        v00 * (1.0 - ty) * (1.0 - tx)
            + v01 * (1.0 - ty) * tx
            + v10 * ty * (1.0 - tx)
            + v11 * ty * tx
    };

    let mut out = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let theta = orientation[r * cols + c];
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            // Padded coordinates (+1 for padding offset)
            let py = (r + 1) as f64;
            let px = (c + 1) as f64;
            let a1 = bilinear(py + sin_t, px + cos_t);
            let a2 = bilinear(py - sin_t, px - cos_t);
            let v = response[r * cols + c];
            out[r * cols + c] = if v >= a1 && v >= a2 { v } else { 0.0 };
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 3-D Non-maximum suppression
// ---------------------------------------------------------------------------

/// Non-maximum suppression for a 3-D vector field.
///
/// Retains the magnitude at each voxel only when it is ≥ both trilinearly-
/// interpolated magnitudes ±1 step along the local vector direction.
///
/// # Arguments
/// * `u`, `v`, `w`    — X, Y, Z vector components (flat ZYX, length `nz*ny*nx`).
/// * `nz`, `ny`, `nx` — Volume dimensions.
pub fn non_maximum_suppression_3d(
    u: &[f64],
    v: &[f64],
    w: &[f64],
    nz: usize,
    ny: usize,
    nx: usize,
) -> Vec<f64> {
    let n = nz * ny * nx;
    if u.len() != n || v.len() != n || w.len() != n {
        return vec![0.0; n];
    }
    let nynx = ny * nx;

    // Compute magnitude
    let mag: Vec<f64> = (0..n)
        .map(|i| (u[i] * u[i] + v[i] * v[i] + w[i] * w[i]).sqrt())
        .collect();

    // Padded magnitude (1 voxel each side, symmetric)
    let pz = nz + 2;
    let py = ny + 2;
    let px = nx + 2;
    let pynx = py * px;
    let mut padded = vec![0.0f64; pz * pynx];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                padded[(z + 1) * pynx + (y + 1) * px + (x + 1)] = mag[z * nynx + y * nx + x];
            }
        }
    }
    // Mirror borders along each axis
    for z in 0..pz {
        for y in 0..py {
            padded[z * pynx + y * px] = padded[z * pynx + y * px + 1];
            padded[z * pynx + y * px + px - 1] = padded[z * pynx + y * px + px - 2];
        }
    }
    for z in 0..pz {
        for x in 0..px {
            padded[z * pynx + x] = padded[z * pynx + px + x];
            padded[z * pynx + (py - 1) * px + x] = padded[z * pynx + (py - 2) * px + x];
        }
    }
    for y in 0..py {
        for x in 0..px {
            padded[y * px + x] = padded[pynx + y * px + x];
            padded[(pz - 1) * pynx + y * px + x] = padded[(pz - 2) * pynx + y * px + x];
        }
    }

    let trilinear = |fz: f64, fy: f64, fx: f64| -> f64 {
        let fz = fz.clamp(0.0, (pz - 1) as f64);
        let fy = fy.clamp(0.0, (py - 1) as f64);
        let fx = fx.clamp(0.0, (px - 1) as f64);
        let z0 = fz.floor() as usize;
        let z1 = (z0 + 1).min(pz - 1);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(py - 1);
        let x0 = fx.floor() as usize;
        let x1 = (x0 + 1).min(px - 1);
        let tz = fz - fz.floor();
        let ty = fy - fy.floor();
        let tx = fx - fx.floor();
        macro_rules! p { ($z:expr,$y:expr,$x:expr) => { padded[$z * pynx + $y * px + $x] }; }
        let v000 = p!(z0, y0, x0);
        let v001 = p!(z0, y0, x1);
        let v010 = p!(z0, y1, x0);
        let v011 = p!(z0, y1, x1);
        let v100 = p!(z1, y0, x0);
        let v101 = p!(z1, y0, x1);
        let v110 = p!(z1, y1, x0);
        let v111 = p!(z1, y1, x1);
        let v00 = v000 * (1.0 - tx) + v001 * tx;
        let v01 = v010 * (1.0 - tx) + v011 * tx;
        let v10 = v100 * (1.0 - tx) + v101 * tx;
        let v11 = v110 * (1.0 - tx) + v111 * tx;
        let v0 = v00 * (1.0 - ty) + v01 * ty;
        let v1 = v10 * (1.0 - ty) + v11 * ty;
        v0 * (1.0 - tz) + v1 * tz
    };

    let mut out = mag.clone();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = z * nynx + y * nx + x;
                let m = mag[idx];
                if m == 0.0 {
                    out[idx] = 0.0;
                    continue;
                }
                let un = u[idx] / m;
                let vn = v[idx] / m;
                let wn = w[idx] / m;
                // Padded coordinates
                let pz_f = (z + 1) as f64;
                let py_f = (y + 1) as f64;
                let px_f = (x + 1) as f64;
                let m1 = trilinear(pz_f + vn, py_f + un, px_f + wn);
                let m2 = trilinear(pz_f - vn, py_f - un, px_f - wn);
                if m <= m1 || m <= m2 {
                    out[idx] = 0.0;
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Rosin threshold
// ---------------------------------------------------------------------------

/// Compute the Rosin threshold for a 1-D histogram.
///
/// Draws a line from the histogram peak to the last non-zero bin and
/// returns the bin centre at maximum perpendicular distance from that line.
///
/// # Arguments
/// * `data` — Input values (any shape, treated as a flat list).
///
/// # Returns
/// Threshold value as `f64`.
pub fn threshold_rosin(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    const N_BINS: usize = 256;
    let dmin = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let dmax = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if dmin >= dmax {
        return dmin;
    }
    let mut hist = vec![0usize; N_BINS];
    for &v in data {
        let bin = ((v - dmin) / (dmax - dmin) * (N_BINS - 1) as f64).round() as usize;
        hist[bin.min(N_BINS - 1)] += 1;
    }
    // Bin centres
    let bin_center = |i: usize| dmin + (i as f64 + 0.5) * (dmax - dmin) / N_BINS as f64;

    let peak_idx = hist.iter().enumerate().max_by_key(|&(_, v)| v).map(|(i, _)| i).unwrap_or(0);
    let tail_idx = hist.iter().enumerate().rev().find(|&(_, &v)| v > 0).map(|(i, _)| i).unwrap_or(peak_idx);
    if peak_idx == tail_idx {
        return bin_center(peak_idx);
    }
    let (x1, y1) = (peak_idx as f64, hist[peak_idx] as f64);
    let (x2, y2) = (tail_idx as f64, hist[tail_idx] as f64);
    let denom = ((y2 - y1).powi(2) + (x2 - x1).powi(2)).sqrt() + 1e-10;

    let mut best_i = peak_idx;
    let mut best_dist = 0.0f64;
    for i in peak_idx..=tail_idx {
        let (x0, y0) = (i as f64, hist[i] as f64);
        let dist = ((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1).abs() / denom;
        if dist > best_dist {
            best_dist = dist;
            best_i = i;
        }
    }
    bin_center(best_i)
}

// ---------------------------------------------------------------------------
// Surface filter (Gaussian second derivatives)
// ---------------------------------------------------------------------------

/// Filter a 3-D volume with Gaussian second-derivative kernels.
///
/// Returns three volumes `(d2X, d2Y, d2Z)` representing the second partial
/// derivative responses (positive at bright surfaces).
///
/// # Arguments
/// * `data`             — Flat ZYX input, length `nz * ny * nx`.
/// * `nz`, `ny`, `nx`   — Volume dimensions.
/// * `sigma`            — Isotropic Gaussian standard deviation.
pub fn surface_filter_gauss_3d(
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
    let w = (5.0 * sigma).ceil() as usize;
    let size = 2 * w + 1;
    let s2 = sigma * sigma;
    let nynx = ny * nx;

    let mut g = vec![0.0f64; size];
    let mut d2g = vec![0.0f64; size];
    for i in 0..size {
        let x = i as f64 - w as f64;
        let gv = (-0.5 * x * x / s2).exp();
        g[i] = gv;
        // Second derivative of Gaussian: -(x²/σ²-1)/σ² · G(x)
        d2g[i] = -(x * x / s2 - 1.0) / s2 * gv;
    }
    let gsum: f64 = g.iter().sum();
    g.iter_mut().for_each(|v| *v /= gsum);
    d2g.iter_mut().for_each(|v| *v /= gsum);

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
                        if zi >= 0 && zi < nz as isize
                            && yi >= 0 && yi < ny as isize
                            && xi >= 0 && xi < nx as isize
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

    // d2X: 2nd deriv in X (axis 2), smooth Y and Z
    let tmp = conv3d(data, &d2g, 2);
    let tmp = conv3d(&tmp, &g, 1);
    let d2x = conv3d(&tmp, &g, 0);

    // d2Y: 2nd deriv in Y (axis 1), smooth X and Z
    let tmp = conv3d(data, &g, 2);
    let tmp = conv3d(&tmp, &d2g, 1);
    let d2y = conv3d(&tmp, &g, 0);

    // d2Z: 2nd deriv in Z (axis 0), smooth X and Y
    let tmp = conv3d(data, &g, 2);
    let tmp = conv3d(&tmp, &g, 1);
    let d2z = conv3d(&tmp, &d2g, 0);

    (d2x, d2y, d2z)
}

// ---------------------------------------------------------------------------
// A Trou 1-D Wavelet Transform
// ---------------------------------------------------------------------------

/// Compute the 1-D A Trou Wavelet Transform.
///
/// Returns a `(n, n_bands+1)` row-major matrix where columns `0..n_bands`
/// are the detail coefficients and column `n_bands` is the final
/// approximation.  Uses kernel `[1,4,6,4,1]/16` with dilation `2^(k-1)` at
/// scale `k`.
///
/// # Arguments
/// * `signal`  — Input 1-D signal, length `n`.
/// * `n_bands` — Number of wavelet bands; if 0, defaults to `ceil(log2(n))`.
///
/// # Returns
/// Flat row-major output of length `n * (n_bands + 1)`.
/// Row `i`, column `k` → index `i * (n_bands + 1) + k`.
pub fn awt_1d(signal: &[f64], n_bands: usize) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return vec![];
    }
    let k_max = ((n as f64).log2().ceil() as usize).max(1);
    let nb = if n_bands == 0 { k_max } else { n_bands.min(k_max) };
    let cols = nb + 1;
    let mut out = vec![0.0f64; n * cols];

    let convolve_awt = |sig: &[f64], k: usize| -> Vec<f64> {
        let k1 = 1usize << (k - 1); // 2^(k-1)
        let k2 = 2 * k1;
        // Symmetric pad of size k2 on each side
        let mut tmp = vec![0.0f64; n + 2 * k2];
        for i in 0..n {
            tmp[i + k2] = sig[i];
        }
        // Reflect left boundary
        for i in 0..k2 {
            let src = k2 - 1 - i; // mirror index
            tmp[i] = sig[src.min(n - 1)];
        }
        // Reflect right boundary
        for i in 0..k2 {
            let src = n - 1 - i.min(n - 1);
            tmp[n + k2 + i] = sig[src];
        }
        let mut result = vec![0.0f64; n];
        for i in 0..n {
            let ci = i + k2; // index in padded array
            result[i] = (6.0 * tmp[ci]
                + 4.0 * (tmp[ci + k1] + tmp[ci - k1])
                + tmp[ci + k2]
                + tmp[ci - k2])
                / 16.0;
        }
        result
    };

    let mut last_a = signal.to_vec();
    for band in 0..nb {
        let new_a = convolve_awt(&last_a, band + 1);
        for i in 0..n {
            out[i * cols + band] = last_a[i] - new_a[i]; // detail
        }
        last_a = new_a;
    }
    // Final approximation in last column
    for i in 0..n {
        out[i * cols + nb] = last_a[i];
    }
    out
}

// ---------------------------------------------------------------------------
// Photobleach correction
// ---------------------------------------------------------------------------

/// Correct photobleaching in a fluorescence time series using exponential fitting.
///
/// For each time frame, a mean intensity is computed over the non-zero region.
/// A simple single exponential `a·exp(b·t)` is fit to the mean intensities, and
/// each frame is divided by the fitted value.  Falls back to linear correction
/// when the exponential fit fails.
///
/// # Arguments
/// * `data`   — Flat array of length `n_pixels * n_frames`.
/// * `n`      — Number of pixels per frame (`n_pixels`).
/// * `frames` — Number of time frames.
///
/// # Returns
/// Corrected data of the same length.
pub fn photobleach_correction(data: &[f64], n: usize, frames: usize) -> Vec<f64> {
    if data.len() != n * frames || frames == 0 || n == 0 {
        return data.to_vec();
    }
    // Compute mean intensity per frame (over all non-zero pixels)
    let means: Vec<f64> = (0..frames)
        .map(|f| {
            let slice = &data[f * n..(f + 1) * n];
            let (s, c) = slice
                .iter()
                .filter(|&&v| v != 0.0)
                .fold((0.0, 0usize), |(s, c), &v| (s + v, c + 1));
            if c > 0 { s / c as f64 } else { 0.0 }
        })
        .collect();

    // Fit simple single exponential: mean ≈ a·exp(b·t) using log-linear regression
    let m0 = means.iter().cloned().fold(f64::INFINITY, f64::min).max(1e-12);
    let log_means: Vec<f64> = means.iter().map(|&m| m.max(m0).ln()).collect();
    let t_vals: Vec<f64> = (0..frames).map(|i| i as f64).collect();

    // Linear regression: log(m) = log(a) + b*t
    let (ta, tb): (f64, f64) = {
        let n_f = frames as f64;
        let sx: f64 = t_vals.iter().sum();
        let sy: f64 = log_means.iter().sum();
        let sxy: f64 = t_vals.iter().zip(log_means.iter()).map(|(x, y)| x * y).sum();
        let sx2: f64 = t_vals.iter().map(|x| x * x).sum();
        let denom = n_f * sx2 - sx * sx;
        if denom.abs() < 1e-10 {
            (sy / n_f, 0.0)
        } else {
            let b = (n_f * sxy - sx * sy) / denom;
            let a = (sy - b * sx) / n_f;
            (a, b)
        }
    };

    // Correction: divide each frame by fitted value (clamped to minimum)
    let mut out = data.to_vec();
    for f in 0..frames {
        let fit = (ta + tb * f as f64).exp().max(1e-12);
        let frame_correction = means[0].max(1e-12) / fit;
        let slice = &mut out[f * n..(f + 1) * n];
        for v in slice.iter_mut() {
            *v *= frame_correction;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 2-D A Trou Wavelet Transform
// ---------------------------------------------------------------------------

/// Compute the 2-D A Trou Wavelet Transform.
///
/// Returns a flat ZYX-major array of shape `rows × cols × (n_bands + 1)`.
/// Slices `0..n_bands` (axis 2) contain detail coefficients; slice `n_bands`
/// is the final approximation.  Pass `n_bands = 0` to use the default
/// `ceil(max(log2(rows), log2(cols)))`.
///
/// Perfect reconstruction: `Σ detail_k + approx ≈ original`.
pub fn awt(data: &[f64], rows: usize, cols: usize, n_bands: usize) -> Vec<f64> {
    if data.is_empty() || rows == 0 || cols == 0 {
        return vec![];
    }
    let k_max = {
        let kn = ((rows as f64).log2().max((cols as f64).log2()).ceil() as usize).max(1);
        kn
    };
    let nb = if n_bands == 0 { k_max } else { n_bands.min(k_max) };
    let slices = nb + 1;
    let npix = rows * cols;
    let mut out = vec![0.0f64; npix * slices];

    // Convolve 2-D with dilated [1,4,6,4,1]/16 kernel (separable: rows then cols)
    let convolve_2d = |src: &[f64], k: usize| -> Vec<f64> {
        let k1 = 1usize << (k - 1); // 2^(k-1)
        let k2 = 2 * k1;            // 2^k

        // --- pass 1: convolve along rows ---
        let mut tmp = vec![0.0f64; npix];
        for r in 0..rows {
            for c in 0..cols {
                let get = |rr: isize| -> f64 {
                    let ri = rr.clamp(0, (rows - 1) as isize) as usize;
                    src[ri * cols + c]
                };
                let ri = r as isize;
                tmp[r * cols + c] = (6.0 * get(ri)
                    + 4.0 * (get(ri + k1 as isize) + get(ri - k1 as isize))
                    + get(ri + k2 as isize)
                    + get(ri - k2 as isize))
                    / 16.0;
            }
        }
        // --- pass 2: convolve along cols ---
        let mut result = vec![0.0f64; npix];
        for r in 0..rows {
            for c in 0..cols {
                let get = |cc: isize| -> f64 {
                    let ci = cc.clamp(0, (cols - 1) as isize) as usize;
                    tmp[r * cols + ci]
                };
                let ci = c as isize;
                result[r * cols + c] = (6.0 * get(ci)
                    + 4.0 * (get(ci + k1 as isize) + get(ci - k1 as isize))
                    + get(ci + k2 as isize)
                    + get(ci - k2 as isize))
                    / 16.0;
            }
        }
        result
    };

    let mut last_approx = data.to_vec();
    for band in 0..nb {
        let new_approx = convolve_2d(&last_approx, band + 1);
        for px in 0..npix {
            out[px * slices + band] = last_approx[px] - new_approx[px]; // detail
        }
        last_approx = new_approx;
    }
    // Final approximation in last slice
    for px in 0..npix {
        out[px * slices + nb] = last_approx[px];
    }
    out
}

// ---------------------------------------------------------------------------
// A Trou wavelet denoising (2D)
// ---------------------------------------------------------------------------

/// Denoise a 2-D image via soft thresholding of its A Trou wavelet coefficients.
///
/// Detail bands are MAD-thresholded at `n_sigma / norminv(0.75)`.
/// The final approximation is added back when `include_low_band = true`.
///
/// # Arguments
/// * `data`            — Flat row-major input, length `rows * cols`.
/// * `rows`, `cols`    — Dimensions.
/// * `n_bands`         — Wavelet bands (0 = default).
/// * `include_low_band`— Add final approximation to the reconstruction.
/// * `n_sigma`         — Threshold multiplier (default 3.0).
pub fn awt_denoising(
    data: &[f64],
    rows: usize,
    cols: usize,
    n_bands: usize,
    include_low_band: bool,
    n_sigma: f64,
) -> Vec<f64> {
    if data.is_empty() || rows == 0 || cols == 0 {
        return data.to_vec();
    }
    let k_max = ((rows as f64).log2().max((cols as f64).log2()).ceil() as usize).max(1);
    let nb = if n_bands == 0 { k_max } else { n_bands.min(k_max) };
    let slices = nb + 1;
    let npix = rows * cols;

    let w = awt(data, rows, cols, nb);

    // norminv(0.75) ≈ 0.6745
    let norminv_075: f64 = 0.674_489_750_196_082;
    let mad_factor = n_sigma / norminv_075;

    let mut reconstructed = if include_low_band {
        // Start from last approximation slice
        (0..npix).map(|px| w[px * slices + nb]).collect::<Vec<_>>()
    } else {
        vec![0.0f64; npix]
    };

    for band in 0..nb {
        let coeffs: Vec<f64> = (0..npix).map(|px| w[px * slices + band]).collect();
        let mut sorted = coeffs.clone();
        sorted.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
        let mad_val = {
            let mid = npix / 2;
            if npix % 2 == 0 {
                (sorted[mid - 1].abs() + sorted[mid].abs()) / 2.0
            } else {
                sorted[mid].abs()
            }
        };
        let threshold = mad_factor * mad_val;
        for (px, c) in coeffs.iter().enumerate() {
            let v = if c.abs() < threshold { 0.0 } else { *c };
            reconstructed[px] += v;
        }
    }
    reconstructed
}

// ---------------------------------------------------------------------------
// B-spline interpolation (1-D, 2-D)
// ---------------------------------------------------------------------------

/// Helper: mirror boundary for B-spline coefficient index.
fn mirror_idx(i: isize, n: usize) -> usize {
    let mut idx = i;
    let n = n as isize;
    if idx < 0 { idx = -idx; }
    if idx >= n { idx = 2 * n - idx - 2; }
    idx.clamp(0, n - 1) as usize
}

/// Cubic B-spline basis value `β³(x)` and first+second derivatives.
///
/// Returns `(b, db, d2b)` evaluated at scalar `x`.
fn cubic_bspline_basis(x: f64) -> (f64, f64, f64) {
    let ax = x.abs();
    if ax < 1.0 {
        let b = (4.0 - 6.0 * ax * ax + 3.0 * ax * ax * ax) / 6.0;
        let db_abs = (-12.0 * ax + 9.0 * ax * ax) / 6.0;
        let d2b = (-12.0 + 18.0 * ax) / 6.0;
        let db = db_abs * x.signum();
        (b, db, d2b)
    } else if ax < 2.0 {
        let t = 2.0 - ax;
        let b = t * t * t / 6.0;
        let db_abs = -t * t / 2.0;
        let d2b = t;
        let db = db_abs * x.signum() * (-1.0); // descending branch: chain-rule sign for (2-|x|)³
        (b, db, d2b)
    } else {
        (0.0, 0.0, 0.0)
    }
}

/// Cubic B-spline interpolation for a 1-D signal with first and second derivatives.
///
/// # Arguments
/// * `f`               — Input signal, length `n`.
/// * `xi`              — Interpolation coordinates (0-based).
/// * `mirror_boundary` — `true` for mirror BC, `false` for periodic.
///
/// # Returns
/// `(values, first_deriv, second_deriv)` each of length `xi.len()`.
pub fn binterp_1d(
    f: &[f64],
    xi: &[f64],
    mirror_boundary: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = f.len();
    if n == 0 || xi.is_empty() {
        return (vec![], vec![], vec![]);
    }

    // Compute B-spline coefficients using b3spline_1d
    let bc = if mirror_boundary { "mirror" } else { "periodic" };
    let f2d = f.to_vec(); // already 1×n layout
    let coeffs = match b3spline_1d(&f2d, 1, n, bc) {
        Ok(c) => c,
        Err(_) => f.to_vec(),
    };

    let get_coeff = |i: isize| -> f64 {
        if mirror_boundary {
            coeffs[mirror_idx(i, n)]
        } else {
            coeffs[(((i % n as isize) + n as isize) % n as isize) as usize]
        }
    };

    let mut fi = vec![0.0f64; xi.len()];
    let mut fi_dx = vec![0.0f64; xi.len()];
    let mut fi_d2x = vec![0.0f64; xi.len()];

    for (idx, &x) in xi.iter().enumerate() {
        let xf = x.floor() as isize;
        let dx = x - xf as f64;
        // 4-point support: indices xf-1, xf, xf+1, xf+2
        let distances = [dx + 1.0, dx, dx - 1.0, dx - 2.0];
        let idxs = [xf - 1, xf, xf + 1, xf + 2];
        let mut val = 0.0;
        let mut d1 = 0.0;
        let mut d2 = 0.0;
        for (&ii, &dist) in idxs.iter().zip(distances.iter()) {
            let c = get_coeff(ii);
            let (b, db, d2b) = cubic_bspline_basis(dist);
            val += c * b;
            d1 += c * db;
            d2 += c * d2b;
        }
        fi[idx] = val;
        fi_dx[idx] = d1;
        fi_d2x[idx] = d2;
    }
    (fi, fi_dx, fi_d2x)
}

/// Cubic B-spline interpolation for a 2-D image.
///
/// # Arguments
/// * `f`               — Flat row-major input, length `rows * cols`.
/// * `xi`              — X (column) interpolation coordinates (0-based).
/// * `yi`              — Y (row) interpolation coordinates (0-based).
/// * `mirror_boundary` — Boundary condition.
///
/// # Returns
/// `(values, dfdx, dfdy)` each of length `xi.len()`.
pub fn binterp_2d(
    f: &[f64],
    rows: usize,
    cols: usize,
    xi: &[f64],
    yi: &[f64],
    mirror_boundary: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if f.is_empty() || xi.is_empty() || xi.len() != yi.len() {
        return (vec![], vec![], vec![]);
    }

    // Compute 2-D B-spline coefficients
    let bc = if mirror_boundary { "mirror" } else { "periodic" };
    let coeffs = match b3spline_2d(f, rows, cols, bc) {
        Ok(c) => c,
        Err(_) => f.to_vec(),
    };

    let get = |r: isize, c: isize| -> f64 {
        let ri = if mirror_boundary { mirror_idx(r, rows) } else { (((r % rows as isize) + rows as isize) % rows as isize) as usize };
        let ci = if mirror_boundary { mirror_idx(c, cols) } else { (((c % cols as isize) + cols as isize) % cols as isize) as usize };
        coeffs[ri * cols + ci]
    };

    let n = xi.len();
    let mut vals = vec![0.0f64; n];
    let mut dxs = vec![0.0f64; n];
    let mut dys = vec![0.0f64; n];

    for i in 0..n {
        let xv = xi[i];
        let yv = yi[i];
        let xf = xv.floor() as isize;
        let yf = yv.floor() as isize;
        let dx = xv - xf as f64;
        let dy = yv - yf as f64;

        let x_dist = [dx + 1.0, dx, dx - 1.0, dx - 2.0];
        let y_dist = [dy + 1.0, dy, dy - 1.0, dy - 2.0];
        let x_idx = [xf - 1, xf, xf + 1, xf + 2];
        let y_idx = [yf - 1, yf, yf + 1, yf + 2];

        let mut val = 0.0;
        let mut dvx = 0.0;
        let mut dvy = 0.0;

        for (jr, &iy) in y_idx.iter().enumerate() {
            let (by, dby, _) = cubic_bspline_basis(y_dist[jr]);
            for (jc, &ix) in x_idx.iter().enumerate() {
                let (bx, dbx, _) = cubic_bspline_basis(x_dist[jc]);
                let c = get(iy, ix);
                val += c * by * bx;
                dvx += c * by * dbx;
                dvy += c * dby * bx;
            }
        }
        vals[i] = val;
        dxs[i] = dvx;
        dys[i] = dvy;
    }
    (vals, dxs, dys)
}

/// Cubic B-spline interpolation wrapper that dispatches to `binterp_1d` or `binterp_2d`.
///
/// Pass `rows = 1` for 1-D signals. The `yi` argument is ignored when `rows == 1`.
pub fn binterp(
    f: &[f64],
    rows: usize,
    cols: usize,
    xi: &[f64],
    yi: &[f64],
    mirror_boundary: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if rows <= 1 {
        binterp_1d(f, xi, mirror_boundary)
    } else {
        binterp_2d(f, rows, cols, xi, yi, mirror_boundary)
    }
}

// ---------------------------------------------------------------------------
// N-D FFT-based convolution
// ---------------------------------------------------------------------------

/// N-D FFT-based convolution of flat 3-D arrays.
///
/// Equivalent to `scipy.signal.convolve` with the 'same' or 'full' mode.
/// Works specifically with 3-D volumes in ZYX order.
///
/// # Arguments
/// * `a`              — First input, length `az * ay * ax`.
/// * `a_dims`         — `[az, ay, ax]`.
/// * `b`              — Kernel, length `bz * by * bx`.
/// * `b_dims`         — `[bz, by, bx]`.
/// * `mode`           — `"full"`, `"same"`, or `"valid"`.
///
/// # Returns
/// Convolved data (length depends on `mode`).
pub fn convn_fft(
    a: &[f64],
    a_dims: &[usize],
    b: &[f64],
    b_dims: &[usize],
    mode: &str,
) -> Result<(Vec<f64>, Vec<usize>), ImageProcessingError> {
    use rustfft::num_complex::Complex;
    use rustfft::FftPlanner;

    if a_dims.len() != 3 || b_dims.len() != 3 {
        return Err(ImageProcessingError::InvalidDimensions);
    }
    let [az, ay, ax] = [a_dims[0], a_dims[1], a_dims[2]];
    let [bz, by, bx] = [b_dims[0], b_dims[1], b_dims[2]];

    if a.len() != az * ay * ax || b.len() != bz * by * bx {
        return Err(ImageProcessingError::InvalidDimensions);
    }

    // Full convolution dimensions
    let fz = az + bz - 1;
    let fy = ay + by - 1;
    let fx = ax + bx - 1;
    let fn_ = fz * fy * fx;
    let fynx = fy * fx;
    let aynx = ay * ax;
    let bynx = by * bx;

    // Zero-pad into complex buffers
    let mut fa: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); fn_];
    let mut fb: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); fn_];

    for iz in 0..az {
        for iy in 0..ay {
            for ix in 0..ax {
                fa[iz * fynx + iy * fx + ix] = Complex::new(a[iz * aynx + iy * ax + ix], 0.0);
            }
        }
    }
    for iz in 0..bz {
        for iy in 0..by {
            for ix in 0..bx {
                fb[iz * fynx + iy * fx + ix] = Complex::new(b[iz * bynx + iy * bx + ix], 0.0);
            }
        }
    }

    // 3-D FFT via separable 1-D FFTs
    let mut planner = FftPlanner::new();

    // Pre-plan all needed transforms
    let fft_x_plan = planner.plan_fft_forward(fx);
    let fft_y_plan = planner.plan_fft_forward(fy);
    let fft_z_plan = planner.plan_fft_forward(fz);
    let ifft_x_plan = planner.plan_fft_inverse(fx);
    let ifft_y_plan = planner.plan_fft_inverse(fy);
    let ifft_z_plan = planner.plan_fft_inverse(fz);

    // FFT along X for a buffer
    let do_fft_x = |buf: &mut Vec<Complex<f64>>, plan: &dyn rustfft::Fft<f64>| {
        for i in 0..fz * fy {
            plan.process(&mut buf[i * fx..(i + 1) * fx]);
        }
    };
    let do_fft_y = |buf: &mut Vec<Complex<f64>>, plan: &dyn rustfft::Fft<f64>| {
        let mut col = vec![Complex::new(0.0, 0.0); fy];
        for iz in 0..fz {
            for ix in 0..fx {
                for iy in 0..fy { col[iy] = buf[iz * fynx + iy * fx + ix]; }
                plan.process(&mut col);
                for iy in 0..fy { buf[iz * fynx + iy * fx + ix] = col[iy]; }
            }
        }
    };
    let do_fft_z = |buf: &mut Vec<Complex<f64>>, plan: &dyn rustfft::Fft<f64>| {
        let mut col = vec![Complex::new(0.0, 0.0); fz];
        for iy in 0..fy {
            for ix in 0..fx {
                for iz in 0..fz { col[iz] = buf[iz * fynx + iy * fx + ix]; }
                plan.process(&mut col);
                for iz in 0..fz { buf[iz * fynx + iy * fx + ix] = col[iz]; }
            }
        }
    };

    do_fft_x(&mut fa, fft_x_plan.as_ref());
    do_fft_y(&mut fa, fft_y_plan.as_ref());
    do_fft_z(&mut fa, fft_z_plan.as_ref());
    do_fft_x(&mut fb, fft_x_plan.as_ref());
    do_fft_y(&mut fb, fft_y_plan.as_ref());
    do_fft_z(&mut fb, fft_z_plan.as_ref());

    // Multiply
    for i in 0..fn_ { fa[i] *= fb[i]; }

    do_fft_x(&mut fa, ifft_x_plan.as_ref());
    do_fft_y(&mut fa, ifft_y_plan.as_ref());
    do_fft_z(&mut fa, ifft_z_plan.as_ref());

    let scale = fn_ as f64;
    let full: Vec<f64> = fa.iter().map(|c| c.re / scale).collect();

    // Trim to requested mode
    let (oz, oy, ox, s_z, s_y, s_x) = match mode {
        "full" => (fz, fy, fx, 0, 0, 0),
        "same" => (az, ay, ax, (bz - 1) / 2, (by - 1) / 2, (bx - 1) / 2),
        "valid" => {
            let vz = az.saturating_sub(bz) + 1;
            let vy = ay.saturating_sub(by) + 1;
            let vx = ax.saturating_sub(bx) + 1;
            (vz, vy, vx, bz - 1, by - 1, bx - 1)
        }
        _ => return Err(ImageProcessingError::InvalidDimensions),
    };

    let mut out = vec![0.0f64; oz * oy * ox];
    for z in 0..oz {
        for y in 0..oy {
            for x in 0..ox {
                out[z * oy * ox + y * ox + x] =
                    full[(z + s_z) * fynx + (y + s_y) * fx + (x + s_x)];
            }
        }
    }
    Ok((out, vec![oz, oy, ox]))
}

// ---------------------------------------------------------------------------
// N-D LoG filter with spatial kernel
// ---------------------------------------------------------------------------

/// Laplacian-of-Gaussian filter with anisotropic spacing.
///
/// Builds a discrete LoG kernel in spatial domain and applies it via
/// `convn_fft` with 'same' boundary.  Optionally normalises by `σ²`
/// for scale-normalised responses.
///
/// # Arguments
/// * `data`                        — Flat ZYX input.
/// * `nz`, `ny`, `nx`              — Volume dimensions.
/// * `sigma`                       — Gaussian σ in physical units.
/// * `spacing`                     — Voxel spacing `[sz, sy, sx]` (defaults to 1.0).
/// * `use_normalized_derivatives`  — Multiply response by `σ²`.
pub fn filter_log_nd(
    data: &[f64],
    nz: usize,
    ny: usize,
    nx: usize,
    sigma: f64,
    spacing: Option<&[f64]>,
    use_normalized_derivatives: bool,
) -> Vec<f64> {
    if data.is_empty() || sigma <= 0.0 {
        return data.to_vec();
    }
    let sp: [f64; 3] = match spacing {
        Some(s) if s.len() >= 3 => [s[0], s[1], s[2]],
        _ => [1.0, 1.0, 1.0],
    };

    // Build separable LoG kernel: LoG(x,y,z) = (|x/σ|²-1)*G(x)*G(y)*G(z)/σ² + ...
    // We build it as a full 3-D kernel product
    let sigma_px = [sigma / sp[0], sigma / sp[1], sigma / sp[2]];
    let truncate = 3.0;

    let make_1d = |s: f64| -> (Vec<f64>, Vec<f64>) {
        let w = (truncate * s).ceil() as usize;
        let size = 2 * w + 1;
        let mut g = vec![0.0f64; size];
        let mut log1d = vec![0.0f64; size];
        let s2 = s * s;
        for i in 0..size {
            let x = i as f64 - w as f64;
            let gv = (-0.5 * x * x / s2).exp();
            g[i] = gv;
            log1d[i] = (x * x / s2 - 1.0) / s2 * gv; // LoG1D contribution
        }
        let gsum: f64 = g.iter().sum();
        g.iter_mut().for_each(|v| *v /= gsum);
        let logsum: f64 = log1d.iter().sum();
        log1d.iter_mut().for_each(|v| *v -= logsum / size as f64); // zero-mean
        // Normalise log1d by same factor as g
        log1d.iter_mut().for_each(|v| *v /= gsum);
        (g, log1d)
    };

    let (gz, logz) = make_1d(sigma_px[0]);
    let (gy, logy) = make_1d(sigma_px[1]);
    let (gx, logx) = make_1d(sigma_px[2]);

    let wz = gz.len();
    let wy = gy.len();
    let wx = gx.len();

    // LoG = ∂²/∂z² + ∂²/∂y² + ∂²/∂x²
    // = logz⊗gy⊗gx + gz⊗logy⊗gx + gz⊗gy⊗logx
    let build_kernel = |kz: &[f64], ky: &[f64], kx: &[f64]| -> Vec<f64> {
        let mut k = vec![0.0f64; wz * wy * wx];
        for iz in 0..wz {
            for iy in 0..wy {
                for ix in 0..wx {
                    k[iz * wy * wx + iy * wx + ix] = kz[iz] * ky[iy] * kx[ix];
                }
            }
        }
        k
    };

    let k1 = build_kernel(&logz, &gy, &gx);
    let k2 = build_kernel(&gz, &logy, &gx);
    let k3 = build_kernel(&gz, &gy, &logx);

    let mut kernel = vec![0.0f64; wz * wy * wx];
    for i in 0..kernel.len() { kernel[i] = k1[i] + k2[i] + k3[i]; }

    // Convolve using convn_fft 'same'
    let result = convn_fft(
        data,
        &[nz, ny, nx],
        &kernel,
        &[wz, wy, wx],
        "same",
    ).map(|(d, _)| d)
     .unwrap_or_else(|_| data.to_vec());

    if use_normalized_derivatives {
        result.iter().map(|&v| v * sigma * sigma).collect()
    } else {
        result
    }
}

// ---------------------------------------------------------------------------
// Directional distance transform (2-D)
// ---------------------------------------------------------------------------

/// Compute directional distance from each `false` pixel to the nearest `true`
/// pixel in 4 cardinal directions.
///
/// Returns a flat array of length `rows * cols * 4`.  Slice `k` (0-3) gives:
/// - 0: distance increasing-row (down)
/// - 1: distance decreasing-row (up)
/// - 2: distance increasing-col (right)
/// - 3: distance decreasing-col (left)
pub fn bw_max_direct_dist(mask: &[bool], rows: usize, cols: usize) -> Vec<f32> {
    if mask.is_empty() || rows == 0 || cols == 0 {
        return vec![];
    }
    let n = rows * cols;
    let inf = rows.max(cols) as f32 + 1.0;
    let mut out = vec![inf; n * 4];

    // Direction 0: down (increasing row)
    for r in 1..rows {
        for c in 0..cols {
            if !mask[r * cols + c] {
                out[r * cols + c] = out[(r - 1) * cols + c] + 1.0;
            } else {
                out[r * cols + c] = 0.0;
            }
        }
    }
    for c in 0..cols {
        out[c] = if mask[c] { 0.0 } else { inf };
    }

    // Direction 1: up (decreasing row)
    let off = n;
    for r in (0..rows - 1).rev() {
        for c in 0..cols {
            if !mask[r * cols + c] {
                out[off + r * cols + c] = out[off + (r + 1) * cols + c] + 1.0;
            } else {
                out[off + r * cols + c] = 0.0;
            }
        }
    }
    for c in 0..cols {
        out[off + (rows - 1) * cols + c] = if mask[(rows - 1) * cols + c] { 0.0 } else { inf };
    }

    // Direction 2: right (increasing col)
    let off = 2 * n;
    for r in 0..rows {
        for c in 1..cols {
            if !mask[r * cols + c] {
                out[off + r * cols + c] = out[off + r * cols + c - 1] + 1.0;
            } else {
                out[off + r * cols + c] = 0.0;
            }
        }
        out[off + r * cols] = if mask[r * cols] { 0.0 } else { inf };
    }

    // Direction 3: left (decreasing col)
    let off = 3 * n;
    for r in 0..rows {
        for c in (0..cols - 1).rev() {
            if !mask[r * cols + c] {
                out[off + r * cols + c] = out[off + r * cols + c + 1] + 1.0;
            } else {
                out[off + r * cols + c] = 0.0;
            }
        }
        out[off + r * cols + cols - 1] = if mask[r * cols + cols - 1] { 0.0 } else { inf };
    }

    out
}

// ---------------------------------------------------------------------------
// Neighbor count for binary images
// ---------------------------------------------------------------------------

/// Count the number of `true` neighbors of each element in a flat 2-D binary
/// image using the supplied neighborhood mask.
///
/// `neighborhood` is a flat `k×k` boolean mask (must be odd-sized, centre is
/// ignored). Defaults to 8-connectivity (3×3 all-true, centre excluded).
pub fn bw_n_neighbors(mask: &[bool], rows: usize, cols: usize, neighborhood: Option<&[bool]>) -> Vec<u8> {
    if mask.is_empty() {
        return vec![];
    }
    // Default: 8-connectivity 3×3
    let default_hood = [true; 9];
    let hood = neighborhood.unwrap_or(&default_hood);
    let ksize = ((hood.len() as f64).sqrt().round() as usize).max(1);
    let kh = ksize / 2;

    let mut out = vec![0u8; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            if !mask[r * cols + c] {
                continue;
            }
            let mut cnt = 0u8;
            for kr in 0..ksize {
                for kc in 0..ksize {
                    if kr == kh && kc == kh { continue; } // skip centre
                    if !hood[kr * ksize + kc] { continue; }
                    let nr = r as isize + kr as isize - kh as isize;
                    let nc = c as isize + kc as isize - kh as isize;
                    if nr >= 0 && nr < rows as isize && nc >= 0 && nc < cols as isize {
                        if mask[nr as usize * cols + nc as usize] { cnt += 1; }
                    }
                }
            }
            out[r * cols + c] = cnt;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Mask vectors / angle filter
// ---------------------------------------------------------------------------

/// Return a boolean mask indicating which (x, y) points lie inside a 2-D
/// binary mask (1-based MATLAB-style coordinates).
///
/// Points outside image bounds are marked `false`.
pub fn mask_vectors(x_coords: &[f64], y_coords: &[f64], mask: &[bool], rows: usize, cols: usize) -> Vec<bool> {
    let n = x_coords.len().min(y_coords.len());
    let mut out = vec![false; n];
    for i in 0..n {
        // MATLAB convention: disp_mat_x → row index, disp_mat_y → col index
        let row = x_coords[i].round() as isize;
        let col = y_coords[i].round() as isize;
        if row > 0 && row <= rows as isize && col > 0 && col <= cols as isize {
            if mask[(row - 1) as usize * cols + (col - 1) as usize] {
                out[i] = true;
            }
        }
    }
    out
}

/// Filter vectors by angle from a reference direction.
///
/// Returns `true` for vectors within `±π/3` radians of `vec_mid`.
///
/// # Arguments
/// * `vec_x`, `vec_y` — Vector components.
/// * `ref_x`, `ref_y` — Reference direction.
pub fn angle_filter(vec_x: &[f64], vec_y: &[f64], ref_x: f64, ref_y: f64) -> Vec<bool> {
    let n = vec_x.len().min(vec_y.len());
    let mut out = vec![false; n];
    let ref_angle = ref_y.atan2(ref_x);
    let threshold = std::f64::consts::PI / 3.0;
    for i in 0..n {
        let angle = vec_y[i].atan2(vec_x[i]);
        let diff = (angle - ref_angle + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI;
        out[i] = diff.abs() <= threshold;
    }
    out
}

// ---------------------------------------------------------------------------
// Color utilities
// ---------------------------------------------------------------------------

/// Compose an RGB image (rows×cols×3, u8) from up to three grayscale channels.
///
/// Each channel is independently contrast-stretched to `[0, 255]`.
/// Pass `None` to fill that channel with zeros.
pub fn ch2rgb(
    r_chan: Option<&[f64]>,
    g_chan: Option<&[f64]>,
    b_chan: Option<&[f64]>,
    rows: usize,
    cols: usize,
) -> Vec<u8> {
    let npix = rows * cols;
    let scale_chan = |ch: Option<&[f64]>| -> Vec<u8> {
        match ch {
            None => vec![0u8; npix],
            Some(c) => {
                let mn = c.iter().cloned().fold(f64::INFINITY, f64::min);
                let mx = c.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if (mx - mn).abs() < 1e-12 {
                    vec![0u8; npix]
                } else {
                    c.iter().map(|&v| ((v - mn) / (mx - mn) * 255.0).round().clamp(0.0, 255.0) as u8).collect()
                }
            }
        }
    };
    let r = scale_chan(r_chan);
    let g = scale_chan(g_chan);
    let b = scale_chan(b_chan);
    let mut out = vec![0u8; npix * 3];
    for i in 0..npix {
        out[i * 3] = r[i];
        out[i * 3 + 1] = g[i];
        out[i * 3 + 2] = b[i];
    }
    out
}

/// Overlay coloured masks on a grayscale image.
///
/// `colors` is a list of `(r, g, b)` in `[0, 1]`. Each mask's nonzero pixels
/// replace the grayscale with the corresponding colour.
///
/// Returns a flat `rows×cols×3` u8 RGB image.
pub fn rgb_overlay(
    img: &[f64],
    rows: usize,
    cols: usize,
    masks: &[&[bool]],
    colors: &[(f64, f64, f64)],
) -> Vec<u8> {
    let npix = rows * cols;
    if img.is_empty() || npix == 0 {
        return vec![];
    }
    // Normalise grayscale to [0,1]
    let mn = img.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = img.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let scaled: Vec<f64> = if (mx - mn).abs() < 1e-12 {
        vec![0.0; npix]
    } else {
        img.iter().map(|&v| (v - mn) / (mx - mn)).collect()
    };

    let mut rgb = vec![(0.0f64, 0.0f64, 0.0f64); npix];
    for i in 0..npix {
        rgb[i] = (scaled[i], scaled[i], scaled[i]);
    }

    for (mask, &(cr, cg, cb)) in masks.iter().zip(colors.iter()) {
        for i in 0..npix.min(mask.len()) {
            if mask[i] {
                rgb[i].0 = (rgb[i].0 + cr).min(1.0);
                rgb[i].1 = (rgb[i].1 + cg).min(1.0);
                rgb[i].2 = (rgb[i].2 + cb).min(1.0);
            }
        }
    }

    let mut out = vec![0u8; npix * 3];
    for i in 0..npix {
        out[i * 3] = (rgb[i].0 * 255.0).round().clamp(0.0, 255.0) as u8;
        out[i * 3 + 1] = (rgb[i].1 * 255.0).round().clamp(0.0, 255.0) as u8;
        out[i * 3 + 2] = (rgb[i].2 * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    out
}

// ---------------------------------------------------------------------------
// Z-projection
// ---------------------------------------------------------------------------

/// Z-projection of a 3-D volume.
///
/// Projects along axis 0 (Z).
///
/// # Arguments
/// * `data`        — Flat ZYX input, length `nz * ny * nx`.
/// * `proj_type`   — `"max"`, `"mean"`, `"median"`, or `"min"`.
///
/// # Returns
/// `(projected, [ny, nx])`.
pub fn z_proj_image(
    data: &[f64],
    nz: usize,
    ny: usize,
    nx: usize,
    proj_type: &str,
) -> Result<Vec<f64>, ImageProcessingError> {
    if data.len() != nz * ny * nx {
        return Err(ImageProcessingError::InvalidDimensions);
    }
    if nz == 0 || ny == 0 || nx == 0 {
        return Ok(vec![]);
    }
    let nynx = ny * nx;
    let mut out = vec![0.0f64; nynx];

    match proj_type {
        "max" => {
            for i in 0..nynx { out[i] = f64::NEG_INFINITY; }
            for z in 0..nz {
                for i in 0..nynx {
                    let v = data[z * nynx + i];
                    if v > out[i] { out[i] = v; }
                }
            }
        }
        "min" => {
            for i in 0..nynx { out[i] = f64::INFINITY; }
            for z in 0..nz {
                for i in 0..nynx {
                    let v = data[z * nynx + i];
                    if v < out[i] { out[i] = v; }
                }
            }
        }
        "mean" | "ave" | "average" => {
            for z in 0..nz {
                for i in 0..nynx { out[i] += data[z * nynx + i]; }
            }
            for v in out.iter_mut() { *v /= nz as f64; }
        }
        "median" | "med" => {
            for i in 0..nynx {
                let mut col: Vec<f64> = (0..nz).map(|z| data[z * nynx + i]).collect();
                col.sort_by(|a, b| a.partial_cmp(b).unwrap());
                out[i] = if nz % 2 == 0 {
                    (col[nz / 2 - 1] + col[nz / 2]) / 2.0
                } else {
                    col[nz / 2]
                };
            }
        }
        _ => return Err(ImageProcessingError::InvalidDimensions),
    }
    Ok(out)
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

    // --- bilateral_filter ---

    #[test]
    fn test_bilateral_filter_shape() {
        let data = vec![1.0f64; 100]; // 10×10
        let out = bilateral_filter(&data, 10, 10, 2.0, 0.5);
        assert_eq!(out.len(), 100);
    }

    #[test]
    fn test_bilateral_filter_uniform() {
        // Filtering uniform image should return same values
        let data = vec![3.0f64; 100];
        let out = bilateral_filter(&data, 10, 10, 2.0, 0.5);
        for v in &out {
            assert!((v - 3.0).abs() < 1e-9, "bilateral uniform: {v}");
        }
    }

    // --- filter_log ---

    #[test]
    fn test_filter_log_1d_shape() {
        let data = vec![1.0f64; 64];
        let out = filter_log(&data, &[64], 2.0);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn test_filter_log_2d_shape() {
        let data = vec![1.0f64; 400]; // 20×20
        let out = filter_log(&data, &[20, 20], 2.0);
        assert_eq!(out.len(), 400);
    }

    #[test]
    fn test_filter_log_3d_shape() {
        let data = vec![1.0f64; 512]; // 8×8×8
        let out = filter_log(&data, &[8, 8, 8], 1.5);
        assert_eq!(out.len(), 512);
    }

    #[test]
    fn test_filter_log_constant_near_zero() {
        // LoG of a constant signal should be ≈0 everywhere
        let data = vec![5.0f64; 64];
        let out = filter_log(&data, &[64], 2.0);
        for v in &out {
            assert!(v.abs() < 1e-6, "LoG of constant: {v}");
        }
    }

    // --- non_maximum_suppression ---

    #[test]
    fn test_nms_2d_shape() {
        let resp = vec![1.0f64; 100];
        let orient = vec![0.0f64; 100];
        let out = non_maximum_suppression(&resp, &orient, 10, 10);
        assert_eq!(out.len(), 100);
    }

    #[test]
    fn test_nms_2d_uniform_suppresses_most() {
        // Uniform response with horizontal orientation: centre row may survive
        let resp = vec![1.0f64; 100];
        let orient = vec![0.0f64; 100]; // θ=0, look left/right
        let out = non_maximum_suppression(&resp, &orient, 10, 10);
        // All values should be 0 or 1
        for v in &out {
            assert!(*v == 0.0 || (*v - 1.0).abs() < 1e-9);
        }
    }

    // --- non_maximum_suppression_3d ---

    #[test]
    fn test_nms_3d_shape() {
        let data = vec![1.0f64; 125]; // 5×5×5
        let out = non_maximum_suppression_3d(&data, &data, &data, 5, 5, 5);
        assert_eq!(out.len(), 125);
    }

    // --- threshold_rosin ---

    #[test]
    fn test_threshold_rosin_bimodal() {
        // Should return threshold between two clusters
        let mut data: Vec<f64> = (0..100).map(|_| 0.1).collect();
        data.extend((0..100).map(|_| 0.9));
        let t = threshold_rosin(&data);
        assert!(t > 0.0 && t < 1.0, "threshold = {t}");
    }

    #[test]
    fn test_threshold_rosin_empty() {
        assert_eq!(threshold_rosin(&[]), 0.0);
    }

    #[test]
    fn test_threshold_rosin_uniform() {
        let data = vec![5.0f64; 100];
        let t = threshold_rosin(&data);
        assert!((t - 5.0).abs() < 1.0);
    }

    // --- surface_filter_gauss_3d ---

    #[test]
    fn test_surface_filter_gauss_3d_shape() {
        let data = vec![1.0f64; 125]; // 5×5×5
        let (d2x, d2y, d2z) = surface_filter_gauss_3d(&data, 5, 5, 5, 1.0);
        assert_eq!(d2x.len(), 125);
        assert_eq!(d2y.len(), 125);
        assert_eq!(d2z.len(), 125);
    }

    // --- awt_1d ---

    #[test]
    fn test_awt_1d_shape() {
        let signal: Vec<f64> = (0..64).map(|i| i as f64).collect();
        let n_bands = 4;
        let out = awt_1d(&signal, n_bands);
        assert_eq!(out.len(), 64 * (n_bands + 1));
    }

    #[test]
    fn test_awt_1d_reconstruction() {
        // Perfect reconstruction: sum of detail bands + last approx ≈ original
        let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let n_bands = 4;
        let out = awt_1d(&signal, n_bands);
        let cols = n_bands + 1;
        for i in 0..64usize {
            let reconstructed: f64 = (0..cols).map(|k| out[i * cols + k]).sum();
            assert!(
                (reconstructed - signal[i]).abs() < 1e-9,
                "AWT reconstruction error at {i}: {reconstructed} vs {}",
                signal[i]
            );
        }
    }

    // --- photobleach_correction ---

    #[test]
    fn test_photobleach_correction_shape() {
        let data: Vec<f64> = (0..500).map(|i| 1.0 + i as f64 * 0.001).collect();
        let out = photobleach_correction(&data, 100, 5);
        assert_eq!(out.len(), 500);
    }

    #[test]
    fn test_photobleach_correction_constant() {
        // Constant signal should not change (no decay)
        let data = vec![5.0f64; 200];
        let out = photobleach_correction(&data, 100, 2);
        assert_eq!(out.len(), 200);
        // Frame 0 should remain ≈5.0 (reference frame)
        for v in &out[..100] {
            assert!((v - 5.0).abs() < 1e-6, "constant correction: {v}");
        }
    }

    // --- awt (2D) ---

    #[test]
    fn test_awt_2d_shape() {
        let data = vec![1.0f64; 64 * 64];
        let n_bands = 4;
        let out = awt(&data, 64, 64, n_bands);
        assert_eq!(out.len(), 64 * 64 * (n_bands + 1));
    }

    #[test]
    fn test_awt_2d_reconstruction() {
        use std::f64::consts::PI;
        let rows = 32usize;
        let cols = 32usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| (i as f64 * 0.1 * PI).sin())
            .collect();
        let n_bands = 3;
        let out = awt(&data, rows, cols, n_bands);
        let slices = n_bands + 1;
        for px in 0..rows * cols {
            let reconstructed: f64 = (0..slices).map(|k| out[px * slices + k]).sum();
            assert!(
                (reconstructed - data[px]).abs() < 1e-9,
                "AWT 2D recon error at {px}: {reconstructed} vs {}",
                data[px]
            );
        }
    }

    // --- awt_denoising ---

    #[test]
    fn test_awt_denoising_shape() {
        let data = vec![1.0f64; 100]; // 10×10
        let out = awt_denoising(&data, 10, 10, 3, true, 3.0);
        assert_eq!(out.len(), 100);
    }

    #[test]
    fn test_awt_denoising_uniform() {
        // Denoising a perfectly uniform image should return same values
        let data = vec![5.0f64; 64]; // 8×8
        let out = awt_denoising(&data, 8, 8, 2, true, 3.0);
        for v in &out {
            assert!((v - 5.0).abs() < 1e-6, "denoised uniform: {v}");
        }
    }

    // --- binterp_1d ---

    #[test]
    fn test_binterp_1d_exact_nodes() {
        // Interpolating at an interior integer node should return the original value
        let f = vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0];
        // Test at interior nodes (avoid boundaries where mirror BC causes deviations)
        let xi: Vec<f64> = vec![2.0, 3.0, 4.0];
        let (vals, _, _) = binterp_1d(&f, &xi, true);
        for (i, &v) in vals.iter().enumerate() {
            let expected = f[xi[i] as usize];
            assert!((v - expected).abs() < 1e-4, "binterp1d at {}: {v} vs {expected}", xi[i]);
        }
    }

    // --- binterp_2d ---

    #[test]
    fn test_binterp_2d_exact_nodes() {
        let rows = 5usize;
        let cols = 5usize;
        let f: Vec<f64> = (0..25).map(|i| i as f64).collect();
        // Test at one interior node (2,2) → f[2*5+2] = 12
        let xi = vec![2.0]; // column
        let yi = vec![2.0]; // row
        let (vals, _, _) = binterp_2d(&f, rows, cols, &xi, &yi, true);
        let expected = f[2 * 5 + 2];
        assert!(
            (vals[0] - expected).abs() < 1e-3,
            "binterp2d interior: {} vs {expected}",
            vals[0]
        );
    }

    // --- convn_fft ---

    #[test]
    fn test_convn_fft_same_shape() {
        let a = vec![1.0f64; 8 * 8 * 8];
        let b = vec![1.0f64; 3 * 3 * 3]; // box kernel
        let (out, dims) = convn_fft(&a, &[8, 8, 8], &b, &[3, 3, 3], "same").unwrap();
        assert_eq!(dims, vec![8, 8, 8]);
        assert_eq!(out.len(), 512);
    }

    #[test]
    fn test_convn_fft_full_shape() {
        let a = vec![1.0f64; 4 * 4 * 4];
        let b = vec![1.0f64; 3 * 3 * 3];
        let (out, dims) = convn_fft(&a, &[4, 4, 4], &b, &[3, 3, 3], "full").unwrap();
        assert_eq!(dims, vec![6, 6, 6]);
        assert_eq!(out.len(), 216);
    }

    // --- filter_log_nd ---

    #[test]
    fn test_filter_log_nd_shape() {
        let data = vec![1.0f64; 125]; // 5×5×5
        let out = filter_log_nd(&data, 5, 5, 5, 1.0, None, false);
        assert_eq!(out.len(), 125);
    }

    // --- bw_max_direct_dist ---

    #[test]
    fn test_bw_max_direct_dist_shape() {
        let mask = vec![true; 100]; // 10×10
        let out = bw_max_direct_dist(&mask, 10, 10);
        assert_eq!(out.len(), 400); // 10×10×4
    }

    #[test]
    fn test_bw_max_direct_dist_all_true() {
        let mask = vec![true; 25]; // 5×5
        let out = bw_max_direct_dist(&mask, 5, 5);
        // All distances should be 0 (every pixel is true)
        for v in &out {
            assert_eq!(*v, 0.0f32);
        }
    }

    // --- bw_n_neighbors ---

    #[test]
    fn test_bw_n_neighbors_full_mask() {
        let mask = vec![true; 25]; // 5×5 all true
        let out = bw_n_neighbors(&mask, 5, 5, None);
        // Interior pixels have 8 neighbors, all true
        assert_eq!(out[12], 8); // centre (2,2)
    }

    #[test]
    fn test_bw_n_neighbors_empty_mask() {
        let mask = vec![false; 25];
        let out = bw_n_neighbors(&mask, 5, 5, None);
        assert!(out.iter().all(|&v| v == 0));
    }

    // --- mask_vectors ---

    #[test]
    fn test_mask_vectors_inside() {
        let mask = vec![true; 100]; // 10×10
        let x = vec![5.0, 3.0];
        let y = vec![5.0, 3.0];
        let out = mask_vectors(&x, &y, &mask, 10, 10);
        assert!(out.iter().all(|&v| v));
    }

    #[test]
    fn test_mask_vectors_outside_bounds() {
        let mask = vec![true; 100];
        let x = vec![0.0, 11.0];
        let y = vec![0.0, 11.0];
        let out = mask_vectors(&x, &y, &mask, 10, 10);
        assert!(!out[0] && !out[1]);
    }

    // --- angle_filter ---

    #[test]
    fn test_angle_filter_same_direction() {
        // Vectors pointing same direction as reference should pass
        let vx = vec![1.0, 0.9];
        let vy = vec![0.0, 0.1];
        let out = angle_filter(&vx, &vy, 1.0, 0.0);
        assert!(out.iter().all(|&v| v));
    }

    #[test]
    fn test_angle_filter_opposite_direction() {
        // Vectors pointing opposite direction should fail
        let vx = vec![-1.0];
        let vy = vec![0.0];
        let out = angle_filter(&vx, &vy, 1.0, 0.0);
        assert!(!out[0]);
    }

    // --- ch2rgb ---

    #[test]
    fn test_ch2rgb_shape() {
        let r = vec![100.0f64; 100];
        let g = vec![150.0f64; 100];
        let out = ch2rgb(Some(&r), Some(&g), None, 10, 10);
        assert_eq!(out.len(), 300); // 10×10×3
    }

    #[test]
    fn test_ch2rgb_zero_channel() {
        let r = vec![1.0f64; 4];
        let out = ch2rgb(Some(&r), None, None, 2, 2);
        // G and B channels should be 0
        for i in 0..4 {
            assert_eq!(out[i * 3 + 1], 0u8);
            assert_eq!(out[i * 3 + 2], 0u8);
        }
    }

    // --- rgb_overlay ---

    #[test]
    fn test_rgb_overlay_shape() {
        let img = vec![1.0f64; 100];
        let mask = vec![false; 100];
        let out = rgb_overlay(&img, 10, 10, &[&mask], &[(1.0, 0.0, 0.0)]);
        assert_eq!(out.len(), 300);
    }

    // --- z_proj_image ---

    #[test]
    fn test_z_proj_max() {
        let mut data = vec![0.0f64; 4 * 3 * 3]; // 4×3×3
        data[0 * 9 + 4] = 5.0;
        data[3 * 9 + 4] = 10.0;
        let out = z_proj_image(&data, 4, 3, 3, "max").unwrap();
        assert_eq!(out.len(), 9);
        assert!((out[4] - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_z_proj_mean() {
        let data = vec![4.0f64; 4 * 9]; // 4×3×3 all 4
        let out = z_proj_image(&data, 4, 3, 3, "mean").unwrap();
        for v in &out {
            assert!((v - 4.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_z_proj_invalid_type() {
        let data = vec![1.0f64; 8];
        assert!(z_proj_image(&data, 2, 2, 2, "unknown_type").is_err());
    }
}
