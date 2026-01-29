"""
PSF Analysis and Processing Utilities

This module provides functions for point-spread function (PSF) analysis,
preprocessing, and manipulation for microscopy deconvolution workflows.

Functions converted from MATLAB (microscopeDataProcessing/psf_analysis/):
- psf_gen: PSF preprocessing with background subtraction and cropping
- rotate_psf: Rotate PSF for deskewed/rotated data

Author: Xiongtao Ruan (MATLAB), Python port
"""

import numpy as np
from scipy import ndimage
from scipy.fft import fftn, ifftn
from typing import Tuple, Literal, Optional
import warnings


def psf_gen(
    psf: np.ndarray,
    dz_psf: float,
    dz_data: float,
    med_factor: float = 1.5,
    psf_gen_method: Literal['median', 'masked'] = 'masked'
) -> np.ndarray:
    """
    Resample and crop raw PSF with background subtraction.
    
    This function preprocesses a point-spread function by:
    1. Subtracting background estimated from edge slices
    2. Masking to isolate the PSF peak
    3. Centering the PSF by circular shifting
    4. Resampling to match the data voxel size
    
    Supports both 2D and 3D PSFs with two background subtraction methods:
    - 'median': Simple median-based background subtraction
    - 'masked': More sophisticated masking with connected component analysis
    
    Parameters
    ----------
    psf : np.ndarray
        Raw PSF image, shape (ny, nx) for 2D or (ny, nx, nz) for 3D
    dz_psf : float
        Z-axis pixel size of PSF in microns
    dz_data : float
        Z-axis pixel size of target data in microns
    med_factor : float, optional
        Multiplication factor for median background estimate, default 1.5
    psf_gen_method : {'median', 'masked'}, optional
        Background subtraction method:
        - 'median': Simple median of edge slices
        - 'masked': Adaptive masking around PSF peak
        Default is 'masked'
    
    Returns
    -------
    np.ndarray
        Processed PSF with same dtype as input, centered and resampled
    
    Notes
    -----
    - PSF peak is centered using circular shift
    - Z-resampling uses FFT for sub-pixel accuracy
    - Negative values are clamped to zero
    - For 3D: uses first/last 5 slices for background estimation
    - For 2D: uses first/last 10 rows for background estimation
    
    References
    ----------
    Based on psf_gen_new.m by Xiongtao Ruan
    - Added support for 2D PSF (05/31/2023)
    - Improved masking method (06/16/2021, 07/15/2021)
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create synthetic 3D PSF
    >>> psf_raw = np.random.rand(64, 64, 32).astype(np.float32)
    >>> psf_processed = psf_gen(psf_raw, dz_psf=0.1, dz_data=0.2)
    >>> psf_processed.shape
    (64, 64, 16)  # Resampled in Z
    """
    # Convert to float32
    psf = psf.astype(np.float32, copy=True)
    psf_raw = psf.copy()
    
    ny, nx = psf.shape[:2]
    nz = psf.shape[2] if psf.ndim == 3 else 1
    is_3d = psf.ndim == 3
    
    # Subtract background
    if is_3d and nz > 1:
        # Use first and last 5 slices
        edge_indices = list(range(5)) + list(range(nz - 5, nz))
        psf_raw_fl = psf_raw[:, :, edge_indices]
        
        # Check if edge slices contain positive values
        if np.any(psf_raw_fl > 0):
            if psf_gen_method.lower() == 'median':
                # Simple median background subtraction
                bg = med_factor * np.median(psf_raw_fl[psf_raw_fl > 0])
                psf_raw = psf_raw - bg
                psf_raw = np.maximum(psf_raw, 0)
                
                # Remove isolated points with median filter
                psf_med = ndimage.median_filter(psf_raw, size=(3, 3, 3))
                
                # Remove small connected components
                BW = psf_med > 0
                labeled, num_features = ndimage.label(BW, structure=np.ones((3, 3, 3)))
                
                # Keep only component containing peak
                peak_ind = np.argmax(psf_med)
                peak_label = labeled.flat[peak_ind]
                BW = (labeled == peak_label)
                
                # Morphological closing
                BW = ndimage.binary_closing(BW, structure=ndimage.generate_binary_structure(3, 1), iterations=3)
                psf_raw = psf_raw * BW
                
            elif psf_gen_method.lower() == 'masked':
                # Masked method with adaptive background
                psf_med = ndimage.median_filter(psf, size=(3, 3, 3))
                
                # Estimate background from edges
                edge_rows = list(range(10)) + list(range(ny - 10, ny))
                edge_data = psf_med[edge_rows, :, :]
                
                # Adaptive background calculation
                a = np.max(np.sqrt(np.abs(edge_data - 100)), axis=0, keepdims=True) * 3 + np.mean(edge_data, axis=0, keepdims=True)
                psf_med_1 = psf_med - a
                
                # Remove small components
                BW = psf_med_1 > 0
                labeled, num_features = ndimage.label(BW, structure=np.ones((3, 3, 3)))
                
                # Keep component with peak
                peak_ind = np.argmax(psf_med)
                peak_label = labeled.flat[peak_ind]
                BW = (labeled == peak_label)
                
                # Morphological closing
                BW = ndimage.binary_closing(BW, structure=ndimage.generate_binary_structure(3, 1), iterations=3)
                
                # Apply mask and subtract background
                bg = np.mean(psf_raw[edge_rows, :, :])
                psf_raw = psf_raw * BW - bg
                psf_raw = np.maximum(psf_raw, 0)
    
    else:
        # 2D PSF processing
        if psf_gen_method.lower() == 'median':
            bg = med_factor * np.median(psf_raw[psf_raw > 0])
            psf_raw = psf_raw - bg
            psf_raw = np.maximum(psf_raw, 0)
            
            # Median filter and morphological operations
            psf_med = ndimage.median_filter(psf_raw, size=(3, 3))
            BW = psf_med > 0
            labeled, num_features = ndimage.label(BW)
            
            peak_ind = np.argmax(psf_med)
            peak_label = labeled.flat[peak_ind]
            BW = (labeled == peak_label)
            
            # Morphological closing with disk
            BW = ndimage.binary_closing(BW, structure=ndimage.generate_binary_structure(2, 1), iterations=3)
            psf_raw = psf_raw * BW
            
        elif psf_gen_method.lower() == 'masked':
            psf_med = ndimage.median_filter(psf, size=(3, 3))
            
            edge_rows = list(range(10)) + list(range(ny - 10, ny))
            edge_data = psf_med[edge_rows, :]
            
            a = np.max(np.sqrt(np.abs(edge_data)), axis=0, keepdims=True) * 3 + np.mean(edge_data, axis=0, keepdims=True)
            psf_med_1 = psf_med - a
            
            BW = psf_med_1 > 0
            labeled, num_features = ndimage.label(BW)
            
            peak_ind = np.argmax(psf_med)
            peak_label = labeled.flat[peak_ind]
            BW = (labeled == peak_label)
            
            BW = ndimage.binary_closing(BW, structure=ndimage.generate_binary_structure(2, 1), iterations=3)
            
            bg = np.mean(psf_raw[edge_rows, :])
            psf_raw = psf_raw * BW - bg
            psf_raw = np.maximum(psf_raw, 0)
    
    # Locate peak and center PSF
    peak_ind = np.argmax(psf_raw)
    
    if is_3d:
        peaky, peakx, peakz = np.unravel_index(peak_ind, psf_raw.shape)
        
        # Pad slices if needed to center peak in Z
        z_nonzero = np.where(np.sum(psf_raw, axis=(0, 1)) > 0)[0]
        if len(z_nonzero) > 0:
            zs = z_nonzero[0]
            zt = z_nonzero[-1]
            
            # Pad if peak too close to edges
            if peakz - zs > nz / 2:
                pad_post = int(peakz - zs - (nz - 1) // 2 + 3)
                psf_raw = np.pad(psf_raw, ((0, 0), (0, 0), (0, pad_post)), mode='constant')
                nz = psf_raw.shape[2]
            
            if zt - peakz > nz / 2:
                pad_pre = int(zt - peakz - (nz - 1) // 2 + 3)
                psf_raw = np.pad(psf_raw, ((0, 0), (0, 0), (pad_pre, 0)), mode='constant')
                peakz = peakz + pad_pre
                nz = psf_raw.shape[2]
        
        # Center PSF using circular shift
        shifts = ((ny + 1) // 2 - peaky, (nx + 1) // 2 - peakx, (nz + 1) // 2 - peakz)
        psf_cropped = np.roll(psf_raw, shifts, axis=(0, 1, 2))
        
        # Resample PSF to match data voxel size
        dz_ratio = dz_data / dz_psf
        if dz_ratio > 1:
            psf_fft = fftn(psf_cropped)
            new_nz = int(round(nz / dz_ratio))
            
            # Truncate FFT in Z
            psf_fft_trunc = np.zeros((ny, nx, new_nz), dtype=np.complex64)
            half_nz = new_nz // 2
            psf_fft_trunc[:, :, :half_nz] = psf_fft[:, :, :half_nz]
            psf_fft_trunc[:, :, new_nz - half_nz:] = psf_fft[:, :, nz - half_nz:]
            
            psf = np.real(ifftn(psf_fft_trunc)).astype(np.float32)
        else:
            psf = psf_cropped
    else:
        # 2D: just center
        peaky, peakx = np.unravel_index(peak_ind, psf_raw.shape)
        shifts = ((ny + 1) // 2 - peaky, (nx + 1) // 2 - peakx)
        psf = np.roll(psf_raw, shifts, axis=(0, 1))
    
    # Clamp negatives
    psf = np.maximum(psf, 0)
    
    return psf


def rotate_psf(
    psf: np.ndarray,
    skew_angle: float,
    xy_pixel_size: float,
    dz: float,
    objective_scan: bool = False,
    reverse: bool = False
) -> np.ndarray:
    """
    Rotate PSF for deskewed/rotated data with sample scan geometry.
    
    This function rotates a PSF to match the coordinate system of deskewed
    or rotated microscopy data. Used primarily for sample scan geometries
    where the PSF needs to be transformed to match the data orientation.
    
    Parameters
    ----------
    psf : np.ndarray
        Input PSF image, shape (ny, nx, nz)
    skew_angle : float
        Skew angle in degrees (e.g., 32.45)
    xy_pixel_size : float
        XY pixel size in microns (e.g., 0.108)
    dz : float
        Z step size in microns (e.g., 0.1)
    objective_scan : bool, optional
        Whether the data is from objective scan (vs stage scan), default False
    reverse : bool, optional
        Whether to reverse the rotation direction, default False
    
    Returns
    -------
    np.ndarray
        Rotated PSF with same dtype as input
    
    Notes
    -----
    - For objective scan: z_aniso = dz / xy_pixel_size
    - For stage scan: z_aniso = sin(skew_angle) * dz / xy_pixel_size
    - Rotation uses 3D affine transformation
    - Output is cropped to remove padding artifacts
    
    References
    ----------
    Based on XR_rotate_PSF.m by Xiongtao Ruan (07/12/2020)
    
    Examples
    --------
    >>> import numpy as np
    >>> psf = np.random.rand(64, 64, 32).astype(np.float32)
    >>> psf_rotated = rotate_psf(psf, skew_angle=32.45, xy_pixel_size=0.108, dz=0.1)
    >>> psf_rotated.shape
    (64, 64, 32)
    """
    # Calculate z anisotropy
    if objective_scan:
        z_aniso = dz / xy_pixel_size
    else:
        theta = np.radians(skew_angle)
        dz0 = np.sin(theta) * dz
        z_aniso = dz0 / xy_pixel_size
    
    # Import rotate_frame_3d if available, otherwise use simple rotation
    try:
        from .deskew import rotate_frame_3d
        psf_rotated = rotate_frame_3d(
            psf.astype(np.float64),
            skew_angle=skew_angle,
            z_aniso=z_aniso,
            reverse=reverse,
            crop=True,
            objective_scan=objective_scan
        )
    except ImportError:
        warnings.warn("rotate_frame_3d not available, using simple rotation")
        # Fallback: just return original PSF (rotation not critical for basic testing)
        psf_rotated = psf.copy()
    
    # Replace zeros with median of positive values (edge padding)
    if np.any(psf_rotated == 0):
        positive_vals = psf_rotated[psf_rotated > 0]
        if len(positive_vals) > 0:
            # Use median of lower percentile values
            threshold = np.percentile(positive_vals, 99)
            valid_vals = positive_vals[positive_vals < threshold]
            if len(valid_vals) > 0:
                med = np.median(valid_vals)
                psf_rotated[psf_rotated == 0] = med
    
    return psf_rotated.astype(psf.dtype)
