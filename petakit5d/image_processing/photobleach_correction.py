"""
Photobleaching correction for fluorescence microscopy time series.

Ported from MATLAB PetaKit5D imageProcessing/photobleachCorrection.m
"""

import numpy as np
from typing import Tuple
from scipy.optimize import curve_fit
import warnings


def photobleach_correction(
    images: np.ndarray,
    masks: np.ndarray
) -> np.ndarray:
    """
    Correct photobleaching in a fluorescence microscopy time series.
    
    This function fits an exponential decay model to the mean intensity
    over time and corrects each frame by dividing by the fitted curve.
    Uses a double exponential model for better fitting flexibility.
    
    Parameters
    ----------
    images : np.ndarray
        3D array of images (x, y, n_frames) or (n_frames, x, y)
        representing a time series of fluorescence images
    masks : np.ndarray
        Binary mask array with same shape as images, indicating
        regions to use for intensity calculation (True/1 = use, False/0 = ignore)
        
    Returns
    -------
    corrected_images : np.ndarray
        Photobleaching-corrected image stack with same shape as input
        
    Examples
    --------
    >>> import numpy as np
    >>> # Create synthetic photobleaching data
    >>> n_frames = 100
    >>> images = np.zeros((128, 128, n_frames))
    >>> for i in range(n_frames):
    ...     images[:, :, i] = np.exp(-0.02 * i) * np.random.rand(128, 128)
    >>> masks = np.ones_like(images)
    >>> corrected = photobleach_correction(images, masks)
    >>> 
    >>> # With specific ROI
    >>> masks = np.zeros_like(images)
    >>> masks[40:90, 40:90, :] = 1  # Only use central region
    >>> corrected = photobleach_correction(images, masks)
    
    Notes
    -----
    - Fits a double exponential decay: f(t) = a*exp(b*t) + c*exp(d*t)
    - Warns if R² < 0.8, indicating poor fit quality
    - Returns corrected images even if fit is poor (with warning)
    - Each frame is corrected by: corrected[i] = images[i] / fitted_curve[i]
    - Masked regions are used to compute mean intensity per frame
    
    Warnings
    --------
    - If fit quality (R²) is less than 0.8, a warning is issued
    - Images and masks must have identical shapes
    
    References
    ----------
    Based on exponential photobleaching model commonly used in fluorescence
    microscopy. See: Handbook of Time Series Analysis. Wiley, 2006. p. 438-460
    
    Author
    ------
    Original MATLAB: Marco Vilela, 2014
    Python port: 2025
    """
    if images.shape != masks.shape:
        raise ValueError(f"Images and masks have different shapes: "
                        f"{images.shape} vs {masks.shape}")
    
    if images.ndim != 3:
        raise ValueError(f"Expected 3D array (x, y, n_frames), got shape {images.shape}")
    
    x_len, y_len, n_frames = images.shape
    
    # Compute mean fluorescence intensity per frame in masked regions
    mean_intensities = np.zeros(n_frames)
    
    for i in range(n_frames):
        frame = images[:, :, i]
        mask = masks[:, :, i]
        
        # Sum intensities in masked region
        total_intensity = np.sum(frame * mask)
        pixel_count = np.sum(mask)
        
        if pixel_count > 0:
            mean_intensities[i] = total_intensity / pixel_count
        else:
            mean_intensities[i] = 0
    
    # Time points (0-indexed)
    time = np.arange(n_frames, dtype=float)
    
    # Handle single frame case
    if n_frames == 1:
        # No correction needed for single frame
        return images.copy()
    
    # Fit double exponential: y = a*exp(b*t) + c*exp(d*t)
    def double_exp(t, a, b, c, d):
        return a * np.exp(b * t) + c * np.exp(d * t)
    
    # Initial guess for parameters
    # Start with simple decay: one fast, one slow component
    p0 = [mean_intensities[0] * 0.6, -0.01,
          mean_intensities[0] * 0.4, -0.001]
    
    try:
        # Fit the curve
        popt, pcov = curve_fit(double_exp, time, mean_intensities, p0=p0,
                              maxfev=10000)
        
        # Compute fitted values
        fitted_curve = double_exp(time, *popt)
        
        # Calculate R-squared (coefficient of determination)
        ss_res = np.sum((mean_intensities - fitted_curve) ** 2)
        ss_tot = np.sum((mean_intensities - np.mean(mean_intensities)) ** 2)
        
        if ss_tot > 0:
            r_squared = 1 - (ss_res / ss_tot)
        else:
            r_squared = 0
        
        # Warn if fit is poor
        if r_squared < 0.8:
            warnings.warn(
                f"WARNING: Poor fit quality. Model explained {r_squared:.2%} "
                f"of data variance (< 80%). Photobleach correction may be inaccurate.",
                UserWarning
            )
    
    except RuntimeError as e:
        # If fitting fails, use simple linear correction
        warnings.warn(
            f"Curve fitting failed: {e}. Using linear decay model instead.",
            UserWarning
        )
        
        # Simple linear fit as fallback
        if n_frames > 1:
            slope = (mean_intensities[-1] - mean_intensities[0]) / (n_frames - 1)
            fitted_curve = mean_intensities[0] + slope * time
        else:
            fitted_curve = mean_intensities.copy()
        
        r_squared = 0.0
    
    # Correct images
    corrected_images = np.zeros_like(images)
    
    for i in range(n_frames):
        if fitted_curve[i] > 0:
            # Normalize by fitted curve
            corrected_images[:, :, i] = images[:, :, i] / fitted_curve[i]
        else:
            # If fitted value is zero or negative, don't correct
            corrected_images[:, :, i] = images[:, :, i]
    
    return corrected_images


def photobleach_correction_with_fit(
    images: np.ndarray,
    masks: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Correct photobleaching and return fit information.
    
    This extended version returns the corrected images along with
    the fitted curve, measured intensities, and R² value.
    
    Parameters
    ----------
    images : np.ndarray
        3D array of images (x, y, n_frames)
    masks : np.ndarray
        Binary mask array with same shape as images
        
    Returns
    -------
    corrected_images : np.ndarray
        Photobleaching-corrected image stack
    fitted_curve : np.ndarray
        Fitted exponential decay curve (length = n_frames)
    mean_intensities : np.ndarray
        Measured mean intensities per frame (length = n_frames)
    r_squared : float
        R² goodness of fit statistic (0 to 1, higher is better)
        
    See Also
    --------
    photobleach_correction : Main correction function
    """
    if images.shape != masks.shape:
        raise ValueError(f"Images and masks have different shapes")
    
    if images.ndim != 3:
        raise ValueError(f"Expected 3D array")
    
    x_len, y_len, n_frames = images.shape
    
    # Compute mean intensities
    mean_intensities = np.zeros(n_frames)
    
    for i in range(n_frames):
        frame = images[:, :, i]
        mask = masks[:, :, i]
        total_intensity = np.sum(frame * mask)
        pixel_count = np.sum(mask)
        
        if pixel_count > 0:
            mean_intensities[i] = total_intensity / pixel_count
        else:
            mean_intensities[i] = 0
    
    time = np.arange(n_frames, dtype=float)
    
    # Handle single frame case
    if n_frames == 1:
        fitted_curve = mean_intensities.copy()
        corrected_images = images.copy()
        return corrected_images, fitted_curve, mean_intensities, 1.0
    
    def double_exp(t, a, b, c, d):
        return a * np.exp(b * t) + c * np.exp(d * t)
    
    p0 = [mean_intensities[0] * 0.6, -0.01,
          mean_intensities[0] * 0.4, -0.001]
    
    try:
        popt, pcov = curve_fit(double_exp, time, mean_intensities, p0=p0,
                              maxfev=10000)
        fitted_curve = double_exp(time, *popt)
        
        ss_res = np.sum((mean_intensities - fitted_curve) ** 2)
        ss_tot = np.sum((mean_intensities - np.mean(mean_intensities)) ** 2)
        
        if ss_tot > 0:
            r_squared = 1 - (ss_res / ss_tot)
        else:
            r_squared = 0
        
        if r_squared < 0.8:
            warnings.warn(
                f"WARNING: Poor fit quality (R² = {r_squared:.2%})",
                UserWarning
            )
    
    except RuntimeError as e:
        warnings.warn(f"Curve fitting failed: {e}", UserWarning)
        
        if n_frames > 1:
            slope = (mean_intensities[-1] - mean_intensities[0]) / (n_frames - 1)
            fitted_curve = mean_intensities[0] + slope * time
        else:
            fitted_curve = mean_intensities.copy()
        
        r_squared = 0.0
    
    # Correct images
    corrected_images = np.zeros_like(images)
    
    for i in range(n_frames):
        if fitted_curve[i] > 0:
            corrected_images[:, :, i] = images[:, :, i] / fitted_curve[i]
        else:
            corrected_images[:, :, i] = images[:, :, i]
    
    return corrected_images, fitted_curve, mean_intensities, r_squared
