"""
High Dynamic Range (HDR) image merging functions.

This module provides functions for merging images taken at different exposure levels
to create high dynamic range images.

Based on MATLAB implementation by Jessica Tytell (October 25, 2011).
"""

import numpy as np
from typing import Tuple, Optional
import warnings


def threshold_rosin(image: np.ndarray) -> float:
    """
    Compute Rosin threshold for image segmentation.
    
    Simple implementation of Rosin thresholding based on histogram analysis.
    Finds threshold by drawing a line from histogram peak to tail and finding
    the point with maximum perpendicular distance.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
        
    Returns
    -------
    float
        Threshold value
    """
    # Compute histogram
    hist, bin_edges = np.histogram(image.ravel(), bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find peak
    peak_idx = np.argmax(hist)
    
    # Find last non-zero bin
    non_zero = np.where(hist > 0)[0]
    if len(non_zero) == 0:
        return float(np.mean(image))
    tail_idx = non_zero[-1]
    
    if peak_idx == tail_idx:
        return bin_centers[peak_idx]
    
    # Line from peak to tail
    x1, y1 = peak_idx, hist[peak_idx]
    x2, y2 = tail_idx, hist[tail_idx]
    
    # Perpendicular distances from line
    max_dist = 0
    threshold_idx = peak_idx
    
    for i in range(peak_idx, tail_idx + 1):
        x0, y0 = i, hist[i]
        # Distance from point to line
        dist = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2 + 1e-10)
        if dist > max_dist:
            max_dist = dist
            threshold_idx = i
    
    return bin_centers[threshold_idx]


def high_dynamic_range_merge(
    image_low: np.ndarray,
    image_high: np.ndarray,
    mystery_offset_factor: float = 1.25,
    saturation_value: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge two images taken at different exposure levels into high dynamic range image.
    
    Takes two images of the same field taken at different exposure levels, removes
    the saturated region from the higher exposed image, and replaces it with a 
    rescaled version of the same region from the lower exposed image.
    
    Parameters
    ----------
    image_low : np.ndarray
        Image taken at lower exposure (dimmer image)
    image_high : np.ndarray
        Image taken at higher exposure (brighter image)
    mystery_offset_factor : float, optional
        Scaling factor for merging (default: 1.25)
    saturation_value : float, optional
        Value indicating pixel saturation. If None, uses 4095 for 12-bit images
        
    Returns
    -------
    combined_mystery : np.ndarray
        Combined HDR image using mystery offset factor
    log_mystery : np.ndarray
        Log-scaled version of combined image for visualization
    log_image : np.ndarray
        Log-transformed combined image using linear correction
        
    Notes
    -----
    This function merges images by:
    1. Creating mask of saturated pixels in high-exposure image
    2. Finding linear relationship between images in non-saturated regions
    3. Replacing saturated regions with rescaled low-exposure data
    4. Applying log transform for better visualization
    
    Based on MATLAB implementation by Jessica Tytell (October 25, 2011).
    
    Examples
    --------
    >>> import numpy as np
    >>> # Simulate low and high exposure images
    >>> low = np.random.randint(0, 2000, (100, 100), dtype=np.uint16)
    >>> high = np.random.randint(0, 4095, (100, 100), dtype=np.uint16)
    >>> combined, log_m, log_img = high_dynamic_range_merge(low, high)
    """
    # Input validation
    if image_low.shape != image_high.shape:
        raise ValueError("Images must have the same shape")
    
    image_low = image_low.astype(np.float64)
    image_high = image_high.astype(np.float64)
    
    # Determine saturation value
    if saturation_value is None:
        # Assume 12-bit images (common in microscopy)
        saturation_value = 4095
    
    # Make mask of saturated pixels (saturated pixels = 1)
    sat_mask = (image_high >= saturation_value)
    
    # Make masked image of bright that gets rid of saturated pixels
    bright_masked = image_high * (~sat_mask)
    
    # Make image of only regions from dim image that are saturated in bright image
    dim_masked = image_low * sat_mask
    
    # Make a mask for background region (coarse, for merging purpose only)
    segment_level = threshold_rosin(image_high)
    background_mask = image_high < segment_level
    
    # A combined mask for both the saturated area and the background area
    bg_st_mask = np.logical_or(sat_mask, background_mask)
    
    # Find polynomial fit (linear) that says relationship between two images
    # Only use non-saturated pixels
    non_sat_pixels_high = image_high[~sat_mask]
    non_sat_pixels_low = image_low[~sat_mask]
    
    if len(non_sat_pixels_high) > 0:
        # Fit: low = coeff * high + offset
        coeffs = np.polyfit(non_sat_pixels_high, non_sat_pixels_low, 1)
        coeff = coeffs[0]
        offset = coeffs[1]
    else:
        warnings.warn("All pixels are saturated, using default scaling")
        coeff = 1.0
        offset = 0.0
    
    # Find a linear relationship between two images (linear, not affine)
    # Using non-background and non-saturated pixels
    high_data = image_high[~bg_st_mask]
    low_data = image_low[~bg_st_mask]
    
    if len(high_data) > 0 and np.sum(high_data**2) > 0:
        bright2dim_scaler = np.dot(high_data, low_data) / np.dot(high_data, high_data)
    else:
        bright2dim_scaler = coeff
    
    # Combine images using correction
    combined = (bright_masked * coeff + offset) + dim_masked
    
    # Combine images using linear correction
    combined_linear = (bright_masked * bright2dim_scaler) + dim_masked
    
    # Flatten combined image by taking the log of the image
    # Add small constant to avoid log(0)
    epsilon = 1.0
    log_image = np.log(combined_linear + epsilon)
    
    # Scale image using mystery factor
    combined_mystery = (bright_masked * coeff * mystery_offset_factor + offset) + dim_masked
    
    # Log of mystery-scaled image
    log_mystery = np.log(combined_mystery + epsilon)
    
    return combined_mystery, log_mystery, log_image
