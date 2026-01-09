"""
Local statistics utilities.

Ported from MATLAB imageProcessing/ directory.
"""

import numpy as np
from scipy import ndimage
from typing import Tuple


def local_avg_std_2d(image: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute local average and standard deviation within a square window.
    
    Calculates the mean and standard deviation in a sliding window across the image.
    Handles NaN values appropriately.
    
    Args:
        image: Input 2D image
        window_size: Side length of square window (must be odd)
        
    Returns:
        Tuple of (average, std_dev) arrays with same shape as input
        
    Raises:
        ValueError: If window_size is not odd
        
    Examples:
        >>> img = np.random.rand(100, 100)
        >>> avg, std = local_avg_std_2d(img, 5)
        
    Original MATLAB function: localAvgStd2D.m
    Author: Francois Aguet (Last modified 09/19/2011)
    """
    if window_size % 2 == 0:
        raise ValueError('The window length w must be an odd integer.')
    
    nan_mask = np.isnan(image)
    
    # Create uniform kernel
    kernel = np.ones((window_size, window_size))
    
    # Count of non-NaN elements (use logical_not instead of ~)
    n = ndimage.convolve((~nan_mask).astype(float), kernel, mode='nearest')
    
    # Replace NaN with 0 for computation
    img_clean = image.copy()
    img_clean[nan_mask] = 0
    
    # Compute sum and sum of squares
    E = ndimage.convolve(img_clean, kernel, mode='nearest')
    E2 = ndimage.convolve(img_clean**2, kernel, mode='nearest')
    
    # Compute variance
    sigma = E2 - E**2 / n
    sigma[sigma < 0] = 0
    sigma = np.sqrt(sigma / (n - 1))
    
    # Compute average
    avg = E / n
    
    # Restore NaN values
    avg[nan_mask] = np.nan
    sigma[nan_mask] = np.nan
    
    return avg, sigma
