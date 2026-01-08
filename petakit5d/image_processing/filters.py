"""
Filtering utilities.

Ported from MATLAB imageProcessing/ directory.
"""

import numpy as np
from scipy import ndimage
from typing import Optional, Tuple, Union


def filter_gauss_2d(
    image: np.ndarray,
    sigma: float,
    border_condition: str = 'reflect'
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Filter an image with a 2-D Gaussian mask.
    
    Args:
        image: 2-D input array
        sigma: Standard deviation of the Gaussian
        border_condition: Border handling mode. Options: 'reflect', 'constant',
                         'nearest', 'mirror', 'wrap'. Default: 'reflect'
                         (maps to MATLAB's 'symmetric')
        
    Returns:
        Tuple of (filtered_image, gaussian_kernel)
        
    Examples:
        >>> img = np.random.rand(100, 100)
        >>> filtered, kernel = filter_gauss_2d(img, sigma=2.0)
        
    Original MATLAB function: filterGauss2D.m
    Author: Francois Aguet (01/21/2010)
    """
    # Cutoff radius of the gaussian kernel
    w = int(np.ceil(3 * sigma))
    x = np.arange(-w, w + 1)
    
    # Create 1D Gaussian
    g = np.exp(-x**2 / (2 * sigma**2))
    g = g / np.sum(g)
    
    # Apply Gaussian filter using scipy (which handles padding internally)
    # Use ndimage.convolve for 2D separable Gaussian
    mode_map = {
        'symmetric': 'reflect',
        'replicate': 'nearest',
        'circular': 'wrap',
        'reflect': 'reflect',
        'constant': 'constant',
        'nearest': 'nearest',
        'mirror': 'mirror',
        'wrap': 'wrap'
    }
    
    scipy_mode = mode_map.get(border_condition, 'reflect')
    
    # Apply separable convolution: first along rows, then columns
    out = ndimage.convolve1d(image, g, axis=1, mode=scipy_mode)
    out = ndimage.convolve1d(out, g, axis=0, mode=scipy_mode)
    
    # Create 2D Gaussian kernel for output
    G = np.outer(g, g)
    
    return out, G


def filter_gauss_3d(
    input_volume: np.ndarray,
    sigma: Union[float, Tuple[float, float]],
    border_condition: str = 'reflect'
) -> np.ndarray:
    """
    Filter a data volume with a 3-D Gaussian mask.
    
    Args:
        input_volume: 3-D input array
        sigma: Standard deviation of the Gaussian. Can be:
               - Single float: same sigma for x, y, and z
               - Tuple of 2 floats: (sigma_xy, sigma_z)
        border_condition: Border handling mode. Options: 'reflect', 'constant',
                         'nearest', 'mirror', 'wrap'. Default: 'reflect'
        
    Returns:
        np.ndarray: Filtered volume
        
    Examples:
        >>> vol = np.random.rand(50, 50, 30)
        >>> filtered = filter_gauss_3d(vol, sigma=2.0)
        >>> filtered_aniso = filter_gauss_3d(vol, sigma=(2.0, 1.0))
        
    Original MATLAB function: filterGauss3D.m
    Author: Francois Aguet (01/21/2010)
    """
    # Handle sigma input
    if isinstance(sigma, (int, float)):
        sigma_xy = float(sigma)
        sigma_z = float(sigma)
    else:
        sigma_xy = float(sigma[0])
        sigma_z = float(sigma[1])
    
    # Cutoff radius
    w_xy = int(np.ceil(3 * sigma_xy))
    w_z = int(np.ceil(3 * sigma_z))
    
    # Create 1D Gaussians
    x_xy = np.arange(-w_xy, w_xy + 1)
    g_xy = np.exp(-x_xy**2 / (2 * sigma_xy**2))
    g_xy = g_xy / np.sum(g_xy)
    
    x_z = np.arange(-w_z, w_z + 1)
    g_z = np.exp(-x_z**2 / (2 * sigma_z**2))
    g_z = g_z / np.sum(g_z)
    
    # Map border conditions
    mode_map = {
        'symmetric': 'reflect',
        'replicate': 'nearest',
        'circular': 'wrap',
        'reflect': 'reflect',
        'constant': 'constant',
        'nearest': 'nearest',
        'mirror': 'mirror',
        'wrap': 'wrap'
    }
    
    scipy_mode = mode_map.get(border_condition, 'reflect')
    
    # Apply separable 3D convolution
    out = ndimage.convolve1d(input_volume, g_xy, axis=1, mode=scipy_mode)  # x
    out = ndimage.convolve1d(out, g_xy, axis=0, mode=scipy_mode)  # y
    out = ndimage.convolve1d(out, g_z, axis=2, mode=scipy_mode)  # z
    
    return out
