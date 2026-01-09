"""
Surface filtering with 3D Gaussian second derivatives.

This module provides surface detection using partial second derivatives of Gaussian kernels.

Author: Converted from MATLAB by GitHub Copilot
Original MATLAB author: Hunter Elliott (01/21/2010)
"""

import numpy as np
from scipy import ndimage
from typing import Union, Tuple, List


def surface_filter_gauss_3d(
    input_img: np.ndarray,
    sigma: Union[float, List[float], Tuple[float, ...]],
    border_condition: str = 'reflect'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter a 3D volume with Gaussian second derivative kernels for surface detection.
    
    Filters the input matrix using partial second derivatives of a Gaussian,
    giving a filtered "surface" image. The second derivatives are inverted so that
    the response is positive at bright surfaces and negative at troughs.
    
    Parameters
    ----------
    input_img : np.ndarray
        3D input array.
    sigma : float or sequence of 3 floats
        Standard deviation of the Gaussian. If scalar, same sigma is used for all
        dimensions. If 3-element sequence, specifies different sigmas for each dimension.
    border_condition : str, default='reflect'
        Border handling mode. Options: 'reflect', 'constant', 'nearest', 'mirror', 'wrap'.
        
    Returns
    -------
    d2X : np.ndarray
        Image filtered with second derivative in X direction (axis 2).
    d2Y : np.ndarray
        Image filtered with second derivative in Y direction (axis 1).
    d2Z : np.ndarray
        Image filtered with second derivative in Z direction (axis 0).
        
    Raises
    ------
    ValueError
        If input is not 3D.
        
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(50, 50, 30)
    >>> d2x, d2y, d2z = surface_filter_gauss_3d(img, sigma=2.0)
    >>> d2x.shape
    (50, 50, 30)
    
    >>> # Different sigma per dimension
    >>> d2x, d2y, d2z = surface_filter_gauss_3d(img, sigma=[1.0, 1.0, 2.0])
    """
    # Input validation
    if input_img.ndim != 3:
        raise ValueError(f"Input must be 3D, got {input_img.ndim}D")
    
    # Convert sigma to array
    if np.isscalar(sigma):
        sigma = np.array([sigma, sigma, sigma])
    else:
        sigma = np.array(sigma)
        if sigma.size != 3:
            raise ValueError(f"Sigma must have 3 elements, got {sigma.size}")
    
    # Map border condition to scipy mode
    mode_map = {
        'symmetric': 'reflect',
        'replicate': 'nearest',
        'circular': 'wrap',
        'antisymmetric': 'mirror'
    }
    mode = mode_map.get(border_condition, border_condition)
    
    # Filter in X direction (axis 2, corresponding to MATLAB dimension 2)
    w = int(np.ceil(5 * sigma[0]))
    x = np.arange(-w, w + 1)
    
    # Gaussian and its second derivative
    g = np.exp(-x**2 / (2 * sigma[0]**2))
    d2g = -(x**2 / sigma[0]**2 - 1) / sigma[0]**2 * np.exp(-x**2 / (2 * sigma[0]**2))
    
    # Normalize
    g_sum = np.sum(g)
    g = g / g_sum
    d2g = d2g / g_sum
    
    # Apply filters - note the order and axis mapping
    # scipy uses (z, y, x) ordering, MATLAB uses (y, x, z)
    d2X = ndimage.convolve1d(input_img, d2g, axis=1, mode=mode)  # Y axis in numpy = X in MATLAB
    d2X = ndimage.convolve1d(d2X, g, axis=0, mode=mode)  # X axis in numpy = Y in MATLAB  
    d2X = ndimage.convolve1d(d2X, g, axis=2, mode=mode)  # Z axis
    
    # Filter in Y direction (axis 1, corresponding to MATLAB dimension 1)
    w = int(np.ceil(5 * sigma[1]))
    x = np.arange(-w, w + 1)
    
    g = np.exp(-x**2 / (2 * sigma[1]**2))
    d2g = -(x**2 / sigma[1]**2 - 1) / sigma[1]**2 * np.exp(-x**2 / (2 * sigma[1]**2))
    
    g_sum = np.sum(g)
    g = g / g_sum
    d2g = d2g / g_sum
    
    d2Y = ndimage.convolve1d(input_img, g, axis=1, mode=mode)
    d2Y = ndimage.convolve1d(d2Y, d2g, axis=0, mode=mode)
    d2Y = ndimage.convolve1d(d2Y, g, axis=2, mode=mode)
    
    # Filter in Z direction (axis 0, corresponding to MATLAB dimension 3)
    w = int(np.ceil(5 * sigma[2]))
    x = np.arange(-w, w + 1)
    
    g = np.exp(-x**2 / (2 * sigma[2]**2))
    d2g = -(x**2 / sigma[2]**2 - 1) / sigma[2]**2 * np.exp(-x**2 / (2 * sigma[2]**2))
    
    g_sum = np.sum(g)
    g = g / g_sum
    d2g = d2g / g_sum
    
    d2Z = ndimage.convolve1d(input_img, g, axis=1, mode=mode)
    d2Z = ndimage.convolve1d(d2Z, g, axis=0, mode=mode)
    d2Z = ndimage.convolve1d(d2Z, d2g, axis=2, mode=mode)
    
    return d2X, d2Y, d2Z
