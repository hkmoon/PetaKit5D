"""
Gradient filtering functions for image processing.

This module provides functions for computing image gradients using Gaussian derivatives.
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Union, Optional


def gradient_filter_gauss_2d(
    image: np.ndarray,
    sigma: float,
    border_condition: str = 'symmetric'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter image with 2D Gaussian gradient mask to compute gradient.
    
    Filters the input matrix using partial derivatives of a Gaussian,
    giving a filtered gradient image. This is equivalent to MATLAB's
    gradientFilterGauss2D function.
    
    Parameters
    ----------
    image : np.ndarray
        2D input array to be filtered
    sigma : float
        Standard deviation of the Gaussian to use derivatives of for filtering
    border_condition : str, optional
        Border handling mode. Options are:
        - 'symmetric': mirror padding (default)
        - 'replicate': edge value replication
        - 'wrap': circular wrapping
        - 'constant': zero padding
        Default is 'symmetric'
    
    Returns
    -------
    dX : np.ndarray
        Matrix filtered with partial derivative in X direction (axis 1)
    dY : np.ndarray
        Matrix filtered with partial derivative in Y direction (axis 0)
    
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(100, 100)
    >>> dx, dy = gradient_filter_gauss_2d(img, sigma=2.0)
    >>> gradient_mag = np.sqrt(dx**2 + dy**2)
    
    Notes
    -----
    - X corresponds to matrix dimension 2 (columns)
    - Y corresponds to matrix dimension 1 (rows)
    - Uses separable convolution for efficiency
    
    References
    ----------
    Based on MATLAB gradientFilterGauss2D by Hunter Elliott (2/2014)
    """
    if image.ndim != 2:
        raise ValueError("Input image must be 2-dimensional")
    
    # Map border condition names to scipy equivalents
    mode_map = {
        'symmetric': 'reflect',
        'replicate': 'nearest',
        'circular': 'wrap',
        'constant': 'constant'
    }
    
    if border_condition in mode_map:
        mode = mode_map[border_condition]
    else:
        # Assume it's a constant value
        mode = 'constant'
        cval = float(border_condition)
    
    # Cutoff radius of the gaussian kernel
    w = int(np.ceil(3 * sigma))
    x = np.arange(-w, w + 1)
    
    # Gaussian and its derivative
    g = np.exp(-x**2 / (2 * sigma**2))
    dg = -x / sigma**2 * np.exp(-x**2 / (2 * sigma**2))
    
    # Normalize
    g_sum = np.sum(g)
    g = g / g_sum
    dg = dg / g_sum
    
    # Compute gradients using separable convolution
    # dX: derivative in X direction (horizontal, along columns)
    if mode == 'constant' and 'cval' in locals():
        padded = np.pad(image, w, mode=mode, constant_values=cval)
    else:
        padded = np.pad(image, w, mode=mode)
    
    # Apply horizontal derivative, then vertical smoothing
    dX = ndimage.convolve1d(padded, dg, axis=1, mode='constant')[w:-w, w:-w]
    dX = ndimage.convolve1d(dX, g, axis=0, mode='constant')
    
    # dY: derivative in Y direction (vertical, along rows)
    if mode == 'constant' and 'cval' in locals():
        padded = np.pad(image, w, mode=mode, constant_values=cval)
    else:
        padded = np.pad(image, w, mode=mode)
    
    # Apply vertical derivative, then horizontal smoothing
    dY = ndimage.convolve1d(padded, g, axis=1, mode='constant')[w:-w, w:-w]
    dY = ndimage.convolve1d(dY, dg, axis=0, mode='constant')
    
    return dX, dY


def gradient_filter_gauss_3d(
    image: np.ndarray,
    sigma: Union[float, Tuple[float, float, float]],
    border_condition: str = 'symmetric'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter image with 3D Gaussian gradient mask to compute 3D gradient.
    
    Parameters
    ----------
    image : np.ndarray
        3D input array to be filtered
    sigma : float or tuple of 3 floats
        Standard deviation(s) of the Gaussian. If scalar, same sigma
        is used for all dimensions
    border_condition : str, optional
        Border handling mode (default: 'symmetric')
    
    Returns
    -------
    dX : np.ndarray
        Gradient in X direction (axis 2)
    dY : np.ndarray
        Gradient in Y direction (axis 1)
    dZ : np.ndarray
        Gradient in Z direction (axis 0)
    
    Examples
    --------
    >>> import numpy as np
    >>> vol = np.random.rand(50, 50, 50)
    >>> dx, dy, dz = gradient_filter_gauss_3d(vol, sigma=2.0)
    
    Notes
    -----
    Extension of gradient_filter_gauss_2d to 3D volumes
    """
    if image.ndim != 3:
        raise ValueError("Input image must be 3-dimensional")
    
    # Handle sigma
    if np.isscalar(sigma):
        sigma = (sigma, sigma, sigma)
    elif len(sigma) != 3:
        raise ValueError("Sigma must be scalar or tuple of 3 values")
    
    # Map border condition
    mode_map = {
        'symmetric': 'reflect',
        'replicate': 'nearest',
        'circular': 'wrap',
        'constant': 'constant'
    }
    mode = mode_map.get(border_condition, 'reflect')
    
    # Compute for each dimension
    gradients = []
    
    for dim in range(3):
        sig = sigma[dim]
        w = int(np.ceil(3 * sig))
        x = np.arange(-w, w + 1)
        
        # Gaussian and derivative
        g = np.exp(-x**2 / (2 * sig**2))
        dg = -x / sig**2 * np.exp(-x**2 / (2 * sig**2))
        
        # Normalize
        g_sum = np.sum(g)
        g = g / g_sum
        dg = dg / g_sum
        
        # Apply separable convolution
        result = image.copy()
        
        for axis in range(3):
            if axis == dim:
                # Apply derivative in this dimension
                result = ndimage.convolve1d(result, dg, axis=axis, mode=mode)
            else:
                # Apply smoothing in other dimensions
                result = ndimage.convolve1d(result, g, axis=axis, mode=mode)
        
        gradients.append(result)
    
    return gradients[2], gradients[1], gradients[0]  # dX, dY, dZ
