"""
B-spline interpolation utilities.

Ported from MATLAB imageProcessing/ directory.
"""

import numpy as np
from typing import Literal


def b3spline_1d(
    image: np.ndarray,
    boundary: Literal['mirror', 'periodic'] = 'mirror'
) -> np.ndarray:
    """
    Compute 1D cubic B-spline coefficients.
    
    Args:
        image: Input image array
        boundary: Boundary conditions ('mirror' or 'periodic')
        
    Returns:
        np.ndarray: B-spline coefficients
        
    Examples:
        >>> img = np.random.rand(10, 100)
        >>> coeffs = b3spline_1d(img, 'mirror')
        
    Original MATLAB function: b3spline1D.m
    Author: Francois Aguet (June 2010)
    """
    # Cubic spline parameters
    c0 = 6
    z1 = -2 + np.sqrt(3)
    
    N = image.shape[1]
    cp = np.zeros_like(image, dtype=float)
    cn = np.zeros_like(image, dtype=float)
    
    if boundary == 'mirror':
        cp[:, 0] = _get_causal_init_mirror(image, z1)
        for k in range(1, N):
            cp[:, k] = image[:, k] + z1 * cp[:, k - 1]
        
        cn[:, N - 1] = _get_anticausal_init_mirror(cp, z1)
        for k in range(N - 2, -1, -1):
            cn[:, k] = z1 * (cn[:, k + 1] - cp[:, k])
            
    elif boundary == 'periodic':
        cp[:, 0] = _get_causal_init_periodic(image, z1)
        for k in range(1, N):
            cp[:, k] = image[:, k] + z1 * cp[:, k - 1]
        
        cn[:, N - 1] = _get_anticausal_init_periodic(cp, z1)
        for k in range(N - 2, -1, -1):
            cn[:, k] = z1 * (cn[:, k + 1] - cp[:, k])
    else:
        raise ValueError(f"Unknown boundary condition: {boundary}")
    
    cn = c0 * cn
    return cn


def _get_anticausal_init_mirror(image: np.ndarray, a: float) -> np.ndarray:
    """Get anticausal initialization for mirror boundary."""
    N = image.shape[1]
    c0 = (a / (a * a - 1)) * (image[:, N - 1] + a * image[:, N - 2])
    return c0


def _get_anticausal_init_periodic(image: np.ndarray, a: float) -> np.ndarray:
    """Get anticausal initialization for periodic boundary."""
    N = image.shape[1]
    img = np.concatenate([image[:, N - 1:N], image, image[:, 0:N - 3]], axis=1)
    k = np.arange(2 * N - 3)
    # Broadcast k across rows
    c0 = -a / (1 - a**N) * np.sum((a**k)[np.newaxis, :] * img, axis=1)
    return c0


def _get_causal_init_mirror(image: np.ndarray, a: float) -> np.ndarray:
    """Get causal initialization for mirror boundary."""
    N = image.shape[1]
    k = np.arange(2 * N - 3)
    img = np.concatenate([image, image[:, -2:0:-1]], axis=1)
    # Broadcast k across rows
    out = np.sum(img * (a**k)[np.newaxis, :], axis=1) / (1 - a**(2 * N - 2))
    return out


def _get_causal_init_periodic(image: np.ndarray, a: float) -> np.ndarray:
    """Get causal initialization for periodic boundary."""
    N = image.shape[1]
    k = np.arange(N)
    img = np.concatenate([image[:, 0:1], image[:, -1:0:-1]], axis=1)
    # Broadcast k across rows
    out = np.sum(img * (a**k)[np.newaxis, :], axis=1) / (1 - a**N)
    return out


def b3spline_2d(
    image: np.ndarray,
    boundary: Literal['mirror', 'periodic'] = 'mirror'
) -> np.ndarray:
    """
    Compute 2D cubic B-spline coefficients.
    
    Applies 1D B-spline computation along each dimension.
    
    Args:
        image: Input 2D image
        boundary: Boundary conditions ('mirror' or 'periodic')
        
    Returns:
        np.ndarray: 2D B-spline coefficients
        
    Examples:
        >>> img = np.random.rand(100, 100)
        >>> coeffs = b3spline_2d(img, 'mirror')
        
    Original MATLAB function: b3spline2D.m
    Author: Francois Aguet (June 2010)
    """
    # Apply along rows
    c = b3spline_1d(image, boundary)
    # Apply along columns (transpose, apply, transpose back)
    c = b3spline_1d(c.T, boundary).T
    return c
