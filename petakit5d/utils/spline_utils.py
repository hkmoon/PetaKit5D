"""
B-spline interpolation utilities.

This module provides inverse B-spline interpolation functions for image resampling.
"""

import numpy as np
from typing import Optional, Tuple


def ib3spline_1d(
    coeffs: np.ndarray,
    nx: Optional[int] = None
) -> np.ndarray:
    """
    Inverse 1D cubic B-spline interpolation.
    
    Interpolates along the second axis (columns) using cubic B-spline basis
    functions, given the B-spline coefficients.
    
    Args:
        coeffs: B-spline coefficients, shape (ny, nx_coeffs)
        nx: Target number of samples. If None, uses coeffs.shape[1]
    
    Returns:
        Interpolated image, shape (ny, nx)
    
    Notes:
        - Uses mirror (symmetric) boundary conditions
        - Cubic B-spline uses 4-point support with weights based on distance
    
    Examples:
        >>> coeffs = np.random.rand(10, 20)
        >>> result = ib3spline_1d(coeffs, 40)
        >>> result.shape
        (10, 40)
    """
    if coeffs.ndim != 2:
        raise ValueError("coeffs must be 2D array")
    
    ny_coeffs, nx_coeffs = coeffs.shape
    
    if nx is None:
        nx = nx_coeffs
    
    # Rescale coordinates
    x_scaled = (np.arange(1, nx + 1)) / (nx / nx_coeffs)
    
    # Compute the interpolation indexes (4-point support)
    # In MATLAB: xIndex = bsxfun(@plus, floor(xScaled), (-1:2)');
    x_floor = np.floor(x_scaled)
    x_index = x_floor[np.newaxis, :] + np.arange(-1, 3)[:, np.newaxis]  # Shape: (4, nx)
    
    # Compute weights based on fractional position
    t = x_scaled - x_floor  # Distance from floor
    
    # Cubic B-spline basis weights
    w = np.zeros((4, nx))
    w[3, :] = (1/6) * t**3
    w[0, :] = (1/6) + (1/2) * t * (t - 1) - w[3, :]
    w[2, :] = t + w[0, :] - 2 * w[3, :]
    w[1, :] = 1 - w[0, :] - w[2, :] - w[3, :]
    
    # Add symmetric padding (mirror condition at border)
    # padarrayXT with 'symmetric' and 'both' pads with 2 on each side
    coeffs_padded = np.pad(coeffs, ((0, 0), (2, 2)), mode='symmetric')
    
    # Interpolate
    ima = np.zeros((ny_coeffs, nx))
    
    for k in range(4):
        # Convert indices to integers and add offset for padding
        indices = (x_index[k, :] + 2).astype(int)
        # Ensure indices are within bounds
        indices = np.clip(indices, 0, coeffs_padded.shape[1] - 1)
        ima += w[k, :][np.newaxis, :] * coeffs_padded[:, indices]
    
    return ima


def ib3spline_2d(
    coeffs: np.ndarray,
    image_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Inverse 2D cubic B-spline interpolation.
    
    Applies 1D B-spline interpolation separably along both dimensions.
    
    Args:
        coeffs: B-spline coefficients, shape (ny_coeffs, nx_coeffs)
        image_size: Target size (ny, nx). If None, uses coeffs.shape
    
    Returns:
        Interpolated image, shape (ny, nx)
    
    Notes:
        - Applies ib3spline_1d along both axes
        - Uses separable interpolation for efficiency
    
    Examples:
        >>> coeffs = np.random.rand(10, 20)
        >>> result = ib3spline_2d(coeffs, (20, 40))
        >>> result.shape
        (20, 40)
    """
    if coeffs.ndim != 2:
        raise ValueError("coeffs must be 2D array")
    
    ny_coeffs, nx_coeffs = coeffs.shape
    
    if image_size is None:
        ny = ny_coeffs
        nx = nx_coeffs
    else:
        if len(image_size) != 2:
            raise ValueError("image_size must be a tuple of (ny, nx)")
        ny, nx = image_size
    
    # Interpolate along x (columns), then along y (rows)
    ima = ib3spline_1d(coeffs, nx)
    ima = ib3spline_1d(ima.T, ny).T
    
    return ima
