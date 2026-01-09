"""
B-spline coefficient computation for cubic B-spline interpolation.

This module provides functions to compute cubic B-spline coefficients
for 1D and 2D images, which are required for B-spline interpolation.

Author: Converted from MATLAB (Francois Aguet, June 2010)
Date: 2026-01-09
"""

import numpy as np
from typing import Literal


def b3spline_1d(
    img: np.ndarray,
    boundary: Literal['mirror', 'periodic'] = 'mirror'
) -> np.ndarray:
    """
    Compute 1D cubic B-spline coefficients.
    
    This function computes the cubic B-spline coefficients for an input image
    along the second dimension (columns). These coefficients are used for
    B-spline interpolation.
    
    Parameters
    ----------
    img : np.ndarray
        Input image of shape (M, N) or 1D array of shape (N,).
    boundary : {'mirror', 'periodic'}, default='mirror'
        Boundary conditions for coefficient computation:
        - 'mirror': Mirror boundary conditions (symmetric extension)
        - 'periodic': Periodic boundary conditions (wraparound)
    
    Returns
    -------
    np.ndarray
        B-spline coefficients with same shape as input.
    
    Raises
    ------
    ValueError
        If boundary is not 'mirror' or 'periodic'.
    
    Notes
    -----
    The cubic B-spline parameters are:
    - c0 = 6
    - z1 = -2 + sqrt(3) ≈ -0.2679
    
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(10, 20)
    >>> coeffs = b3spline_1d(img, boundary='mirror')
    >>> coeffs.shape
    (10, 20)
    """
    if boundary not in ['mirror', 'periodic']:
        raise ValueError("Boundary must be 'mirror' or 'periodic'")
    
    # Handle 1D input
    if img.ndim == 1:
        img = img.reshape(1, -1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Cubic spline parameters
    c0 = 6.0
    z1 = -2.0 + np.sqrt(3.0)
    
    N = img.shape[1]
    cp = np.zeros_like(img, dtype=np.float64)
    cn = np.zeros_like(img, dtype=np.float64)
    
    if boundary == 'mirror':
        # Causal filter initialization (mirror boundary)
        cp[:, 0] = _get_causal_init_mirror(img, z1)
        for k in range(1, N):
            cp[:, k] = img[:, k] + z1 * cp[:, k - 1]
        
        # Anti-causal filter initialization (mirror boundary)
        cn[:, N - 1] = _get_anticausal_init_mirror(cp, z1)
        for k in range(N - 2, -1, -1):
            cn[:, k] = z1 * (cn[:, k + 1] - cp[:, k])
    
    elif boundary == 'periodic':
        # Causal filter initialization (periodic boundary)
        cp[:, 0] = _get_causal_init_periodic(img, z1)
        for k in range(1, N):
            cp[:, k] = img[:, k] + z1 * cp[:, k - 1]
        
        # Anti-causal filter initialization (periodic boundary)
        cn[:, N - 1] = _get_anticausal_init_periodic(cp, z1)
        for k in range(N - 2, -1, -1):
            cn[:, k] = z1 * (cn[:, k + 1] - cp[:, k])
    
    # Scale by c0
    cn = c0 * cn
    
    if squeeze_output:
        cn = cn.squeeze()
    
    return cn


def b3spline_2d(
    img: np.ndarray,
    boundary: Literal['mirror', 'periodic'] = 'mirror'
) -> np.ndarray:
    """
    Compute 2D cubic B-spline coefficients.
    
    This function computes the cubic B-spline coefficients for a 2D image
    by applying 1D B-spline coefficient computation separably along both
    dimensions.
    
    Parameters
    ----------
    img : np.ndarray
        Input 2D image of shape (M, N).
    boundary : {'mirror', 'periodic'}, default='mirror'
        Boundary conditions for coefficient computation:
        - 'mirror': Mirror boundary conditions (symmetric extension)
        - 'periodic': Periodic boundary conditions (wraparound)
    
    Returns
    -------
    np.ndarray
        B-spline coefficients with same shape as input.
    
    Raises
    ------
    ValueError
        If boundary is not 'mirror' or 'periodic'.
        If img is not a 2D array.
    
    Notes
    -----
    This function applies b3spline_1d separably:
    1. First along columns (dimension 1)
    2. Then along rows (dimension 0)
    
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(100, 100)
    >>> coeffs = b3spline_2d(img, boundary='mirror')
    >>> coeffs.shape
    (100, 100)
    """
    if img.ndim != 2:
        raise ValueError("Input must be a 2D array")
    
    # Apply B-spline computation along columns
    c = b3spline_1d(img, boundary=boundary)
    
    # Apply B-spline computation along rows
    c = b3spline_1d(c.T, boundary=boundary).T
    
    return c


def _get_anticausal_init_mirror(img: np.ndarray, a: float) -> np.ndarray:
    """
    Compute anti-causal filter initialization for mirror boundary.
    
    Parameters
    ----------
    img : np.ndarray
        Input causal filtered image.
    a : float
        Filter parameter z1.
    
    Returns
    -------
    np.ndarray
        Initial value for anti-causal filter.
    """
    N = img.shape[1]
    c0 = (a / (a * a - 1.0)) * (img[:, N - 1] + a * img[:, N - 2])
    return c0


def _get_anticausal_init_periodic(img: np.ndarray, a: float) -> np.ndarray:
    """
    Compute anti-causal filter initialization for periodic boundary.
    
    Parameters
    ----------
    img : np.ndarray
        Input causal filtered image.
    a : float
        Filter parameter z1.
    
    Returns
    -------
    np.ndarray
        Initial value for anti-causal filter.
    """
    N = img.shape[1]
    # Construct periodic extension: [img[:, N-1], img, img[:, 0:N-3]]
    img_ext = np.concatenate([
        img[:, N - 1:N],
        img,
        img[:, 0:N - 3]
    ], axis=1)
    
    # Compute weighted sum
    k = np.arange(2 * N - 2)
    a_powers = a ** k
    c0 = -a / (1 - a ** N) * np.sum(a_powers * img_ext, axis=1)
    
    return c0


def _get_causal_init_mirror(img: np.ndarray, a: float) -> np.ndarray:
    """
    Compute causal filter initialization for mirror boundary.
    
    Parameters
    ----------
    img : np.ndarray
        Input image.
    a : float
        Filter parameter z1.
    
    Returns
    -------
    np.ndarray
        Initial value for causal filter.
    """
    N = img.shape[1]
    # Construct mirror extension: [img, img[:, N-2:0:-1]]
    img_mirror = np.concatenate([img, img[:, N - 2:0:-1]], axis=1)
    
    # Compute weighted sum
    k = np.arange(2 * N - 2)
    a_powers = a ** k
    out = np.sum(img_mirror * a_powers, axis=1) / (1 - a ** (2 * N - 2))
    
    return out


def _get_causal_init_periodic(img: np.ndarray, a: float) -> np.ndarray:
    """
    Compute causal filter initialization for periodic boundary.
    
    Parameters
    ----------
    img : np.ndarray
        Input image.
    a : float
        Filter parameter z1.
    
    Returns
    -------
    np.ndarray
        Initial value for causal filter.
    """
    N = img.shape[1]
    # Construct periodic extension: [img[:, 0], img[:, N-1:1:-1]]
    img_ext = np.concatenate([
        img[:, 0:1],
        img[:, N - 1:0:-1]
    ], axis=1)
    
    # Compute weighted sum
    k = np.arange(N)
    a_powers = a ** k
    out = np.sum(img_ext * a_powers, axis=1) / (1 - a ** N)
    
    return out
