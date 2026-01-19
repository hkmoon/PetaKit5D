"""
B-spline interpolation with derivatives.

This module provides cubic spline interpolation for 1D and 2D signals with
computation of first and second derivatives at interpolation points.

Based on MATLAB implementation by Francois Aguet (May 2012).

References:
[1] Unser, IEEE Signal Proc. Mag. 16(6), pp. 22-38, 1999
[2] Unser et al., IEEE Trans. Signal Proc. 41(2), pp. 834-848, 1993
"""

import numpy as np
from typing import Tuple, Union, Literal, Optional
from .bspline_coeffs import b3spline_1d, b3spline_2d


def _cubic_bspline_basis(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute cubic B-spline basis functions and their derivatives.
    
    Parameters
    ----------
    x : np.ndarray
        Distance from knot, in range [0, 4] for 4-point support
        
    Returns
    -------
    b : np.ndarray
        Basis function values
    db : np.ndarray
        First derivative values
    d2b : np.ndarray
        Second derivative values
    """
    # Cubic B-spline basis
    x = np.abs(x)
    
    # Initialize
    b = np.zeros_like(x)
    db = np.zeros_like(x)
    d2b = np.zeros_like(x)
    
    # Region 0 <= |x| < 1
    mask0 = x < 1
    if np.any(mask0):
        x0 = x[mask0]
        b[mask0] = (4.0 - 6.0 * x0**2 + 3.0 * x0**3) / 6.0
        db[mask0] = (-12.0 * x0 + 9.0 * x0**2) / 6.0
        d2b[mask0] = (-12.0 + 18.0 * x0) / 6.0
    
    # Region 1 <= |x| < 2
    mask1 = (x >= 1) & (x < 2)
    if np.any(mask1):
        x1 = x[mask1]
        b[mask1] = (2.0 - x1)**3 / 6.0
        db[mask1] = -(2.0 - x1)**2 / 2.0
        d2b[mask1] = (2.0 - x1)
    
    # Handle sign for derivatives (derivative w.r.t. original signed x)
    # For negative x, flip sign of odd derivatives
    
    return b, db, d2b


def binterp_1d(
    f: np.ndarray,
    xi: np.ndarray,
    border_condition: Literal['mirror', 'periodic'] = 'mirror'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cubic spline interpolation for 1D signal with derivatives.
    
    Parameters
    ----------
    f : np.ndarray
        Input 1D signal
    xi : np.ndarray
        Interpolation coordinates (0-based indexing)
    border_condition : {'mirror', 'periodic'}, default='mirror'
        Boundary conditions
        
    Returns
    -------
    fi : np.ndarray
        Interpolated values
    fi_dx : np.ndarray
        First derivative at interpolation points
    fi_d2x : np.ndarray
        Second derivative at interpolation points
    """
    f = np.atleast_1d(f).astype(np.float64)
    xi = np.atleast_1d(xi).astype(np.float64)
    
    # Compute B-spline coefficients
    if f.ndim == 1:
        f_2d = f.reshape(1, -1)
    else:
        f_2d = f
        
    coeffs = b3spline_1d(f_2d, boundary=border_condition)
    
    n = coeffs.shape[1]
    
    # Initialize outputs
    fi = np.zeros_like(xi)
    fi_dx = np.zeros_like(xi)
    fi_d2x = np.zeros_like(xi)
    
    # For each interpolation point
    for idx in range(len(xi)):
        x = xi[idx]
        
        # Find the 4 neighboring coefficients (support of cubic B-spline)
        x_floor = int(np.floor(x))
        dx = x - x_floor
        
        # Get 4-point support indices
        indices = np.array([x_floor - 1, x_floor, x_floor + 1, x_floor + 2])
        
        # Apply boundary conditions
        if border_condition == 'mirror':
            # Mirror boundary
            indices = np.abs(indices)
            indices[indices >= n] = 2 * n - indices[indices >= n] - 2
            indices = np.clip(indices, 0, n - 1)
        else:  # periodic
            indices = indices % n
        
        # Get coefficient values
        c = coeffs[0, indices]
        
        # Compute basis functions and derivatives at distance from each knot
        distances = np.array([dx + 1, dx, dx - 1, dx - 2])
        b, db, d2b = _cubic_bspline_basis(distances)
        
        # Apply sign correction for derivatives
        signs = np.sign(distances)
        signs[signs == 0] = 1
        db = db * signs
        
        # Interpolate
        fi[idx] = np.sum(c * b)
        fi_dx[idx] = np.sum(c * db)
        fi_d2x[idx] = np.sum(c * d2b)
    
    return fi, fi_dx, fi_d2x


def binterp_2d(
    f: np.ndarray,
    xi: np.ndarray,
    yi: np.ndarray,
    border_condition: Literal['mirror', 'periodic'] = 'mirror'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cubic spline interpolation for 2D image with derivatives.
    
    Parameters
    ----------
    f : np.ndarray
        Input 2D image
    xi, yi : np.ndarray
        Interpolation coordinates (must be same shape, 0-based indexing)
    border_condition : {'mirror', 'periodic'}, default='mirror'
        Boundary conditions
        
    Returns
    -------
    fi : np.ndarray
        Interpolated values
    fi_dx : np.ndarray
        Partial derivative w.r.t. x
    fi_dy : np.ndarray
        Partial derivative w.r.t. y
    fi_d2x : np.ndarray
        Second partial derivative w.r.t. x
    fi_d2y : np.ndarray
        Second partial derivative w.r.t. y
    """
    f = f.astype(np.float64)
    xi = np.atleast_1d(xi).astype(np.float64)
    yi = np.atleast_1d(yi).astype(np.float64)
    
    if xi.shape != yi.shape:
        raise ValueError("xi and yi must have the same shape")
    
    # Compute 2D B-spline coefficients
    coeffs = b3spline_2d(f, boundary=border_condition)
    
    ny, nx = coeffs.shape
    
    # Initialize outputs
    output_shape = xi.shape
    fi = np.zeros(output_shape)
    fi_dx = np.zeros(output_shape)
    fi_dy = np.zeros(output_shape)
    fi_d2x = np.zeros(output_shape)
    fi_d2y = np.zeros(output_shape)
    
    # Flatten for iteration
    xi_flat = xi.ravel()
    yi_flat = yi.ravel()
    
    fi_flat = fi.ravel()
    fi_dx_flat = fi_dx.ravel()
    fi_dy_flat = fi_dy.ravel()
    fi_d2x_flat = fi_d2x.ravel()
    fi_d2y_flat = fi_d2y.ravel()
    
    # For each interpolation point
    for idx in range(len(xi_flat)):
        x = xi_flat[idx]
        y = yi_flat[idx]
        
        # Find the 4x4 neighboring coefficients
        x_floor = int(np.floor(x))
        y_floor = int(np.floor(y))
        dx = x - x_floor
        dy = y - y_floor
        
        # Get 4-point support indices for x and y
        x_indices = np.array([x_floor - 1, x_floor, x_floor + 1, x_floor + 2])
        y_indices = np.array([y_floor - 1, y_floor, y_floor + 1, y_floor + 2])
        
        # Apply boundary conditions
        if border_condition == 'mirror':
            # Mirror boundary for x
            x_indices = np.abs(x_indices)
            x_indices[x_indices >= nx] = 2 * nx - x_indices[x_indices >= nx] - 2
            x_indices = np.clip(x_indices, 0, nx - 1)
            
            # Mirror boundary for y
            y_indices = np.abs(y_indices)
            y_indices[y_indices >= ny] = 2 * ny - y_indices[y_indices >= ny] - 2
            y_indices = np.clip(y_indices, 0, ny - 1)
        else:  # periodic
            x_indices = x_indices % nx
            y_indices = y_indices % ny
        
        # Get coefficient patch (4x4)
        c_patch = coeffs[np.ix_(y_indices, x_indices)]
        
        # Compute basis functions for x direction
        x_distances = np.array([dx + 1, dx, dx - 1, dx - 2])
        bx, dbx, d2bx = _cubic_bspline_basis(x_distances)
        x_signs = np.sign(x_distances)
        x_signs[x_signs == 0] = 1
        dbx = dbx * x_signs
        
        # Compute basis functions for y direction
        y_distances = np.array([dy + 1, dy, dy - 1, dy - 2])
        by, dby, d2by = _cubic_bspline_basis(y_distances)
        y_signs = np.sign(y_distances)
        y_signs[y_signs == 0] = 1
        dby = dby * y_signs
        
        # 2D interpolation using outer products
        # f(x,y) = sum_i sum_j c_ij * b_i(x) * b_j(y)
        fi_flat[idx] = np.sum(c_patch * np.outer(by, bx))
        fi_dx_flat[idx] = np.sum(c_patch * np.outer(by, dbx))
        fi_dy_flat[idx] = np.sum(c_patch * np.outer(dby, bx))
        fi_d2x_flat[idx] = np.sum(c_patch * np.outer(by, d2bx))
        fi_d2y_flat[idx] = np.sum(c_patch * np.outer(d2by, bx))
    
    # Reshape outputs
    fi = fi_flat.reshape(output_shape)
    fi_dx = fi_dx_flat.reshape(output_shape)
    fi_dy = fi_dy_flat.reshape(output_shape)
    fi_d2x = fi_d2x_flat.reshape(output_shape)
    fi_d2y = fi_d2y_flat.reshape(output_shape)
    
    return fi, fi_dx, fi_dy, fi_d2x, fi_d2y


def binterp(
    f: np.ndarray,
    *args,
    border_condition: Literal['mirror', 'periodic'] = 'mirror'
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Cubic spline interpolation with derivatives (1D or 2D).
    
    This function provides a unified interface for cubic B-spline interpolation
    of 1D or 2D signals, returning interpolated values along with first and
    second derivatives.
    
    Parameters
    ----------
    f : np.ndarray
        Input signal (1D) or image (2D)
    *args : np.ndarray
        For 1D: xi (interpolation coordinates)
        For 2D: xi, yi (interpolation coordinates, must be same shape)
    border_condition : {'mirror', 'periodic'}, default='mirror'
        Boundary conditions for coefficient computation
        
    Returns
    -------
    For 1D:
        fi : np.ndarray
            Interpolated signal
        fi_dx : np.ndarray
            First derivative
        fi_d2x : np.ndarray
            Second derivative
            
    For 2D:
        fi : np.ndarray
            Interpolated image
        fi_dx, fi_dy : np.ndarray
            First partial derivatives
        fi_d2x, fi_d2y : np.ndarray
            Second partial derivatives
    
    Notes
    -----
    Coordinates are 0-based (Python convention), unlike MATLAB which uses 1-based.
    
    For more information, see:
    [1] Unser, IEEE Signal Proc. Mag. 16(6), pp. 22-38, 1999
    [2] Unser et al., IEEE Trans. Signal Proc. 41(2), pp. 834-848, 1993
    
    Examples
    --------
    >>> # 1D interpolation
    >>> signal = np.sin(np.linspace(0, 2*np.pi, 10))
    >>> xi = np.array([1.5, 3.7, 5.2])
    >>> fi, fi_dx, fi_d2x = binterp(signal, xi)
    
    >>> # 2D interpolation
    >>> image = np.random.rand(20, 20)
    >>> xi = np.array([5.5, 10.3])
    >>> yi = np.array([7.2, 15.8])
    >>> fi, fi_dx, fi_dy, fi_d2x, fi_d2y = binterp(image, xi, yi)
    """
    f = np.asarray(f)
    
    if f.ndim == 1 or (f.ndim == 2 and len(args) == 1):
        # 1D case
        if len(args) != 1:
            raise ValueError("For 1D interpolation, provide f and xi")
        xi = args[0]
        return binterp_1d(f, xi, border_condition)
    
    elif f.ndim == 2 and len(args) == 2:
        # 2D case
        xi, yi = args
        return binterp_2d(f, xi, yi, border_condition)
    
    else:
        raise ValueError(
            "Invalid input dimensions. "
            "For 1D: binterp(f, xi). For 2D: binterp(f, xi, yi)"
        )
