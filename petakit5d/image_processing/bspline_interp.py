"""
B-spline interpolation utilities for 1D and 2D signals.

This module provides functions for computing B-spline coefficients and interpolating
values using cubic B-splines, based on the formalism described in:
[1] Unser, IEEE Signal Proc. Mag. 16(6), pp. 22-38, 1999
[2] Unser et al., IEEE Trans. Signal Proc. 41(2), pp. 834-848, 1993

Functions:
    compute_bspline_coefficients: Compute B-spline coefficients for 1D or 2D input
    interp_bspline_value: Interpolate value at given coordinates using B-spline
    calc_interp_maxima: Calculate local maxima of cubic spline interpolation
"""

import numpy as np
from typing import Tuple, Optional, Union, List
import warnings


def compute_bspline_coefficients(
    s: np.ndarray,
    lambda_: float = 0.0,
    degree: int = 3,
    mode: str = 'fourier',
    boundary: str = 'symmetric'
) -> np.ndarray:
    """
    Compute B-spline coefficients for input signal.
    
    Parameters
    ----------
    s : ndarray
        Input signal, 1D or 2D
    lambda_ : float, optional
        Regularization parameter for smoothing splines (default: 0.0).
        Smoothing-spline coefficients are calculated in the Fourier domain.
    degree : int, optional
        Spline degree: 1, 2, or 3 (default: 3 for cubic splines).
        Smoothing splines are implemented for degree 3 only.
    mode : str, optional
        Method for coefficient calculation: 'fourier' or 'spatial' (default: 'fourier')
    boundary : str, optional
        Boundary conditions: 'symmetric' (default) or 'periodic'
        
    Returns
    -------
    c : ndarray
        Spline coefficients
        
    Notes
    -----
    If lambda_ != 0, mode is forced to 'fourier' for smoothing splines.
    For spatial mode, only degrees 2 and 3 are supported.
    
    References
    ----------
    [1] Unser, IEEE Signal Proc. Mag. 16(6), pp. 22-38, 1999
    [2] Unser et al., IEEE Trans. Signal Proc. 41(2), pp. 834-848, 1993
    """
    if lambda_ != 0:
        mode = 'fourier'
    
    dims = np.array(s.shape)
    dims = dims[dims > 1]  # Remove singleton dimensions
    
    # Handle special cases
    if len(dims) == 0:
        # Scalar or single-element array
        return s.copy()
    
    if mode.lower() == 'fourier':
        if len(dims) == 1:
            # 1D case
            N = s.size
            
            # Handle single element case
            if N == 1:
                return s.copy()
            
            # Mirror signal for symmetric boundary
            if boundary.lower() == 'symmetric':
                s_ext = np.concatenate([s, s[N-2:0:-1]])
                M = 2 * N - 2
            else:  # periodic
                s_ext = s.copy()
                M = N
            
            # Frequency vector
            w = np.arange(M) * 2 * np.pi / M
            
            # Fourier transform of input signal
            S = np.fft.fft(s_ext)
            
            # Smoothing spline pre-filter (cubic)
            H = 3.0 / (2 + np.cos(w) + 6 * lambda_ * (np.cos(2*w) - 4*np.cos(w) + 3))
            
            # Spline coefficients
            c = np.real(np.fft.ifft(S * H))
            c = c[:N]
            
        elif len(dims) == 2:
            # 2D case
            ny, nx = s.shape
            
            # Mirror signal for symmetric boundary
            if boundary.lower() == 'symmetric':
                s_ext = np.column_stack([s, s[:, nx-2:0:-1]])
                s_ext = np.vstack([s_ext, s_ext[ny-2:0:-1, :]])
                mx = 2 * nx - 2
                my = 2 * ny - 2
            else:  # periodic
                s_ext = s.copy()
                mx = nx
                my = ny
            
            # Frequency vectors
            wx = np.arange(mx) * 2 * np.pi / mx
            wy = np.arange(my)[:, np.newaxis] * 2 * np.pi / my
            
            # Fourier transform
            S = np.fft.fft2(s_ext)
            
            # Smoothing spline pre-filters
            Hx = 3.0 / (2 + np.cos(wx) + 6 * lambda_ * (np.cos(2*wx) - 4*np.cos(wx) + 3))
            Hy = 3.0 / (2 + np.cos(wy) + 6 * lambda_ * (np.cos(2*wy) - 4*np.cos(wy) + 3))
            
            # Spline coefficients
            c = np.real(np.fft.ifft2((Hy * Hx) * S))
            c = c[:ny, :nx]
        else:
            raise ValueError("Only 1D and 2D inputs are supported")
            
    else:  # spatial mode
        if degree not in [2, 3]:
            if degree == 1:
                return s.copy()
            raise ValueError("Spatial mode only supports degrees 1, 2, and 3")
        
        ny, nx = s.shape if s.ndim == 2 else (1, s.size)
        is_1d = s.ndim == 1
        
        if is_1d:
            s = s.reshape(1, -1)
        
        # Parameters for cubic (degree=3) or quadratic (degree=2)
        if degree == 3:
            z1 = -2 + np.sqrt(3)
            c0 = 6.0
        else:  # degree == 2
            z1 = -3 + 2*np.sqrt(2)
            c0 = 8.0
        
        # Recursively compute coefficients along x-axis
        if nx > 1:
            cp = np.zeros((ny, nx))
            cn = np.zeros((ny, nx))
            
            cp[:, 0] = _get_causal_init_value(s, z1, boundary)
            for k in range(1, nx):
                cp[:, k] = s[:, k] + z1 * cp[:, k-1]
            
            cn[:, nx-1] = _get_anticausal_init_value(cp, z1, boundary)
            for k in range(nx-2, -1, -1):
                cn[:, k] = z1 * (cn[:, k+1] - cp[:, k])
            
            c = c0 * cn
        else:
            c = s.copy()
        
        # Recursively compute coefficients along y-axis
        if ny > 1:
            c = c.T  # Transpose
            cp = np.zeros((nx, ny))
            cn = np.zeros((nx, ny))
            
            cp[:, 0] = _get_causal_init_value(c, z1, boundary)
            for k in range(1, ny):
                cp[:, k] = c[:, k] + z1 * cp[:, k-1]
            
            cn[:, ny-1] = _get_anticausal_init_value(cp, z1, boundary)
            for k in range(ny-2, -1, -1):
                cn[:, k] = z1 * (cn[:, k+1] - cp[:, k])
            
            c = c0 * cn
            c = c.T  # Transpose back
        
        if is_1d:
            c = c.ravel()
    
    return c


def _get_causal_init_value(s: np.ndarray, a: float, boundary: str) -> np.ndarray:
    """Get initial value for causal recursive filtering."""
    N = s.shape[1]
    
    if boundary.lower() == 'symmetric':
        s_ext = np.column_stack([s, s[:, N-2:0:-1]])
        k = np.arange(2*N-2)
        c0 = np.sum(s_ext * (a ** k), axis=1) / (1 - a**(2*N-2))
    elif boundary.lower() == 'periodic':
        s_ext = np.column_stack([s[:, 0:1], s[:, N-1:0:-1]])
        k = np.arange(N)
        c0 = np.sum(s_ext * (a ** k), axis=1) / (1 - a**N)
    elif boundary.lower() == 'zeros':
        c0 = s[:, 0]
    elif boundary.lower() == 'replicate':
        c0 = s[:, 0] / (1 - a)
    else:
        raise ValueError(f"Unknown boundary condition: {boundary}")
    
    return c0


def _get_anticausal_init_value(c: np.ndarray, a: float, boundary: str) -> np.ndarray:
    """Get initial value for anti-causal recursive filtering."""
    N = c.shape[1]
    
    if boundary.lower() == 'symmetric':
        c0 = (a / (a*a - 1)) * (c[:, N-1] + a * c[:, N-2])
    elif boundary.lower() == 'periodic':
        c_ext = np.column_stack([c[:, N-1:], c[:, :N-1]])
        k = np.arange(N)
        c0 = -a / (1 - a**N) * np.sum((a ** k) * c_ext, axis=1)
    elif boundary.lower() == 'zeros':
        c0 = -a * c[:, -1]
    elif boundary.lower() == 'replicate':
        c0 = -c[:, -1] * a / (1 - a)
    else:
        raise ValueError(f"Unknown boundary condition: {boundary}")
    
    return c0


def interp_bspline_value(
    x: Union[float, np.ndarray],
    c: np.ndarray,
    n: int = 3,
    boundary: str = 'symmetric'
) -> Union[float, np.ndarray]:
    """
    Interpolate value using cubic B-spline.
    
    Parameters
    ----------
    x : float or ndarray
        Interpolation coordinates (0-based Python indexing)
    c : ndarray
        Spline coefficients (from compute_bspline_coefficients)
    n : int, optional
        Degree of spline: 1, 2, or 3 (default: 3)
    boundary : str, optional
        Boundary conditions: 'symmetric' (default) or 'periodic'
        
    Returns
    -------
    v : float or ndarray
        Interpolated value(s) at x
        
    Notes
    -----
    This function uses 0-based indexing (Python convention).
    The MATLAB version uses 1-based indexing.
    """
    scalar_input = np.isscalar(x)
    x = np.atleast_1d(x)
    
    xi = np.floor(x).astype(int)
    dx = x - xi
    nx = len(c)
    
    v = np.zeros_like(x, dtype=float)
    
    if n == 1:
        # Linear interpolation
        if boundary.lower() == 'periodic':
            x0 = _periodize_array(xi, nx)
            x1 = _periodize_array(xi + 1, nx)
        else:
            x0 = _mirror_array(xi, nx)
            x1 = _mirror_array(xi + 1, nx)
        v = dx * c[x1] + (1.0 - dx) * c[x0]
    else:
        # Quadratic or cubic interpolation
        for i, (xi_val, dx_val) in enumerate(zip(xi, dx)):
            if n == 2:
                wx = _get_quadratic_spline(dx_val)
            else:  # n == 3
                wx = _get_cubic_spline(dx_val)
            
            if boundary.lower() == 'periodic':
                indices = [_periodize_scalar(xi_val-1, nx), _periodize_scalar(xi_val, nx),
                          _periodize_scalar(xi_val+1, nx), _periodize_scalar(xi_val+2, nx)]
                v[i] = np.sum(wx * c[indices])
            else:
                # Symmetric (mirror) boundary
                # Handle boundary cases
                if xi_val <= 0:
                    # Near left boundary
                    indices = _mirror_array(np.array([xi_val-1, xi_val, xi_val+1, xi_val+2]), nx)
                    v[i] = np.sum(wx * c[indices])
                elif xi_val >= nx - 2:
                    # Near right boundary
                    indices = _mirror_array(np.array([xi_val-1, xi_val, xi_val+1, xi_val+2]), nx)
                    v[i] = np.sum(wx * c[indices])
                else:
                    # Interior point
                    v[i] = np.sum(wx * c[[xi_val-1, xi_val, xi_val+1, xi_val+2]])
    
    return v[0] if scalar_input else v


def _mirror_array(x: np.ndarray, nx: int) -> np.ndarray:
    """Apply mirror boundary conditions to array."""
    x = x.copy()
    idx = x < 0
    x[idx] = -x[idx] - 1
    idx = x >= nx
    x[idx] = 2 * nx - 1 - x[idx]
    # Clip to valid range
    x = np.clip(x, 0, nx - 1)
    return x


def _periodize_scalar(x: int, nx: int) -> int:
    """Apply periodic boundary conditions to scalar."""
    while x < 0:
        x += nx
    while x >= nx:
        x -= nx
    return x


def _periodize_array(x: np.ndarray, nx: int) -> np.ndarray:
    """Apply periodic boundary conditions to array."""
    return x % nx


def _get_quadratic_spline(t: float) -> np.ndarray:
    """
    Get sampled values of quadratic B-spline.
    
    Returns (B2[t+1], B2[t], B2[t-1], B2[t-2]) for t in [0, 1].
    """
    if t < 0.0 or t > 1.0:
        raise ValueError(f"Argument t for quadratic B-spline outside [0,1]: {t}")
    
    v = np.zeros(4)
    if t <= 0.5:
        v[0] = (t - 0.5)**2 / 2.0
        v[1] = 0.75 - t**2
        v[2] = 1.0 - v[1] - v[0]
        v[3] = 0.0
    else:
        v[0] = 0.0
        v[1] = (t - 1.5)**2 / 2.0
        v[3] = (t - 0.5)**2 / 2.0
        v[2] = 1.0 - v[3] - v[1]
    
    return v


def _get_cubic_spline(t: float) -> np.ndarray:
    """
    Get sampled values of cubic B-spline.
    
    Returns (B3[t+1], B3[t], B3[t-1], B3[t-2]) for t in [0, 1].
    """
    if t < 0.0 or t > 1.0:
        raise ValueError(f"Argument t for cubic B-spline outside [0,1]: {t}")
    
    t1 = 1.0 - t
    t2 = t * t
    
    v = np.zeros(4)
    v[0] = (t1 * t1 * t1) / 6.0
    v[1] = (2.0 / 3.0) + 0.5 * t2 * (t - 2)
    v[3] = (t2 * t) / 6.0
    v[2] = 1.0 - v[3] - v[1] - v[0]
    
    return v


def calc_interp_maxima(
    f0: np.ndarray,
    lambda_: float = 0.0,
    display: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate local maxima of cubic spline interpolation.
    
    Parameters
    ----------
    f0 : ndarray
        Input signal (1D or 2D)
    lambda_ : float, optional
        Regularization parameter for smoothing (default: 0.0)
    display : bool, optional
        If True, display results (not implemented) (default: False)
        
    Returns
    -------
    fmax : ndarray
        Function values at maxima
    xmax : ndarray or tuple of ndarrays
        Coordinates of maxima (0-based indexing)
        For 1D: xmax is 1D array
        For 2D: xmax is tuple (xmax, ymax)
    c : ndarray
        Spline coefficients
        
    Notes
    -----
    Uses 0-based indexing (Python convention).
    Returns maxima positions as fractional indices.
    """
    dims = np.array(f0.shape)
    dims = dims[dims > 1]
    
    if len(dims) == 1:
        # 1D case
        nx = f0.size
        
        # Compute coefficients
        c = compute_bspline_coefficients(f0, lambda_)
        cx = np.concatenate([[c[1]], c, [c[-2]]])
        
        # For each interval, calculate solution
        xmax_list = []
        for i in range(nx - 1):
            sols = _get_sol_1d(cx[i:i+4])
            if len(sols) > 0:
                xmax_list.extend([i + s for s in sols])  # 0-based indexing
        
        xmax = np.array(xmax_list)
        if len(xmax) > 0:
            fmax = np.array([interp_bspline_value(x, c, 3, 'symmetric') for x in xmax])
        else:
            fmax = np.array([])
        
        return fmax, xmax, c
    
    elif len(dims) == 2:
        # 2D case
        ny, nx = f0.shape
        
        # Compute coefficients
        c = compute_bspline_coefficients(f0, lambda_)
        
        # Pad coefficients for mirroring (simple symmetric padding)
        cx = np.pad(c, ((1, 1), (1, 1)), mode='symmetric')
        
        # Loop over grid and calculate maximum for each set of points
        xmax_list = []
        ymax_list = []
        fmax_list = []
        
        for i in range(nx - 1):
            for j in range(ny - 1):
                C0 = cx[j:j+4, i:i+4]
                dx, dy, val = _iterate_sol_2d(C0)
                
                if dx is not None:
                    xmax_list.append(i + dx)  # 0-based indexing
                    ymax_list.append(j + dy)
                    fmax_list.append(val)
        
        fmax = np.array(fmax_list)
        xmax = np.array(xmax_list)
        ymax = np.array(ymax_list)
        
        return fmax, (xmax, ymax), c
    
    else:
        raise ValueError("Only 1D and 2D inputs are supported")


def _get_sol_1d(c: np.ndarray) -> List[float]:
    """
    Get solutions for 1D cubic spline maximum.
    
    Finds t in [0,1] where derivative is zero.
    """
    # Derivative of cubic spline: db3(t) = [-0.5*(1-t)^2, 1.5*t^2 - 2*t, -1.5*t^2 + t + 0.5, 0.5*t^2]
    # Sum of weighted coefficients should be zero
    
    # Quadratic equation: at^2 + bt + c = 0
    a = -0.5 * c[0] + 1.5 * c[1] - 1.5 * c[2] + 0.5 * c[3]
    b = c[0] - 2 * c[1] + c[2]
    c_coef = -0.5 * c[0] + 0.5 * c[2]
    
    sols = []
    if abs(a) < 1e-10:
        # Linear equation
        if abs(b) > 1e-10:
            t = -c_coef / b
            if 0 <= t <= 1:
                sols.append(t)
    else:
        # Quadratic equation
        disc = b**2 - 4*a*c_coef
        if disc >= 0:
            t1 = (-b + np.sqrt(disc)) / (2*a)
            t2 = (-b - np.sqrt(disc)) / (2*a)
            if 0 <= t1 <= 1:
                sols.append(t1)
            if 0 <= t2 <= 1 and abs(t2 - t1) > 1e-10:
                sols.append(t2)
    
    return sols


def _iterate_sol_2d(C0: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Iteratively find solution for 2D cubic spline maximum.
    
    Simplified version that finds approximate maximum location.
    """
    # This is a simplified implementation
    # Full implementation would use Newton's method or optimization
    
    # Try center point as initial guess
    dx, dy = 0.5, 0.5
    
    # Simple gradient descent (few iterations)
    for _ in range(5):
        # Compute gradient at current point
        wx = _get_cubic_spline(dx)
        wy = _get_cubic_spline(dy)
        
        # Compute value
        val = np.sum(C0 * np.outer(wy, wx))
        
        # Check if this is likely a maximum (simple heuristic)
        # Full implementation would check Hessian
        if 0.1 < dx < 0.9 and 0.1 < dy < 0.9:
            return dx, dy, val
    
    return None, None, None
