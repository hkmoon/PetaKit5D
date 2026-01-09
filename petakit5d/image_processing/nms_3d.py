"""
3D Non-Maximum Suppression for vector fields.

This module implements non-maximum suppression (NMS) on 3D vector-valued data,
finding pixels where the magnitude of the value is a local maximum in the 
direction of that pixel's vector.

Author: Converted from MATLAB (Hunter Elliott, 9/2011)
Date: 2026-01-09
"""

import numpy as np
from scipy import ndimage
from typing import Tuple


def non_maximum_suppression_3d(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Perform non-maximum suppression (NMS) on a 3D vector image.
    
    This function performs non-maximum suppression on 3D vector-valued data.
    Pixels are retained where the magnitude of the value is a local maximum
    in the direction of that pixel's vector.
    
    Parameters
    ----------
    u : np.ndarray
        3D array specifying the X component of the data (corresponds to dim 1)
    v : np.ndarray
        3D array specifying the Y component of the data (corresponds to dim 0)
    w : np.ndarray
        3D array specifying the Z component of the data (corresponds to dim 2)
        
    Returns
    -------
    np.ndarray
        3D array with zeros in non-maximum areas and the magnitude
        of the input data at the maxima
        
    Notes
    -----
    The u, v, w components are assumed to correspond to dimensions 1, 0, 2
    of the input matrices respectively, matching MATLAB's convention where
    the first dimension is Y (rows) and second is X (columns).
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create a simple 3D vector field
    >>> shape = (50, 50, 20)
    >>> u = np.random.rand(*shape)
    >>> v = np.random.rand(*shape)
    >>> w = np.random.rand(*shape)
    >>> nms = non_maximum_suppression_3d(u, v, w)
    >>> nms.shape
    (50, 50, 20)
    """
    if u.ndim != 3 or v.ndim != 3 or w.ndim != 3:
        raise ValueError("All inputs u, v, and w must be 3-dimensional matrices")
    
    if not (u.shape == v.shape == w.shape):
        raise ValueError("All inputs u, v, and w must have equal size")
    
    M, N, P = u.shape
    
    # Calculate the magnitude of the vector field at each point
    nms = np.sqrt(u**2 + v**2 + w**2)
    
    # Use magnitude to normalize the vector data
    # Avoid division by zero
    nms_safe = np.where(nms == 0, 1, nms)
    u_norm = u / nms_safe
    v_norm = v / nms_safe
    w_norm = w / nms_safe
    
    # Set NaN values to 0
    u_norm = np.nan_to_num(u_norm, nan=0.0)
    v_norm = np.nan_to_num(v_norm, nan=0.0)
    w_norm = np.nan_to_num(w_norm, nan=0.0)
    
    # Pad array so we can detect maxima at the edges
    # Using 'symmetric' mode to match MATLAB's padarrayXT behavior
    nms_pad = np.pad(nms, ((1, 1), (1, 1), (1, 1)), mode='symmetric')
    
    # Create coordinate grids for interpolation
    # Note: meshgrid indexing='xy' means X corresponds to columns (dim 1),
    # Y to rows (dim 0), matching MATLAB convention
    Y, X, Z = np.mgrid[0:M, 0:N, 0:P]
    Yp, Xp, Zp = np.mgrid[-1:M+1, -1:N+1, -1:P+1]
    
    # Coordinates for interpolation in the padded array
    # We add 1 to account for the padding
    Y_interp = Y + v_norm
    X_interp = X + u_norm
    Z_interp = Z + w_norm
    
    # Get interpolated values 1 pixel in the direction parallel to local orientation
    mag1 = ndimage.map_coordinates(
        nms_pad,
        [Y_interp + 1, X_interp + 1, Z_interp + 1],  # +1 for padding offset
        order=1,  # Linear interpolation
        mode='nearest'
    )
    
    # Get interpolated values 1 pixel in the opposite direction
    Y_interp_neg = Y - v_norm
    X_interp_neg = X - u_norm
    Z_interp_neg = Z - w_norm
    
    mag2 = ndimage.map_coordinates(
        nms_pad,
        [Y_interp_neg + 1, X_interp_neg + 1, Z_interp_neg + 1],  # +1 for padding offset
        order=1,  # Linear interpolation
        mode='nearest'
    )
    
    # Remove pixels which are not greater than both interpolated values
    nms[(nms <= mag1) | (nms <= mag2)] = 0
    
    return nms
