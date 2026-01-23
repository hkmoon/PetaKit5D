"""
Stitching utilities for microscopy image processing.

This module provides functions for image stitching operations including
feather blending and distance map manipulation.
"""

import numpy as np
from scipy.ndimage import zoom
from typing import Tuple

try:
    from ..utils.power_utils import fast_power
except ImportError:
    # Fallback if power_utils not available
    def fast_power(x, n):
        return np.power(x, n)


def feather_distance_map_resize_3d(
    dmat: np.ndarray,
    bbox: Tuple[int, int, int, int, int, int],
    wd: float
) -> np.ndarray:
    """
    Resize feather blending distance map for 3D image stitching.
    
    This function resizes a distance map used for feather blending in image stitching.
    The distance map is resized to match the bounding box dimensions and then raised
    to a power for smooth blending.
    
    Parameters
    ----------
    dmat : np.ndarray
        Distance map as a 2D or 3D array. Values typically in range [0, 1].
    bbox : tuple of int
        Bounding box as (y_start, x_start, z_start, y_end, x_end, z_end).
        Uses 1-based MATLAB indexing (will be converted internally).
    wd : float
        Weighting power for feather blending. Higher values create sharper transitions.
        Typical values are 1-4.
        
    Returns
    -------
    dmat : np.ndarray
        Resized and weighted distance map matching the bounding box dimensions.
        
    Notes
    -----
    This function attempts to use a MEX implementation (feather_distance_map_resize_3d_mex)
    if available, otherwise falls back to scipy-based resizing. The distance map is
    raised to the power `wd` after resizing for smooth feather blending.
    
    The bounding box uses MATLAB 1-based indexing, so bbox = (1, 1, 1, 100, 100, 50)
    means a volume of size (100, 100, 50).
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create a simple distance map
    >>> dmat = np.ones((50, 50))
    >>> # Resize to bounding box
    >>> bbox = (1, 1, 1, 100, 100, 1)  # 100x100x1 volume
    >>> wd = 2.0
    >>> resized = feather_distance_map_resize_3d(dmat, bbox, wd)
    >>> resized.shape
    (100, 100, 1)
    """
    # Convert bbox from 1-based to 0-based and compute target size
    # MATLAB: bbox(4:6) - bbox(1:3) + 1
    target_size = (
        bbox[3] - bbox[0] + 1,  # y size
        bbox[4] - bbox[1] + 1,  # x size
        bbox[5] - bbox[2] + 1   # z size
    )
    
    # Handle 2D case (single z-slice)
    if bbox[2] == bbox[5]:  # z_start == z_end
        # Resize only in 2D
        if dmat.ndim == 3:
            dmat = dmat[:, :, 0]
        
        # Compute zoom factors
        zoom_factors = (
            target_size[0] / dmat.shape[0],
            target_size[1] / dmat.shape[1]
        )
        
        # Use bilinear interpolation (order=1)
        dmat = zoom(dmat, zoom_factors, order=1)
        
        # Add z dimension back
        dmat = dmat[:, :, np.newaxis]
    else:
        # Handle 3D case
        if dmat.ndim == 2:
            # If input is 2D but output should be 3D, replicate
            dmat = np.repeat(dmat[:, :, np.newaxis], 2, axis=2)
        elif dmat.shape[2] == 1 and target_size[2] > 1:
            # If single z-slice but multiple needed, replicate
            dmat = np.repeat(dmat, 2, axis=2)
        
        # Compute zoom factors
        zoom_factors = (
            target_size[0] / dmat.shape[0],
            target_size[1] / dmat.shape[1],
            target_size[2] / dmat.shape[2]
        )
        
        # Use trilinear interpolation (order=1)
        dmat = zoom(dmat, zoom_factors, order=1)
    
    # Apply power weighting for feather blending
    # Use numpy's power for float exponents, fast_power for integers
    if isinstance(wd, (int, np.integer)):
        dmat = fast_power(dmat, wd)
    else:
        dmat = np.power(dmat, wd)
    
    return dmat
