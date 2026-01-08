"""
Masking utilities.

Ported from MATLAB imageProcessing/ directory.
"""

import numpy as np
from typing import Union


def mask_vectors(
    disp_mat_x: np.ndarray,
    disp_mat_y: np.ndarray,
    bw_stack_img: np.ndarray
) -> np.ndarray:
    """
    Determine which displacement vectors fall inside a binary mask.
    
    Args:
        disp_mat_x: X coordinates of displacement vectors
        disp_mat_y: Y coordinates of displacement vectors
        bw_stack_img: Binary mask image (2D array)
        
    Returns:
        np.ndarray: Boolean array indicating which vectors are inside the mask
        
    Examples:
        >>> x_coords = np.array([10, 20, 30])
        >>> y_coords = np.array([15, 25, 35])
        >>> mask = np.ones((50, 50), dtype=bool)
        >>> inside = mask_vectors(x_coords, y_coords, mask)
        
    Original MATLAB function: maskVectors.m
    """
    n_points = len(disp_mat_x)
    inside_idx = np.zeros(n_points, dtype=bool)
    
    for ii in range(n_points):
        # Round coordinates to integers
        y_idx = int(round(disp_mat_y[ii]))
        x_idx = int(round(disp_mat_x[ii]))
        
        # Check if coordinates are within bounds and mask is True
        if (0 < y_idx <= bw_stack_img.shape[0] and
            0 < x_idx <= bw_stack_img.shape[1]):
            # Convert to 0-based indexing
            if bw_stack_img[y_idx - 1, x_idx - 1]:
                inside_idx[ii] = True
    
    return inside_idx
