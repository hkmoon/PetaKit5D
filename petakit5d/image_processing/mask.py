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


def angle_filter(
    vec_x: np.ndarray,
    vec_y: np.ndarray,
    vec_mid: np.ndarray
) -> np.ndarray:
    """
    Filter vectors based on angle from a reference direction.
    
    Returns True for vectors within -pi/3 to pi/3 radians from vec_mid direction.
    
    Args:
        vec_x: X components of vectors to filter
        vec_y: Y components of vectors to filter
        vec_mid: Reference direction vector [x, y]
        
    Returns:
        np.ndarray: Boolean array indicating which vectors pass the angle filter
        
    Examples:
        >>> x = np.array([1, 0, -1])
        >>> y = np.array([0, 1, 0])
        >>> ref = np.array([1, 0])
        >>> in_angle = angle_filter(x, y, ref)
        
    Original MATLAB function: angleFilter.m
    """
    n_points = len(vec_x)
    in_ang_idx = np.zeros(n_points, dtype=bool)
    
    # Normalize reference vector
    vec_mid = np.array(vec_mid)
    vec_mid_norm = np.linalg.norm(vec_mid)
    
    for ii in range(n_points):
        vec = np.array([vec_x[ii], vec_y[ii]])
        vec_norm = np.linalg.norm(vec)
        
        if vec_norm > 0 and vec_mid_norm > 0:
            # Calculate angle using dot product
            cos_ang = np.dot(vec, vec_mid) / (vec_norm * vec_mid_norm)
            # Clamp to [-1, 1] to avoid numerical issues with arccos
            cos_ang = np.clip(cos_ang, -1.0, 1.0)
            ang = np.arccos(cos_ang)
            
            # Check if angle is within -pi/3 to pi/3
            if -np.pi / 3 < ang < np.pi / 3:
                in_ang_idx[ii] = True
    
    return in_ang_idx
