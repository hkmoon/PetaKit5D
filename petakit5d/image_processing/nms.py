"""
Non-maximum suppression utilities.

Ported from MATLAB imageProcessing/ directory.
"""

import numpy as np
from scipy import ndimage, interpolate
from typing import Tuple


def non_maximum_suppression(response: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """
    Perform non-maximum suppression along orientation direction.
    
    Suppresses values in the response that are not local maxima along the
    orientation direction. Uses grid conventions of steerableDetector().
    
    Args:
        response: 2D response array
        orientation: 2D orientation array (in radians)
        
    Returns:
        np.ndarray: Response array with non-maxima suppressed to zero
        
    Examples:
        >>> resp = np.random.rand(100, 100)
        >>> orient = np.random.rand(100, 100) * np.pi
        >>> nms_resp = non_maximum_suppression(resp, orient)
        
    Original MATLAB function: nonMaximumSuppression.m
    Author: Francois Aguet
    """
    ny, nx = response.shape
    
    # Pad response with symmetric boundary
    padded = np.pad(response, ((1, 1), (1, 1)), mode='symmetric')
    
    # Create meshgrid for interpolation
    y, x = np.mgrid[0:ny, 0:nx]
    
    # Interpolation coordinates (+1 for padding offset)
    # +1 direction
    x1 = x + 1 + np.cos(orientation)
    y1 = y + 1 + np.sin(orientation)
    
    # -1 direction  
    x2 = x + 1 - np.cos(orientation)
    y2 = y + 1 - np.sin(orientation)
    
    # Interpolate at both directions
    # Use map_coordinates for efficient interpolation
    coords1 = np.array([y1.ravel(), x1.ravel()])
    coords2 = np.array([y2.ravel(), x2.ravel()])
    
    A1 = ndimage.map_coordinates(padded, coords1, order=1, cval=0).reshape(ny, nx)
    A2 = ndimage.map_coordinates(padded, coords2, order=1, cval=0).reshape(ny, nx)
    
    # Extract original response (remove padding)
    result = padded[1:-1, 1:-1].copy()
    
    # Suppress non-maxima
    result[(result < A1) | (result < A2)] = 0
    
    return result
