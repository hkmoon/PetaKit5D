"""
Morphological operations.

Ported from MATLAB imageProcessing/ directory.
"""

import numpy as np
from scipy import ndimage
from typing import Optional


def bw_largest_obj(mask: np.ndarray, connectivity: Optional[int] = None) -> np.ndarray:
    """
    Remove all but the largest object (connected component) in the input binary mask.
    
    Args:
        mask: Input binary matrix
        connectivity: Neighborhood connectivity to use. 
                     For 2D: 1 (4-connectivity) or 2 (8-connectivity)
                     For 3D: 1 (6-connectivity), 2 (18-connectivity), or 3 (26-connectivity)
                     If None, uses default full connectivity
        
    Returns:
        np.ndarray: Binary mask with only the largest connected component
        
    Examples:
        >>> mask = np.array([[1, 0, 0], [1, 0, 1], [0, 0, 1]])
        >>> largest = bw_largest_obj(mask)
        
    Original MATLAB function: bwLargestObj.m
    Author: Hunter Elliott (3/2013)
    """
    # Convert to boolean if not already
    mask_bool = mask.astype(bool)
    
    # Define connectivity structure
    if connectivity is None:
        # Full connectivity (default)
        structure = None
    else:
        # Create structure based on connectivity
        structure = ndimage.generate_binary_structure(mask.ndim, connectivity)
    
    # Label connected components
    labeled, num_features = ndimage.label(mask_bool, structure=structure)
    
    if num_features == 0:
        # No objects found, return empty mask
        return np.zeros_like(mask, dtype=bool)
    
    # Find sizes of each component
    component_sizes = np.bincount(labeled.ravel())
    # Exclude background (label 0)
    component_sizes[0] = 0
    
    # Find largest component
    largest_label = np.argmax(component_sizes)
    
    # Create output mask with only largest component
    output = (labeled == largest_label)
    
    return output
