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


def binary_sphere(radius: float) -> np.ndarray:
    """
    Create a 3D spherical neighborhood/structuring element for morphological operations.
    
    Generates a 3D logical matrix with values inside a sphere of the specified
    radius being True and those outside being False.
    
    Args:
        radius: Positive scalar radius of the sphere to generate.
                Note that for:
                - 0 < radius < 1: single True voxel returned
                - 1 <= radius < sqrt(2): 6-connected neighborhood
                - sqrt(2) <= radius < sqrt(3): 18-connected neighborhood  
                - sqrt(3) <= radius < 2: 26-connected neighborhood
        
    Returns:
        np.ndarray: 3D cubic logical matrix (neighborhood) of size ~2*radius+1
        
    Examples:
        >>> sphere = binary_sphere(3.0)
        >>> sphere.shape
        (7, 7, 7)
        >>> sphere[3, 3, 3]  # Center should be True
        True
        
    Original MATLAB function: binarySphere.m
    Author: Hunter Elliott (2/2010)
    """
    if radius <= 0:
        raise ValueError('You must specify a single positive radius!')
    
    w = int(np.floor(radius))
    
    # Avoid numerical error for special radii like sqrt(2), sqrt(3)
    if round(radius) != radius:
        radius = radius + np.finfo(float).eps * radius
    
    # Get x, y, z coordinate matrices for distance-from-origin calculation
    xx, yy, zz = np.meshgrid(
        np.arange(-w, w + 1),
        np.arange(-w, w + 1),
        np.arange(-w, w + 1),
        indexing='ij'
    )
    
    # Return all points which are less than radius away from origin
    sphere = (xx**2 + yy**2 + zz**2) <= radius**2
    
    return sphere
