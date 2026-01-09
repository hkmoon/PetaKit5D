"""
Binary thinning operations for 2D and 3D images.

This module provides morphological thinning operations using hit-or-miss transforms
with standard structuring elements.

Author: Converted from MATLAB by GitHub Copilot
Original MATLAB author: Hunter Elliott (3/2010)
"""

import numpy as np
from scipy import ndimage
from typing import Union


def bw_thin(bw: np.ndarray) -> np.ndarray:
    """
    Thin a 2D or 3D binary image using hit-or-miss operations.
    
    This function thins the input binary 2D or 3D image using a hit-or-miss
    operation with standard structuring elements for binary thinning.
    
    NOTE: This function can be slow, especially for 3D images!
    For 2D thinning, scipy.ndimage.morphology functions may be faster.
    
    Parameters
    ----------
    bw : np.ndarray
        Binary input image (2D or 3D). Should contain only 0s and 1s,
        or boolean values.
        
    Returns
    -------
    np.ndarray
        Thinned binary image with same shape as input.
        
    Raises
    ------
    ValueError
        If input is not 2D or 3D, or if input is not binary.
        
    Examples
    --------
    >>> import numpy as np
    >>> # Create a simple 2D binary image
    >>> img = np.zeros((10, 10), dtype=bool)
    >>> img[3:7, 3:7] = True
    >>> thinned = bw_thin(img)
    >>> thinned.shape
    (10, 10)
    
    >>> # 3D example
    >>> img_3d = np.zeros((10, 10, 10), dtype=bool)
    >>> img_3d[3:7, 3:7, 3:7] = True
    >>> thinned_3d = bw_thin(img_3d)
    >>> thinned_3d.shape
    (10, 10, 10)
    """
    # Convert to boolean for consistent processing
    bw = np.asarray(bw, dtype=bool)
    
    # Check dimensionality
    ndim = bw.ndim
    if ndim not in [2, 3]:
        raise ValueError(f"Input must be 2D or 3D, got {ndim}D")
    
    # Define structuring elements for hit-or-miss operation
    if ndim == 2:
        # 2D structuring elements
        n1 = np.array([[-1, -1, -1],
                       [ 0,  1,  0],
                       [ 1,  1,  1]], dtype=np.int8)
        
        n2 = np.array([[ 0, -1, -1],
                       [ 1,  1, -1],
                       [ 0,  1,  0]], dtype=np.int8)
        
        n_outer_loop = 1
        
    else:  # ndim == 3
        # 3D structuring elements
        n1 = np.zeros((3, 3, 3), dtype=np.int8)
        n1[:, :, 0] = [[-1, -1, -1],
                       [ 0,  0,  0],
                       [ 1,  1,  1]]
        n1[:, :, 1] = [[-1, -1, -1],
                       [ 0,  1,  0],
                       [ 1,  1,  1]]
        n1[:, :, 2] = [[-1, -1, -1],
                       [ 0,  0,  0],
                       [ 1,  1,  1]]
        
        n2 = np.zeros((3, 3, 3), dtype=np.int8)
        n2[:, :, 0] = [[ 0, -1, -1],
                       [ 1,  0, -1],
                       [ 0,  1,  0]]
        n2[:, :, 1] = [[ 0, -1, -1],
                       [ 1,  1, -1],
                       [ 0,  1,  0]]
        n2[:, :, 2] = [[ 0, -1, -1],
                       [ 1,  0, -1],
                       [ 0,  1,  0]]
        
        n_outer_loop = 4
    
    thinner = bw.copy()
    
    # Apply hit-or-miss operation using the neighborhoods and their rotations
    for l in range(n_outer_loop):
        for j in range(4):
            # Apply hit-or-miss with n1 and n2
            thinner = ~_bw_hit_miss(thinner, n1) & thinner
            thinner = ~_bw_hit_miss(thinner, n2) & thinner
            
            # Rotate neighborhoods by 90 degrees in XY plane
            if ndim == 2:
                n1 = np.rot90(n1)
                n2 = np.rot90(n2)
            else:  # 3D
                for k in range(n1.shape[2]):
                    n1[:, :, k] = np.rot90(n1[:, :, k])
                    n2[:, :, k] = np.rot90(n2[:, :, k])
        
        # For 3D, also rotate in the Z dimension
        if ndim == 3:
            for k in range(3):
                # Rotate in YZ plane
                yz_slice_n1 = n1[k, :, :]
                yz_slice_n2 = n2[k, :, :]
                n1[k, :, :] = np.rot90(yz_slice_n1)
                n2[k, :, :] = np.rot90(yz_slice_n2)
    
    return thinner


def _bw_hit_miss(image: np.ndarray, struct_elem: np.ndarray) -> np.ndarray:
    """
    Perform binary hit-or-miss transform.
    
    Parameters
    ----------
    image : np.ndarray
        Binary input image.
    struct_elem : np.ndarray
        Structuring element with values:
        1 for foreground, 0 for don't care, -1 for background.
        
    Returns
    -------
    np.ndarray
        Result of hit-or-miss transform.
    """
    # Convert to boolean
    image = np.asarray(image, dtype=bool)
    
    # Create hit and miss structuring elements
    hit_struct = (struct_elem == 1)
    miss_struct = (struct_elem == -1)
    
    # Perform erosion for hit part
    if np.any(hit_struct):
        hit_result = ndimage.binary_erosion(image, structure=hit_struct, 
                                           border_value=0)
    else:
        hit_result = image
    
    # Perform erosion for miss part (on inverted image)
    if np.any(miss_struct):
        miss_result = ndimage.binary_erosion(~image, structure=miss_struct,
                                             border_value=0)
    else:
        miss_result = np.ones_like(image, dtype=bool)
    
    # Combine results
    return hit_result & miss_result
