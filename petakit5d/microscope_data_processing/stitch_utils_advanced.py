"""
Advanced stitching utilities for microscopy image processing.

This module provides functions for tile validation, feather blending, and
cross-correlation based image alignment.

Author: Converted from MATLAB PetaKit5D
Date: 2026
"""

import numpy as np
from typing import Tuple, Optional
from scipy import ndimage


def check_major_tile_valid(fmat: np.ndarray, major_inds: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Check if all major tiles are valid at all xyz voxels.
    
    A tile is considered valid if it contains non-zero values. This function checks
    whether all voxels in the major tiles have at least one non-zero value.
    
    Parameters
    ----------
    fmat : ndarray
        4D array of shape (Y, X, Z, N) where N is the number of tiles
    major_inds : ndarray
        1D array of major tile indices (0-based)
        
    Returns
    -------
    is_valid : bool
        True if all voxels have non-zero values in at least one major tile
    unvalid_bbox : ndarray or None
        If not valid, returns bounding box [y1, x1, z1, y2, x2, z2] (1-based) of
        invalid region. If valid, returns None.
        
    Notes
    -----
    This function checks if all elements are non-zero in at least one of the major
    tiles. If any voxel position has all zeros across major tiles, the tiles are
    considered invalid.
    
    Examples
    --------
    >>> fmat = np.random.rand(10, 10, 5, 3) > 0.5
    >>> major_inds = np.array([0, 1])
    >>> is_valid, bbox = check_major_tile_valid(fmat, major_inds)
    """
    # Select major tiles
    major_tiles = fmat[:, :, :, major_inds]
    
    # Check if any major tile is non-zero at each voxel
    # any along 4th dimension (tiles), then all along spatial dimensions
    has_nonzero = np.any(major_tiles, axis=3)
    is_valid = bool(np.all(has_nonzero))
    
    if not is_valid:
        # Find voxels where all major tiles are zero
        invalid_voxels = ~has_nonzero
        inds = np.where(invalid_voxels)
        
        if len(inds[0]) > 0:
            y, x, z = inds
            # Return 1-based bounding box for MATLAB compatibility
            unvalid_bbox = np.array([
                np.min(y) + 1, np.min(x) + 1, np.min(z) + 1,
                np.max(y) + 1, np.max(x) + 1, np.max(z) + 1
            ])
        else:
            unvalid_bbox = None
    else:
        unvalid_bbox = None
        
    return is_valid, unvalid_bbox


def feather_blending_3d(
    tim_f_block: np.ndarray,
    tim_d_block: np.ndarray,
    tim_block: Optional[np.ndarray] = None,
    bbox: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, bool]:
    """
    3D feather blending of overlapping tiles using distance-weighted blending.
    
    This function blends multiple overlapping tiles using feather blending, where
    each tile is weighted by its distance transform. The blending is computed as
    a weighted average based on distance maps.
    
    Parameters
    ----------
    tim_f_block : ndarray
        4D array of tile intensities, shape (Y, X, Z, N) where N is number of tiles
    tim_d_block : ndarray
        4D array of distance maps for each tile, shape (Y, X, Z, N)
        Each distance map should be 0 outside the tile and >0 inside
    tim_block : ndarray, optional
        Output array to write result into. If provided with bbox, uses in-place
        indexing. Shape should be (Y, X, Z) or larger.
    bbox : ndarray, optional
        Bounding box [y1, x1, z1, y2, x2, z2] (1-based) for placing result in tim_block
        
    Returns
    -------
    tim_block : ndarray
        Blended result, shape (Y, X, Z)
    mex_compute : bool
        Always False (no MEX in Python, kept for API compatibility)
        
    Notes
    -----
    The blending formula is:
        result = sum(intensity * weight) / sum(weight)
    where weight = distance_map * (intensity != 0)
    
    This implements distance-weighted feather blending where tiles with larger
    distance from their edges have higher weights in the blend.
    
    Examples
    --------
    >>> tiles = np.random.rand(10, 10, 5, 2) * 100
    >>> dist_maps = np.ones((10, 10, 5, 2))
    >>> blended, _ = feather_blending_3d(tiles, dist_maps)
    """
    mex_compute = False  # No MEX in Python
    
    # Weight: distance * (intensity != 0)
    tim_w_block = tim_d_block * (tim_f_block != 0)
    
    # Compute weighted average
    numerator = np.sum(tim_f_block.astype(np.float32) * tim_w_block, axis=3)
    denominator = np.sum(tim_w_block, axis=3)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    if tim_block is None or bbox is None:
        # Return full result
        tim_block = result
    else:
        # Place result in specified bounding box (1-based to 0-based conversion)
        y1, x1, z1, y2, x2, z2 = bbox - 1  # Convert to 0-based
        tim_block[y1:y2+1, x1:x2+1, z1:z2+1] = result
    
    return tim_block, mex_compute


def normxcorr2_max_shift(
    T: np.ndarray,
    A: np.ndarray,
    maxShifts: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Compute maximum normalized cross-correlation with constrained shifts.
    
    This function computes 2D normalized cross-correlation between template T
    and image A, constraining the search to a specified shift range. It returns
    the optimal shift and maximum correlation value.
    
    Parameters
    ----------
    T : ndarray
        2D template image
    A : ndarray
        2D search image (should be larger than or equal to T)
    maxShifts : ndarray
        Shift constraints, either:
        - 1D array [max_y, max_x]: symmetric bounds ±[max_y, max_x]
        - 2D array [[min_y, min_x], [max_y, max_x]]: asymmetric bounds
        
    Returns
    -------
    max_off : ndarray
        Optimal shift [dy, dx, 0] where the third element is 0 for 2D compatibility
    max_corr : float
        Maximum correlation coefficient at optimal shift
    C : ndarray
        Cropped correlation map within the specified shift range
        
    Notes
    -----
    Uses skimage's match_template under the hood for normalized cross-correlation.
    The shift constraints help reduce computation and avoid spurious matches.
    
    The correlation coefficient ranges from -1 to 1, where:
    - 1: perfect match
    - 0: no correlation
    - -1: perfect anti-correlation
    
    Examples
    --------
    >>> template = np.random.rand(10, 10)
    >>> image = np.random.rand(50, 50)
    >>> max_shifts = np.array([5, 5])
    >>> offset, corr, C = normxcorr2_max_shift(template, image, max_shifts)
    """
    from skimage.feature import match_template
    
    # Handle shift constraints
    if maxShifts.ndim == 1:
        maxShifts = np.array([[-maxShifts[0], -maxShifts[1]], 
                               [maxShifts[0], maxShifts[1]]])
    
    sz_t = np.array(T.shape)
    
    # Use skimage's match_template which gives normalized correlation
    # pad_input=True makes output size = input size
    C_full = match_template(A, T, pad_input=True)
    
    # match_template output size matches A size
    # To match MATLAB's normxcorr2, we need to adjust the indexing
    # MATLAB's normxcorr2 output has size (size(A) + size(T) - 1)
    # and the peak at position (y,x) means template center is at A[y-sz_t[0]+1, x-sz_t[1]+1]
    
    # Pad C_full to match MATLAB's normxcorr2 output size
    pad_before = sz_t - 1
    pad_after = np.zeros(2, dtype=int)
    C = np.pad(C_full, ((pad_before[0], pad_after[0]), (pad_before[1], pad_after[1])), 
               mode='constant', constant_values=-1)  # Pad with -1 (low correlation)
    
    # Crop C based on maxshift constraints
    # s and t define the valid search region in C
    s = np.maximum(0, np.ceil(maxShifts[0, :] + sz_t).astype(int))
    t = np.minimum(np.floor(maxShifts[1, :] + sz_t).astype(int), C.shape)
    
    C_cropped = C[s[0]:t[0], s[1]:t[1]]
    
    # Find maximum
    max_corr = np.max(C_cropped)
    max_idx = np.argmax(C_cropped)
    y, x = np.unravel_index(max_idx, C_cropped.shape)
    
    # Compute offset (shift from template size to match location)
    max_off = np.array([y + s[0] - sz_t[0], x + s[1] - sz_t[1], 0])
    
    return max_off, float(max_corr), C_cropped
