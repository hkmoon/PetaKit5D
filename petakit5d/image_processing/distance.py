"""
Distance transform functions for binary images.

This module provides functions for computing various distance transforms
on binary images.
"""

import numpy as np
from typing import Optional


def bw_max_direct_dist(mask: np.ndarray) -> np.ndarray:
    """
    Calculate directional distance in each direction to pixels in the mask.
    
    For each False pixel in the input mask, calculates the distance to a True
    pixel in a particular direction. This is analogous to a distance transform:
    the minimum value over all 4 directions is the same as a city-block distance
    transform, while the maximum gives a sort of maximum-distance transform.
    
    Parameters
    ----------
    mask : np.ndarray
        MxN 2D boolean/binary matrix
    
    Returns
    -------
    dist_mat : np.ndarray
        MxNx4 float32 matrix, where the 3rd dimension corresponds to different
        directions:
        - dist_mat[:,:,0]: distance in direction of increasing row index
        - dist_mat[:,:,1]: distance in direction of decreasing row index
        - dist_mat[:,:,2]: distance in direction of increasing column index
        - dist_mat[:,:,3]: distance in direction of decreasing column index
    
    Examples
    --------
    >>> import numpy as np
    >>> mask = np.array([[0, 0, 1, 0],
    ...                   [0, 0, 0, 0],
    ...                   [1, 0, 0, 1]], dtype=bool)
    >>> dist_mat = bw_max_direct_dist(mask)
    >>> # Minimum over directions gives city-block distance
    >>> cityblock_dist = np.min(dist_mat, axis=2)
    
    Notes
    -----
    - Based on MATLAB bwMaxDirectDist by Hunter Elliott (9/2011)
    - Useful for analyzing directional properties of binary images
    - Can be combined with bwdist for validation
    
    References
    ----------
    Hunter Elliott, 9/2011
    """
    if mask.ndim != 2:
        raise ValueError("Input mask must be 2-dimensional")
    
    mask = np.asarray(mask, dtype=bool)
    M, N = mask.shape
    
    dist_mat = np.zeros((M, N, 4), dtype=np.float32)
    
    # Initialize boundaries
    dist_mat[0, :, 0] = M
    dist_mat[M-1, :, 1] = M
    dist_mat[:, 0, 2] = N
    dist_mat[:, N-1, 3] = N
    
    # Direction 0: increasing row index (downward)
    for m in range(1, M):
        dist_mat[m, ~mask[m, :], 0] = dist_mat[m-1, ~mask[m, :], 0] + 1
    
    # Direction 1: decreasing row index (upward)
    for m in range(M-2, -1, -1):
        dist_mat[m, ~mask[m, :], 1] = dist_mat[m+1, ~mask[m, :], 1] + 1
    
    # Direction 2: increasing column index (rightward)
    for n in range(1, N):
        dist_mat[~mask[:, n], n, 2] = dist_mat[~mask[:, n], n-1, 2] + 1
    
    # Direction 3: decreasing column index (leftward)
    for n in range(N-2, -1, -1):
        dist_mat[~mask[:, n], n, 3] = dist_mat[~mask[:, n], n+1, 3] + 1
    
    return dist_mat


def bw_n_neighbors(
    bw: np.ndarray,
    neighborhood: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Count the number of neighbors of each element in a binary matrix.
    
    Returns a matrix the same size as the input, with each value representing
    the number of non-zero neighbors that element has. This is particularly
    useful for analyzing binary skeletons, where:
    - End-points have exactly 1 neighbor
    - Line points have 2 neighbors
    - Junctions have 3 or more neighbors
    
    Parameters
    ----------
    bw : np.ndarray
        2D or 3D binary matrix
    neighborhood : np.ndarray, optional
        Binary matrix specifying the neighborhood to define neighbors.
        Default is all adjacent pixels (8-connectivity for 2D,
        26-connectivity for 3D)
    
    Returns
    -------
    nn : np.ndarray
        uint8 matrix of the same size as bw, with each element replaced
        by the number of non-zero neighbors
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create a simple skeleton
    >>> skel = np.array([[0, 0, 1, 0, 0],
    ...                   [0, 0, 1, 0, 0],
    ...                   [0, 1, 1, 1, 0],
    ...                   [0, 1, 0, 0, 0],
    ...                   [0, 1, 0, 0, 0]], dtype=bool)
    >>> nn = bw_n_neighbors(skel)
    >>> # Find end-points (exactly 1 neighbor)
    >>> endpoints = (nn == 1) & skel
    >>> # Find junctions (3 or more neighbors)
    >>> junctions = (nn >= 3) & skel
    
    Notes
    -----
    - Based on MATLAB bwNneighbors by Hunter Elliott (3/2010)
    - Supports 2D and 3D binary arrays
    - Uses morphological dilation to count neighbors
    
    References
    ----------
    Hunter Elliott, 3/2010
    """
    from scipy import ndimage
    
    nd = bw.ndim
    
    if nd not in [2, 3]:
        raise ValueError("Input matrix must be 2 or 3 dimensional")
    
    bw = np.asarray(bw, dtype=bool)
    
    # Create default neighborhood if not provided
    if neighborhood is None:
        neighborhood = np.ones([3] * nd, dtype=bool)
        # Exclude center element
        center_idx = tuple([1] * nd)
        neighborhood[center_idx] = False
    
    # Find neighbor positions
    neighbor_indices = np.where(neighborhood)
    n_neighbors = len(neighbor_indices[0])
    
    # Initialize output
    nn = np.zeros(bw.shape, dtype=np.uint8)
    
    # Sum up dilations for each neighbor
    for i in range(n_neighbors):
        # Create a structuring element with just one neighbor
        struct = np.zeros_like(neighborhood)
        idx = tuple(neighbor_indices[j][i] for j in range(nd))
        struct[idx] = True
        
        # Dilate and add to count
        nn += ndimage.binary_dilation(bw, structure=struct).astype(np.uint8)
    
    return nn
