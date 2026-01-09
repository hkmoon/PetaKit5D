"""
3D neighborhood connectivity utilities for morphological operations.

This module provides functions for generating 3D neighborhoods with different
connectivity patterns (6, 18, 26) for use in morphological image processing.

Author: Converted from MATLAB (Hunter Elliott, 4/2010)
Date: 2026-01-09
"""

import numpy as np
from typing import Literal


def bwn_hood_3d(conn: Literal[6, 18, 26] = 26) -> np.ndarray:
    """
    Create basic 6, 18, or 26 connectivity 3x3x3 3D neighborhoods for morphological operations.
    
    Creates a 3x3x3 binary mask representing different types of 3D connectivity:
    - 6-connectivity: Face neighbors only (N, S, E, W, U, D)
    - 18-connectivity: Face and edge neighbors
    - 26-connectivity: Face, edge, and corner neighbors (all except center)
    
    Parameters
    ----------
    conn : {6, 18, 26}, optional
        The type of neighborhood connectivity to return. Default is 26.
        
    Returns
    -------
    nhood : np.ndarray
        A 3x3x3 boolean array with the specified connectivity pattern.
        True values indicate neighbor positions.
        
    Raises
    ------
    ValueError
        If conn is not 6, 18, or 26.
        
    Examples
    --------
    >>> # Get 6-connected neighborhood (face neighbors)
    >>> hood6 = bwn_hood_3d(6)
    >>> np.sum(hood6)  # Should be 6
    6
    
    >>> # Get 26-connected neighborhood (all neighbors)
    >>> hood26 = bwn_hood_3d(26)
    >>> np.sum(hood26)  # Should be 26
    26
    
    >>> # The center voxel is always False
    >>> hood = bwn_hood_3d()
    >>> hood[1, 1, 1]
    False
    
    Notes
    -----
    The neighborhoods are returned as 3D boolean arrays where True indicates
    a neighbor position. The center position [1,1,1] is always False since
    a voxel is not its own neighbor.
    
    The indices in MATLAB (1-based) are converted to Python (0-based):
    - MATLAB index 5 -> Python index 4 (center of first layer)
    - MATLAB index 14 -> Python index 13 (center position)
    - MATLAB index 23 -> Python index 22 (center of last layer)
    """
    if conn not in [6, 18, 26]:
        raise ValueError(f"Invalid connectivity: {conn}. Must be 6, 18, or 26")
    
    nhood = np.zeros((3, 3, 3), dtype=bool)
    
    if conn == 6:
        # Face neighbors only (6-connectivity)
        # MATLAB indices: [5 11 13 15 17 23]
        # Python indices: [4 10 12 14 16 22]
        # In 3x3x3 raveled: positions at faces
        indices = [4, 10, 12, 14, 16, 22]
        nhood.flat[indices] = True
        
    elif conn == 18:
        # Face and edge neighbors (18-connectivity)
        # MATLAB indices: [5 10:18 23]
        # Python indices: [4 9:18 22]
        indices = [4] + list(range(9, 18)) + [22]
        nhood.flat[indices] = True
        
    elif conn == 26:
        # All neighbors except center (26-connectivity)
        # MATLAB indices: [1:13 15:end]
        # Python indices: [0:13 14:27]
        indices = list(range(0, 13)) + list(range(14, 27))
        nhood.flat[indices] = True
    
    return nhood
