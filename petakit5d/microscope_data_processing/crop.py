"""
Cropping utilities for microscope data.

Ported from MATLAB microscopeDataProcessing/crop/ directory.
"""

import numpy as np
from typing import Tuple, Union


def crop_3d(
    array: np.ndarray,
    bbox: Union[Tuple[int, int, int, int, int, int], np.ndarray]
) -> np.ndarray:
    """
    Crop a 3D array according to a bounding box.
    
    Args:
        array: 3D input array to crop
        bbox: Bounding box as (y_start, x_start, z_start, y_end, x_end, z_end)
              Uses 1-based indexing to match MATLAB (will be converted internally)
        
    Returns:
        np.ndarray: Cropped 3D array
        
    Examples:
        >>> data = np.random.rand(100, 100, 50)
        >>> bbox = (10, 20, 5, 50, 60, 25)  # MATLAB 1-based indices
        >>> cropped = crop_3d(data, bbox)
        
    Original MATLAB function: crop3d.m
    Author: Xiongtao Ruan (04/18/2024)
    """
    if isinstance(bbox, tuple):
        bbox = np.array(bbox)
    
    # Convert from MATLAB 1-based to Python 0-based indexing
    # MATLAB: bbox(1):bbox(4), bbox(2):bbox(5), bbox(3):bbox(6)
    # Python: bbox[0]-1:bbox[3], bbox[1]-1:bbox[4], bbox[2]-1:bbox[5]
    y_start, x_start, z_start, y_end, x_end, z_end = bbox
    
    # Convert to 0-based indexing
    cropped = array[
        y_start - 1 : y_end,
        x_start - 1 : x_end,
        z_start - 1 : z_end
    ]
    
    return cropped


def crop_4d(
    array: np.ndarray,
    bbox: Union[Tuple[int, int, int, int, int, int, int, int], np.ndarray]
) -> np.ndarray:
    """
    Crop a 4D array according to a bounding box.
    
    Args:
        array: 4D input array to crop
        bbox: Bounding box as (y_start, x_start, z_start, t_start, 
                               y_end, x_end, z_end, t_end)
              Uses 1-based indexing to match MATLAB (will be converted internally)
        
    Returns:
        np.ndarray: Cropped 4D array
        
    Examples:
        >>> data = np.random.rand(100, 100, 50, 10)
        >>> bbox = (10, 20, 5, 1, 50, 60, 25, 5)  # MATLAB 1-based indices
        >>> cropped = crop_4d(data, bbox)
        
    Original MATLAB function: crop4d.m (inferred from crop3d.m pattern)
    Author: Xiongtao Ruan (04/18/2024)
    """
    if isinstance(bbox, tuple):
        bbox = np.array(bbox)
    
    # Convert from MATLAB 1-based to Python 0-based indexing
    y_start, x_start, z_start, t_start, y_end, x_end, z_end, t_end = bbox
    
    # Convert to 0-based indexing
    cropped = array[
        y_start - 1 : y_end,
        x_start - 1 : x_end,
        z_start - 1 : z_end,
        t_start - 1 : t_end
    ]
    
    return cropped
