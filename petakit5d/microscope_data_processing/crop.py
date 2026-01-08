"""
Cropping utilities for microscope data.

Ported from MATLAB microscopeDataProcessing/crop/ directory.
"""

import numpy as np
from typing import Tuple, Union, Optional


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


def trim_border(
    image: np.ndarray,
    border_size: Union[int, Tuple[int, int, int]],
    method: str = 'both'
) -> np.ndarray:
    """
    Trim the border of a 3D image.
    
    Args:
        image: 3D input array
        border_size: Border size to trim. Can be:
                    - Single int: same size for all dimensions
                    - Tuple of 3 ints: (y_border, x_border, z_border)
        method: Trimming method. Options:
               - 'pre': Trim from start only
               - 'post': Trim from end only  
               - 'both': Trim from both ends (default)
        
    Returns:
        np.ndarray: Trimmed array
        
    Examples:
        >>> data = np.random.rand(100, 100, 50)
        >>> trimmed = trim_border(data, 5, method='both')
        >>> trimmed.shape
        (90, 90, 40)
        
    Original MATLAB function: trimBorder.m
    Author: Xiongtao Ruan
    """
    if isinstance(border_size, int):
        border_size = (border_size, border_size, border_size)
    
    border_size = np.array(border_size)
    
    # If all borders are 0, return as is
    if np.all(border_size == 0):
        return image
    
    sz = np.array(image.shape)
    
    if method == 'pre':
        # Trim from start only
        s = border_size
        t = sz
    elif method == 'post':
        # Trim from end only
        s = np.zeros(3, dtype=int)
        t = sz - border_size
    elif method == 'both':
        # Trim from both ends
        s = border_size
        t = sz - border_size
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'pre', 'post', or 'both'")
    
    # Convert to Python 0-based indexing and slice
    trimmed = image[s[0]:t[0], s[1]:t[1], s[2]:t[2]]
    
    return trimmed


def indexing_4d(
    array: np.ndarray,
    bbox: Union[Tuple, np.ndarray],
    region: np.ndarray,
    region_bbox: Optional[Union[Tuple, np.ndarray]] = None
) -> np.ndarray:
    """
    Assign cropped region to array according to bounding box.
    
    This function assigns region R (optionally cropped with region_bbox) to
    array A at the location specified by bbox.
    
    Args:
        array: 4D target array to modify
        bbox: Bounding box in array as (y_start, x_start, z_start, t_start,
                                        y_end, x_end, z_end, t_end)
              Uses 1-based MATLAB indexing (will be converted internally)
        region: 4D region to insert
        region_bbox: Optional bounding box for cropping region before insertion
        
    Returns:
        np.ndarray: Modified array with region inserted
        
    Examples:
        >>> arr = np.zeros((100, 100, 50, 10))
        >>> region = np.ones((20, 30, 15, 5))
        >>> bbox = (10, 20, 5, 2, 30, 50, 20, 7)  # MATLAB 1-based
        >>> result = indexing_4d(arr, bbox, region)
        
    Original MATLAB function: indexing4d.m  
    Author: Xiongtao Ruan (04/18/2024)
    """
    if isinstance(bbox, tuple):
        bbox = np.array(bbox)
    
    # Convert from MATLAB 1-based to Python 0-based indexing
    y_start, x_start, z_start, t_start, y_end, x_end, z_end, t_end = bbox
    
    if region_bbox is None:
        # Insert entire region
        array[
            y_start - 1:y_end,
            x_start - 1:x_end,
            z_start - 1:z_end,
            t_start - 1:t_end
        ] = region
    else:
        # Crop region first, then insert
        if isinstance(region_bbox, tuple):
            region_bbox = np.array(region_bbox)
        
        ry_start, rx_start, rz_start, rt_start, ry_end, rx_end, rz_end, rt_end = region_bbox
        
        array[
            y_start - 1:y_end,
            x_start - 1:x_end,
            z_start - 1:z_end,
            t_start - 1:t_end
        ] = region[
            ry_start - 1:ry_end,
            rx_start - 1:rx_end,
            rz_start - 1:rz_end,
            rt_start - 1:rt_end
        ]
    
    return array
