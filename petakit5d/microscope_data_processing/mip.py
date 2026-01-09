"""
Maximum Intensity Projection (MIP) and pooling utilities.

Ported from MATLAB microscopeDataProcessing/tools/MIP/ directory.
"""

import numpy as np
from typing import Tuple, Union, Literal


def max_pooling_3d(volume: np.ndarray, pool_size: Tuple[int, int, int]) -> np.ndarray:
    """
    Perform 3D max pooling on a volume.
    
    Reduces volume dimensions by taking the maximum value within each pooling block.
    
    Args:
        volume: 3D input array
        pool_size: Pooling size as (y_pool, x_pool, z_pool)
        
    Returns:
        np.ndarray: Downsampled volume with max pooling applied
        
    Examples:
        >>> vol = np.random.rand(100, 100, 50)
        >>> pooled = max_pooling_3d(vol, (2, 2, 2))
        >>> pooled.shape
        (50, 50, 25)
        
    Original MATLAB function: max_pooling_3d.m
    Author: Xiongtao Ruan
    """
    sz = np.array(volume.shape)
    pool_size = np.array(pool_size)
    
    # Pad volume to make it divisible by pool_size
    new_sz = np.ceil(sz / pool_size).astype(int) * pool_size
    pad_size = new_sz - sz
    
    if np.any(pad_size > 0):
        pad_width = [(0, int(p)) for p in pad_size]
        volume = np.pad(volume, pad_width, mode='constant', constant_values=0)
    
    sz = np.array(volume.shape)
    
    # Reshape for pooling
    new_shape = (
        pool_size[0], sz[0] // pool_size[0],
        pool_size[1], sz[1] // pool_size[1],
        pool_size[2], sz[2] // pool_size[2]
    )
    
    reshaped = volume.reshape(new_shape)
    
    # Take max over pooling dimensions (0, 2, 4)
    out = np.max(reshaped, axis=(0, 2, 4))
    
    return out


def min_bbox_3d(volume: np.ndarray, bbox: Union[Tuple, np.ndarray]) -> float:
    """
    Compute minimum value within a 3D bounding box.
    
    Args:
        volume: 3D input array
        bbox: Bounding box as (y_start, x_start, z_start, y_end, x_end, z_end)
              Uses 1-based MATLAB indexing (will be converted internally)
        
    Returns:
        float: Minimum value in the bounding box region
        
    Examples:
        >>> vol = np.random.rand(100, 100, 50)
        >>> bbox = (10, 20, 5, 50, 60, 25)
        >>> min_val = min_bbox_3d(vol, bbox)
        
    Original MATLAB function: min_bbox_3d.m
    Author: Xiongtao Ruan
    """
    # Import crop_3d from crop module
    from .crop import crop_3d
    
    # Crop the region
    cropped = crop_3d(volume, bbox)
    
    # Return minimum
    return np.min(cropped)


def project_3d_to_2d(
    volume: np.ndarray,
    method: Literal[
        'central_xy', 'central_yz', 'central_xz',
        'mip_xy', 'mip_yz', 'mip_xz',
        'mean_xy', 'mean_yz', 'mean_xz'
    ]
) -> np.ndarray:
    """
    Project 3D volume to 2D using various methods.
    
    Supports central slice extraction, maximum intensity projection (MIP),
    and mean projection along different axes.
    
    Args:
        volume: 3D input volume
        method: Projection method. Options:
               - 'central_xy': Central XY slice
               - 'central_yz': Central YZ slice
               - 'central_xz': Central XZ slice
               - 'mip_xy': Maximum intensity projection along Z
               - 'mip_yz': Maximum intensity projection along X
               - 'mip_xz': Maximum intensity projection along Y
               - 'mean_xy': Mean projection along Z
               - 'mean_yz': Mean projection along X
               - 'mean_xz': Mean projection along Y
        
    Returns:
        np.ndarray: 2D projected image
        
    Examples:
        >>> vol = np.random.rand(100, 100, 50)
        >>> mip = project_3d_to_2d(vol, 'mip_xy')
        >>> mip.shape
        (100, 100)
        
    Original MATLAB function: project3DImageto2D.m
    Author: Xiongtao Ruan (January 2020)
    """
    method = method.lower()
    
    if method == 'central_xy':
        z_center = (volume.shape[2] + 1) // 2 - 1  # Convert to 0-based
        frame_out = volume[:, :, z_center]
        
    elif method == 'central_yz':
        x_center = (volume.shape[1] + 1) // 2 - 1
        frame_out = volume[:, x_center, :]
        
    elif method == 'central_xz':
        y_center = (volume.shape[0] + 1) // 2 - 1
        frame_out = volume[y_center, :, :]
        
    elif method == 'mip_xy':
        frame_out = np.max(volume, axis=2)
        
    elif method == 'mip_yz':
        frame_out = np.max(volume, axis=1)
        
    elif method == 'mip_xz':
        frame_out = np.max(volume, axis=0)
        
    elif method == 'mean_xy':
        frame_out = np.nanmean(volume, axis=2)
        
    elif method == 'mean_yz':
        frame_out = np.nanmean(volume, axis=1)
        
    elif method == 'mean_xz':
        frame_out = np.nanmean(volume, axis=0)
        
    else:
        # Unknown method, return original
        frame_out = volume
    
    return frame_out
