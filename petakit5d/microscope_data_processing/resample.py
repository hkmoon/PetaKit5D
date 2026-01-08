"""
Resampling utilities for microscope data.

Ported from MATLAB microscopeDataProcessing/ directory.
"""

import numpy as np
from scipy import interpolate
from typing import Literal, Union


def resample_stack_3d(
    volume: np.ndarray,
    x_factor: float,
    y_factor: float,
    z_factor: float,
    interp_method: Literal['nearest', 'linear', 'cubic'] = 'nearest',
    save_16bit: bool = False
) -> np.ndarray:
    """
    Resample a 3D volume based on x, y, z voxel scaling factors.
    
    Interpolates the volume to resample based on the x, y, z voxel scaling factor.
    
    Args:
        volume: 3D input volume
        x_factor: X-axis scaling factor (e.g., 0.5 for half resolution)
        y_factor: Y-axis scaling factor
        z_factor: Z-axis scaling factor
        interp_method: Interpolation method ('nearest', 'linear', or 'cubic')
        save_16bit: If True, return as uint16; otherwise return as float32
        
    Returns:
        np.ndarray: Resampled volume
        
    Examples:
        >>> vol = np.random.rand(100, 100, 50)
        >>> # Downsample by factor of 2 in all dimensions
        >>> resampled = resample_stack_3d(vol, 2.0, 2.0, 2.0, 'linear')
        >>> resampled.shape
        (50, 50, 25)
        
    Original MATLAB function: GU_resampleStack3D.m
    Author: Gokul Upadhyayula (Nov 2017)
    """
    ny, nx, nz = volume.shape
    
    # Create original coordinate grids
    y_orig = np.arange(ny)
    x_orig = np.arange(nx)
    z_orig = np.arange(nz)
    
    # Create new coordinate grids based on scaling factors
    y_new = np.arange(0, ny, y_factor)
    x_new = np.arange(0, nx, x_factor)
    z_new = np.arange(0, nz, z_factor)
    
    # Use scipy's RegularGridInterpolator
    interpolator = interpolate.RegularGridInterpolator(
        (y_orig, x_orig, z_orig),
        volume.astype(float),
        method=interp_method,
        bounds_error=False,
        fill_value=0
    )
    
    # Create new coordinate meshgrid
    Y_new, X_new, Z_new = np.meshgrid(y_new, x_new, z_new, indexing='ij')
    
    # Stack coordinates for interpolation
    coords = np.stack([Y_new.ravel(), X_new.ravel(), Z_new.ravel()], axis=1)
    
    # Interpolate
    resampled = interpolator(coords).reshape(Y_new.shape)
    
    # Convert to requested output type
    if save_16bit:
        resampled = np.clip(resampled, 0, 65535).astype(np.uint16)
    else:
        resampled = resampled.astype(np.float32)
    
    return resampled
