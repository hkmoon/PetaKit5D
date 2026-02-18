"""
Resampling utilities for microscope data.

Ported from MATLAB microscopeDataProcessing/ directory.
"""

import numpy as np
from scipy import interpolate
from typing import Literal, Union, Tuple


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


def imresize3_average(
    im: np.ndarray,
    resample_factor: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    """
    Resize a 3D volume by averaging blocks of voxels.
    
    This function downsamples a 3D image by averaging blocks of voxels. If the image
    dimensions are not evenly divisible by the resample factor, the image is padded
    with zeros.
    
    Parameters
    ----------
    im : np.ndarray
        Input 3D volume with shape (nz, ny, nx) or (ny, nx, nz).
    resample_factor : int or tuple of int
        Downsampling factor. If int, same factor used for all dimensions.
        If tuple, specifies (factor_dim0, factor_dim1, factor_dim2).
        
    Returns
    -------
    imout : np.ndarray
        Downsampled volume with shape approximately input_shape / resample_factor.
        Output is float32 dtype.
        
    Notes
    -----
    This function uses block averaging for downsampling, which is equivalent to
    applying a box filter followed by subsampling. When dimensions don't divide
    evenly, padding is added and the average is computed only over valid pixels.
    
    The MATLAB version uses size(im, [1, 2, 3]) which gets dimensions 1, 2, 3.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create a 3D volume
    >>> vol = np.random.rand(100, 100, 50).astype(np.float32)
    >>> # Downsample by factor of 2 in all dimensions
    >>> downsampled = imresize3_average(vol, 2)
    >>> downsampled.shape
    (50, 50, 25)
    >>> # Downsample with different factors per dimension
    >>> downsampled = imresize3_average(vol, (2, 2, 1))
    >>> downsampled.shape
    (50, 50, 50)
    """
    # Convert resample_factor to array
    if isinstance(resample_factor, int):
        rs = np.array([resample_factor, resample_factor, resample_factor])
    else:
        rs = np.array(resample_factor)
    
    # Get original size (first 3 dimensions)
    sz = np.array(im.shape[:3])
    
    # Check if padding is needed
    if np.any(sz % rs != 0):
        # Compute padding needed to make dimensions divisible
        pad_size = np.ceil(sz / rs).astype(int) * rs - sz
        
        # Pad array with zeros at the end
        im = np.pad(im, [(0, pad_size[0]), (0, pad_size[1]), (0, pad_size[2])], 
                    mode='constant', constant_values=0)
        
        # Create mask to track valid pixels
        sz_1 = np.array(im.shape[:3])
        im_c = np.zeros(sz_1, dtype=bool)
        im_c[:sz[0], :sz[1], :sz[2]] = True
        
        # Convert to float for averaging
        im = im.astype(np.float32)
        
        # Reshape for block averaging
        # Original shape: (d0, d1, d2)
        # New shape: (d0//rs[0], rs[0], d1//rs[1], rs[1], d2//rs[2], rs[2])
        # This groups elements that should be averaged together
        new_shape = (
            sz_1[0] // rs[0], rs[0],
            sz_1[1] // rs[1], rs[1],
            sz_1[2] // rs[2], rs[2]
        )
        im = im.reshape(new_shape)
        im_c = im_c.reshape(new_shape)
        
        # Sum over the resample dimensions (axes 1, 3, 5)
        imout = np.sum(im, axis=(1, 3, 5))
        im_c_sum = np.sum(im_c, axis=(1, 3, 5))
        
        # Divide by number of valid pixels to get average
        imout = imout / im_c_sum
    else:
        # No padding needed - simple case
        im = im.astype(np.float32)
        
        # Reshape for block averaging
        new_shape = (
            sz[0] // rs[0], rs[0],
            sz[1] // rs[1], rs[1],
            sz[2] // rs[2], rs[2]
        )
        im = im.reshape(new_shape)
        
        # Take mean over the resample dimensions (axes 1, 3, 5)
        imout = np.mean(im, axis=(1, 3, 5))
    
    return imout
