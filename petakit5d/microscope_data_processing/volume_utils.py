"""
Volume processing utilities for microscopy data.

This module provides utilities for 3D volume processing including erosion
and flat field correction.

Author: Xiongtao Ruan (MATLAB), Python port
"""

import numpy as np
from scipy.ndimage import binary_erosion
from typing import Optional
import warnings


def erode_volume_by_2d_projection(
    vol: np.ndarray,
    esize: int
) -> np.ndarray:
    """
    Edge erosion by erosion of the max projection in XZ plane.
    
    Instead of eroding the whole 3D volume, this function erodes the maximum
    projection in the XZ plane. This is more efficient for deskew and
    deskew-rotate-like volumes.
    
    Parameters
    ----------
    vol : np.ndarray
        Input 3D volume (or 2D image)
    esize : int
        Erosion size (scalar). If 0, returns volume unchanged.
        
    Returns
    -------
    np.ndarray
        Eroded volume (modified in-place and returned)
        
    Notes
    -----
    Based on MATLAB implementation by Xiongtao Ruan (10/11/2020)
    
    The function:
    1. Computes XZ projection (max along Y axis)
    2. Erodes the projection
    3. Applies eroded mask back to 3D volume
    4. Zeros out Y edges
    
    Examples
    --------
    >>> vol = np.random.rand(100, 200, 150)
    >>> vol_eroded = erode_volume_by_2d_projection(vol, esize=5)
    """
    if esize == 0:
        return vol
    
    sz = vol.shape
    
    # Handle 2D case
    if vol.ndim == 2 or (vol.ndim == 3 and sz[0] == 1):
        mask = vol != 0
        vol = vol * mask.astype(vol.dtype)
        return vol
    
    # Calculate XZ projection (max along axis 1, which is Y in Python 0-indexed)
    # In MATLAB: squeeze(max(vol, [], 1)) means max along dim 1 (Y in MATLAB 1-indexed)
    # In Python: max along axis 0 (Y in Python 0-indexed, since shape is (Y, X, Z))
    MIP = np.squeeze(np.max(vol, axis=0) > 0)
    
    # Erode the projection
    if np.all(MIP > 0):
        # If all pixels are positive, just set borders to False
        MIP_erode = np.zeros_like(MIP, dtype=bool)
        MIP_erode[esize:-esize if esize > 0 else None, 
                  esize:-esize if esize > 0 else None] = \
            MIP[esize:-esize if esize > 0 else None, 
                esize:-esize if esize > 0 else None]
    else:
        # Pad and erode
        MIP_pad = np.zeros((sz[1] + 2, sz[2] + 2), dtype=bool)
        MIP_pad[1:-1, 1:-1] = MIP
        
        # Create square structuring element
        from scipy.ndimage import generate_binary_structure
        selem = np.ones((esize * 2 + 1, esize * 2 + 1), dtype=bool)
        MIP_pad = binary_erosion(MIP_pad, structure=selem)
        
        MIP_erode = MIP_pad[1:-1, 1:-1]
    
    # Create 3D mask and apply - broadcast 2D mask to 3D
    # MATLAB: permute(MIP_erode, [3, 1, 2]) creates (1, X, Z) from (X, Z)
    # Python: need to add axis 0 for Y dimension
    vol = vol * MIP_erode[np.newaxis, :, :].astype(vol.dtype)
    
    # Zero out Y edges
    vol[:esize, :, :] = 0
    vol[-esize:, :, :] = 0
    
    return vol


def process_flatfield_correction_frame(
    frame: np.ndarray,
    ls_image: np.ndarray,
    background_image: np.ndarray,
    const_offset: Optional[float] = None,
    lower_limit: float = 0.4,
    remove_ff_im_background: bool = True,
    ls_rescale: bool = True,
    cast_data_type: bool = True
) -> np.ndarray:
    """
    Apply flat field correction to a microscopy frame.
    
    Parameters
    ----------
    frame : np.ndarray
        Input microscopy frame (2D or 3D)
    ls_image : np.ndarray
        Light sheet illumination pattern image (2D or 3D)
    background_image : np.ndarray
        Background image (2D or 3D)
    const_offset : float, optional
        Constant offset to add after correction. If None, uses background.
    lower_limit : float, optional
        Minimum value for flat field mask (default: 0.4)
    remove_ff_im_background : bool, optional
        Whether to subtract background from LS image (default: True)
    ls_rescale : bool, optional
        Whether to rescale LS by maximum (default: True)
    cast_data_type : bool, optional
        Whether to cast output to input dtype (default: True)
        
    Returns
    -------
    np.ndarray
        Flat field corrected frame
        
    Notes
    -----
    Based on MATLAB implementation:
    - Gokul Upadhyayula (2016)
    - Xiongtao Ruan updates (2020-2023)
    
    The correction formula is:
    corrected = (raw - background) / ls_mask
    
    where ls_mask is the processed light sheet pattern.
    
    Examples
    --------
    >>> frame = np.random.rand(512, 512, 100) * 1000
    >>> ls_im = np.random.rand(512, 512) * 1.5 + 0.5
    >>> bg_im = np.random.rand(512, 512) * 100
    >>> corrected = process_flatfield_correction_frame(frame, ls_im, bg_im)
    """
    input_dtype = frame.dtype
    
    # Average Z planes of LS image if 3D
    if ls_image.ndim == 3:
        ls_image = np.mean(ls_image, axis=2)
    
    # Get image size from first plane
    if frame.ndim == 3:
        im_size = frame.shape[:2]
    else:
        im_size = frame.shape
    
    # Crop LS data if necessary
    map_size = ls_image.shape
    D = np.ceil((np.array(map_size) - np.array(im_size)) / 2).astype(int)
    if np.any(D > 0):
        ls_image = ls_image[D[0]:D[0]+im_size[0], D[1]:D[1]+im_size[1]]
    
    # Crop background data if necessary
    map_size = background_image.shape
    D = np.ceil((np.array(map_size) - np.array(im_size)) / 2).astype(int)
    if np.any(D > 0):
        background_image = background_image[D[0]:D[0]+im_size[0], D[1]:D[1]+im_size[1]]
    
    # Prepare LS flat-field correction mask
    ls_image = ls_image.astype(np.float32)
    background_image = background_image.astype(np.float32)
    
    if remove_ff_im_background:
        ls_image = ls_image - background_image
    
    if ls_rescale:
        ls_image = ls_image / np.max(ls_image)
    
    # Apply lower limit
    ls_image = np.maximum(ls_image, lower_limit)
    
    # Apply flat field correction
    frame = frame.astype(np.float32)
    
    # Expand dimensions for broadcasting if needed
    if frame.ndim == 3 and ls_image.ndim == 2:
        ls_image = ls_image[:, :, np.newaxis]
        background_image = background_image[:, :, np.newaxis]
    
    # Correction formula: (raw - background) / ls_mask
    if const_offset is not None:
        frame = (frame - background_image) / ls_image + const_offset
    else:
        frame = (frame - background_image) / ls_image + background_image
    
    # Cast back to original dtype if requested
    if cast_data_type:
        # Clip to valid range for integer types
        if np.issubdtype(input_dtype, np.integer):
            info = np.iinfo(input_dtype)
            frame = np.clip(frame, info.min, info.max)
        frame = frame.astype(input_dtype)
    
    return frame
