"""
Optimized 3D Gaussian filtering with border correction.

This module provides fast Gaussian filtering for 2D/3D images with options for
border correction and NaN handling.

Author: Converted from MATLAB by GitHub Copilot
Original MATLAB authors: dT (13/03/01), revamped by jonas
"""

import numpy as np
from scipy import ndimage
from typing import Union, Tuple, Optional, List
import warnings


def fast_gauss_3d(
    img: np.ndarray,
    sigma: Union[float, List[float], Tuple[float, ...]],
    f_sze: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
    correct_border: int = 1,
    filter_mask: Optional[np.ndarray] = None,
    reduce_nan_image: bool = False
) -> np.ndarray:
    """
    Apply a 2D or 3D Gaussian filter with border correction.
    
    This function provides optimized Gaussian filtering with options to correct
    for border effects and handle NaN values.
    
    Parameters
    ----------
    img : np.ndarray
        2D or 3D input image.
    sigma : float or sequence of floats
        Standard deviation of Gaussian filter. Can be single value or one per dimension.
    f_sze : int or sequence of ints, optional
        Size of the Gaussian mask (odd size required for symmetric mask).
        If None, uses +/- 4*sigma.
    correct_border : int, default=1
        Border correction mode:
        - 0: No correction
        - 1: Lessened border effects (default)
        - 2: Old version of correction
    filter_mask : np.ndarray, optional
        Custom filter mask. If provided, sigma and f_sze are ignored.
    reduce_nan_image : bool, default=False
        If True, clip NaN-containing areas before filtering to increase speed.
        
    Returns
    -------
    np.ndarray
        Filtered image with same shape as input.
        
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(50, 50, 30)
    >>> filtered = fast_gauss_3d(img, sigma=2.0)
    >>> filtered.shape
    (50, 50, 30)
    
    >>> # Different sigma per dimension
    >>> filtered = fast_gauss_3d(img, sigma=[1.0, 1.0, 2.0])
    """
    # Input validation
    if img.size == 0:
        raise ValueError("Please pass nonempty image to fast_gauss_3d")
    
    # Get dimensionality
    dims = img.ndim
    if dims not in [2, 3]:
        raise ValueError(f"Image must be 2D or 3D, got {dims}D")
    
    # Convert sigma to list
    if np.isscalar(sigma):
        sigma = [sigma] * dims
    else:
        sigma = list(sigma)
        if len(sigma) != dims:
            raise ValueError(f"Sigma must have length {dims}, got {len(sigma)}")
    
    # Use custom filter mask if provided
    if filter_mask is not None:
        # Apply custom filter
        return ndimage.convolve(img, filter_mask, mode='constant', cval=0.0)
    
    # Determine filter size
    if f_sze is None:
        # Use +/- 4*sigma
        f_sze = [int(np.ceil(4 * s)) * 2 + 1 for s in sigma]
    elif np.isscalar(f_sze):
        f_sze = [f_sze] * dims
    else:
        f_sze = list(f_sze)
        if len(f_sze) != dims:
            raise ValueError(f"f_sze must have length {dims}, got {len(f_sze)}")
    
    # Ensure odd filter sizes
    f_sze = [int(fs) if fs % 2 == 1 else int(fs) + 1 for fs in f_sze]
    
    # Handle NaN values
    has_nan = np.any(np.isnan(img))
    if has_nan:
        nan_mask = np.isnan(img)
        img_work = img.copy()
        img_work[nan_mask] = 0.0
    else:
        img_work = img
        nan_mask = None
    
    # Apply Gaussian filter using scipy
    # scipy.ndimage.gaussian_filter handles multi-dimensional sigma
    if correct_border == 0:
        # No border correction
        out = ndimage.gaussian_filter(img_work, sigma, mode='constant', cval=0.0)
    elif correct_border == 1:
        # Modern border correction
        out = ndimage.gaussian_filter(img_work, sigma, mode='reflect')
        
        # If there are NaNs, correct for them
        if has_nan:
            # Create weight image
            weight = np.ones_like(img)
            weight[nan_mask] = 0.0
            weight_filtered = ndimage.gaussian_filter(weight, sigma, mode='reflect')
            
            # Avoid division by zero
            weight_filtered[weight_filtered < 1e-10] = 1.0
            out = out / weight_filtered
            
            # Restore NaNs where original image had NaNs everywhere in neighborhood
            out[weight_filtered < 0.01] = np.nan
    else:  # correct_border == 2
        # Old version of correction
        out = ndimage.gaussian_filter(img_work, sigma, mode='nearest')
        
        if has_nan:
            weight = np.ones_like(img)
            weight[nan_mask] = 0.0
            weight_filtered = ndimage.gaussian_filter(weight, sigma, mode='nearest')
            weight_filtered[weight_filtered < 1e-10] = 1.0
            out = out / weight_filtered
            out[weight_filtered < 0.01] = np.nan
    
    return out
