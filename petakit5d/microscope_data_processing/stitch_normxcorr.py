"""
3D Normalized Cross-Correlation utilities for stitching.

This module provides functions for computing normalized cross-correlation
between 3D image volumes with constrained shift ranges for efficient tile
alignment and registration.

Author: Xiongtao Ruan (MATLAB), Python port
"""

import numpy as np
from scipy.fft import fftn, ifftn
from typing import Tuple, Optional, Union
from .crop import crop_3d


def integral_image_3d_internal(img: np.ndarray, sz_t: Tuple[int, ...]) -> np.ndarray:
    """
    Compute 3D integral image for normalized cross-correlation.
    
    This is an internal helper that may already exist in zarr_utils.
    For now, we'll use a simple implementation.
    
    Parameters
    ----------
    img : np.ndarray
        Input 3D image
    sz_t : tuple
        Template size (3-tuple)
        
    Returns
    -------
    np.ndarray
        Integral image
    """
    from .zarr_utils import integral_image_3d
    return integral_image_3d(img, sz_t)


def normxcorr3_fast(
    T: np.ndarray,
    A: np.ndarray,
    shape: str = 'full'
) -> np.ndarray:
    """
    Fast 3D normalized cross-correlation.
    
    Computes normalized cross-correlation between a 3D template and a 3D image
    using FFT and integral images for efficient computation.
    
    Parameters
    ----------
    T : np.ndarray
        Template array (3D), must be smaller than or equal to image size
    A : np.ndarray
        Image array (3D)
    shape : str, optional
        Output shape: 'full' (default), 'same', or 'valid'
        
    Returns
    -------
    np.ndarray
        Normalized cross-correlation values in range [-1, 1]
        
    Raises
    ------
    ValueError
        If T or A have more than 3 dimensions, or if T is larger than A
        
    Notes
    -----
    Adapted from Daniel Eaton's version with support for 1D and 2D,
    and optimized performance with MEX-based image integral.
    
    References
    ----------
    Based on MATLAB implementation by Xiongtao Ruan (04/27/2024)
    """
    if T.ndim > 3 or A.ndim > 3:
        raise ValueError('A and T must be no more than 3 dimensional matrices')
    
    # Ensure 3D
    while T.ndim < 3:
        T = T[..., np.newaxis]
    while A.ndim < 3:
        A = A[..., np.newaxis]
    
    szT = T.shape[:3]
    szA = A.shape[:3]
    
    if any(st > sa for st, sa in zip(szT, szA)):
        raise ValueError('template must be smaller than image')
    
    pSzT = np.prod(szT)
    szOut = tuple(st + sa - 1 for st, sa in zip(szT, szA))
    
    # Compute the numerator of the NCC
    # Flip template for correlation
    T_flipped = T[::-1, ::-1, ::-1]
    corrTA = np.real(ifftn(fftn(A, s=szOut) * fftn(T_flipped, s=szOut)))
    
    sumT = np.sum(T)
    denomT = np.std(T)
    
    # Make the running-sum/integral-images of A and A^2
    intImgA = integral_image_3d_internal(A, szT)
    
    num = (corrTA - intImgA * sumT / pSzT) / (pSzT - 1)
    
    # Compute the denominator of the NCC
    intImgA2 = integral_image_3d_internal(A * A, szT)
    
    denom = denomT * np.sqrt(np.maximum(intImgA2 - (intImgA**2) / pSzT, 0) / (pSzT - 1))
    
    # Compute the NCC
    C = num / (denom + np.finfo(float).eps) * (denom != 0)
    
    # Handle shape parameter
    shape_lower = shape.lower()
    if shape_lower == 'full':
        pass  # Already in full shape
    elif shape_lower == 'same':
        szTp = tuple((st - 1) // 2 for st in szT)
        C = C[szTp[0]:szTp[0]+szA[0], 
              szTp[1]:szTp[1]+szA[1], 
              szTp[2]:szTp[2]+szA[2]]
    elif shape_lower == 'valid':
        C = C[szT[0]-1:-(szT[0]-1) if szT[0] > 1 else None,
              szT[1]-1:-(szT[1]-1) if szT[1] > 1 else None,
              szT[2]-1:-(szT[2]-1) if szT[2] > 1 else None]
    else:
        raise ValueError(f'unknown SHAPE {shape}, use full, same, or valid')
    
    return C


def normxcorr3_max_shift(
    T: np.ndarray,
    A: np.ndarray,
    maxShifts: Union[np.ndarray, list]
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Compute maximum cross-correlation with constrained shift range.
    
    Computes normalized cross-correlation and finds the maximum correlation
    within a specified shift range constraint.
    
    Parameters
    ----------
    T : np.ndarray
        Template array (3D)
    A : np.ndarray
        Image array (3D)
    maxShifts : array_like
        Maximum allowed shifts. Can be:
        - 1D array of length 3: symmetric bounds [-maxShifts, maxShifts]
        - 2D array of shape (2, 3): [lower_bounds, upper_bounds]
        
    Returns
    -------
    max_off : np.ndarray
        Offset of maximum correlation (y, x, z) in 1-based MATLAB indexing
    max_corr : float
        Maximum correlation value
    C : np.ndarray
        Cropped correlation array
        
    Notes
    -----
    Based on MATLAB implementation by Xiongtao Ruan:
    - (10/15/2020): Initial version
    - (11/02/2020): Add constraints for range of max shifts
    - (12/07/2020): Update constraints as lower/upper bounds
    - (02/04/2022): Crop C based on maxshift constraint first
    
    Examples
    --------
    >>> T = np.random.rand(10, 10, 10)
    >>> A = np.random.rand(20, 20, 20)
    >>> max_off, max_corr, C = normxcorr3_max_shift(T, A, [5, 5, 5])
    """
    maxShifts = np.atleast_2d(maxShifts)
    
    # Convert to 2D array [lower, upper] if needed
    if maxShifts.shape[0] == 1:
        maxShifts = np.vstack([-maxShifts, maxShifts])
    
    # Compute normalized cross-correlation
    C = normxcorr3_fast(T, A)
    
    sz_t = np.array(T.shape[:3])
    
    # Compute valid range based on maxShifts (convert to 1-based for MATLAB compatibility)
    s = np.maximum(1, np.ceil(maxShifts[0, :] + sz_t)).astype(int)
    t = np.minimum(np.floor(maxShifts[1, :] + sz_t), C.shape[:3]).astype(int)
    
    # Crop correlation array to constrained region (convert to 0-based indexing)
    C = crop_3d(C, np.hstack([s, t]))
    
    # Find maximum correlation
    max_corr = np.max(C)
    ind = np.argmax(C)
    max_inds = np.unravel_index(ind, C.shape)
    max_inds = np.array(max_inds)
    
    # Compute offset (1-based MATLAB indexing)
    max_off = max_inds + (s - 1) - sz_t
    
    return max_off, max_corr, C
