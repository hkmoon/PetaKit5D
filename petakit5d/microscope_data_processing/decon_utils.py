"""
Deconvolution utilities for microscopy image processing.

This module provides functions for deconvolution-related operations including
conversion between optical transfer functions (OTF) and point spread functions (PSF).
"""

import numpy as np
from typing import Tuple, Optional


def decon_otf2psf(otf: np.ndarray, out_size: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Convert optical transfer function (OTF) to point-spread function (PSF).
    
    This function performs an inverse FFT on the OTF and properly centers the resulting PSF.
    
    Parameters
    ----------
    otf : np.ndarray
        The optical transfer function as a 3D array. Can be complex.
    out_size : tuple of int, optional
        The desired output size as (nz, ny, nx). If None, uses the size of otf.
        
    Returns
    -------
    psf : np.ndarray
        The point-spread function as a real-valued 3D array with shape out_size.
        
    Notes
    -----
    The PSF is computed by:
    1. Taking the inverse FFT of the OTF
    2. Circularly shifting to center the PSF
    3. Cropping to the requested output size
    
    If the OTF is all zeros, returns a zero array of the requested size.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create a simple OTF
    >>> otf = np.random.randn(64, 64, 64) + 1j * np.random.randn(64, 64, 64)
    >>> psf = decon_otf2psf(otf, out_size=(32, 32, 32))
    >>> psf.shape
    (32, 32, 32)
    """
    # Convert to double precision for accuracy
    otf = np.asarray(otf, dtype=np.complex128)
    otf_size = otf.shape
    
    # Use otf size if output size not specified
    if out_size is None:
        out_size = otf_size
    
    # Check if OTF is all zeros
    if not np.any(otf):
        return np.zeros(out_size, dtype=otf.real.dtype)
    
    # Compute crop size
    crop_size = np.array(otf_size) - np.array(out_size)
    
    # Inverse FFT to get PSF
    psf = np.real(np.fft.ifftn(otf))
    
    # Circularly shift PSF so that center is at (0,0,0)
    # This matches MATLAB's circshift with negative floor(cropSize/2)
    shift = -np.floor(crop_size / 2).astype(int)
    psf = np.roll(psf, shift, axis=(0, 1, 2))
    
    # Crop to output size
    psf = psf[
        :out_size[0],
        :out_size[1],
        :out_size[2]
    ]
    
    return psf
