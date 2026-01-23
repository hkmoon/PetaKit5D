"""
Deconvolution utilities for microscopy image processing.

This module provides functions for deconvolution-related operations including
conversion between optical transfer functions (OTF) and point spread functions (PSF),
and mask edge erosion for deconvolution.
"""

import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure
from typing import Tuple, Optional, Union


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


def decon_psf2otf(psf: np.ndarray, out_size: Optional[Union[Tuple[int, ...], np.ndarray]] = None) -> np.ndarray:
    """
    Convert point-spread function (PSF) to optical transfer function (OTF).
    
    This function:
    1. Pads PSF to output size
    2. Circularly shifts so PSF center is at origin
    3. Computes FFT to get OTF
    
    Parameters
    ----------
    psf : np.ndarray
        The point-spread function (2D or 3D array)
    out_size : tuple or array of int, optional
        Desired output size. If None, uses PSF size.
        Must be >= PSF size in all dimensions.
        
    Returns
    -------
    otf : np.ndarray
        The optical transfer function (complex-valued)
        
    Raises
    ------
    ValueError
        If out_size is smaller than PSF size in any dimension
        
    Notes
    -----
    Based on MATLAB's psf2otf function. The PSF is assumed to be centered
    at floor(psfSize/2). After shifting, the "center" is at element (0,0,0).
    
    The function automatically pads size arrays to 3D for internal processing.
    
    Examples
    --------
    >>> import numpy as np
    >>> psf = np.random.rand(11, 11, 11)
    >>> otf = decon_psf2otf(psf, out_size=(64, 64, 64))
    >>> otf.shape
    (64, 64, 64)
    """
    psf_size = np.array(psf.shape)
    
    if out_size is None:
        out_size = psf_size.copy()
    else:
        out_size = np.array(out_size)
    
    # Empty PSF handling
    if psf.size == 0:
        # For empty PSF, just return zeros with proper size
        return np.zeros(tuple(out_size), dtype=np.complex128)
    
    # Pad sizes to 3D if needed for internal calculations
    psf_size_padded, out_size_padded = _decon_padlength(psf_size, out_size)
    
    # Check that output size is not smaller than PSF size
    # Compare based on original dimensions (not padded)
    if np.any(out_size[:len(psf_size)] < psf_size):
        raise ValueError(f"Output size {tuple(out_size)} cannot be smaller than PSF size {tuple(psf_size)}")
    
    # Handle zero PSF
    if not np.any(psf):
        return np.zeros(tuple(out_size), dtype=psf.dtype)
    
    # Pad the PSF to out_size_padded (3D)
    # First pad to match PSF dimensions with out_size (original dimensionality)
    if len(out_size) > len(psf_size):
        # Need to add dimensions to PSF
        while psf.ndim < len(out_size):
            psf = psf[..., np.newaxis]
        psf_size = np.array(psf.shape)
    
    # Now pad to out_size (not out_size_padded which is 3D)
    pad_size = out_size - psf_size[:len(out_size)]
    if np.any(pad_size > 0):
        pad_width = [(0, int(p)) for p in pad_size]
        # Pad remaining dimensions if psf has more dimensions than out_size
        if psf.ndim > len(pad_width):
            pad_width.extend([(0, 0)] * (psf.ndim - len(pad_width)))
        psf = np.pad(psf, pad_width, mode='constant', constant_values=0)
    
    # Circularly shift PSF so that the "center" is at (0,0,0)
    # Center is at floor(psfSize/2) for each dimension
    shift = tuple(-int(np.floor(s / 2)) for s in psf_size[:len(out_size)])
    psf = np.roll(psf, shift, axis=tuple(range(len(shift))))
    
    # Compute the OTF
    otf = np.fft.fftn(psf)
    
    return otf


def _decon_padlength(sz_1: np.ndarray, sz_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad input size vectors with ones to give them equal 3D lengths.
    
    Helper function for decon_psf2otf.
    
    Parameters
    ----------
    sz_1 : np.ndarray
        First size vector
    sz_2 : np.ndarray
        Second size vector
        
    Returns
    -------
    psz_1 : np.ndarray
        Padded first size vector (3D)
    psz_2 : np.ndarray
        Padded second size vector (3D)
        
    Examples
    --------
    >>> sz1 = np.array([64, 64])
    >>> sz2 = np.array([128, 128, 64])
    >>> psz1, psz2 = _decon_padlength(sz1, sz2)
    >>> psz1
    array([64, 64, 1])
    >>> psz2
    array([128, 128, 64])
    """
    num_dims = 3
    
    psz_1 = np.concatenate([sz_1, np.ones(max(0, num_dims - len(sz_1)), dtype=int)])
    psz_2 = np.concatenate([sz_2, np.ones(max(0, num_dims - len(sz_2)), dtype=int)])
    
    return psz_1, psz_2


def decon_mask_edge_erosion(mask: np.ndarray, edge_erosion: int) -> np.ndarray:
    """
    Erode the edges of a binary mask for deconvolution.
    
    For full rectangular masks, directly sets edge pixels to False.
    For irregular shapes, uses morphological erosion with disk/line structuring elements.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask (2D or 3D)
    edge_erosion : int
        Number of pixels to erode from edges
        
    Returns
    -------
    mask_eroded : np.ndarray
        Eroded binary mask (boolean array)
        
    Notes
    -----
    For regular rectangular full images, edges are directly set to False for efficiency.
    For irregular shapes, binary_erosion with appropriate structuring elements is used.
    
    In 3D, erosion is applied:
    - In XY planes using a disk-like structure
    - Along Z using a line structure
    
    Examples
    --------
    >>> import numpy as np
    >>> mask = np.ones((100, 100), dtype=bool)
    >>> eroded = decon_mask_edge_erosion(mask, edge_erosion=5)
    >>> # Creates a mask with 5-pixel border set to False
    >>> np.sum(eroded)
    8100
    """
    mask = mask > 0
    
    # Handle edge_erosion = 0 (no erosion)
    if edge_erosion == 0:
        return mask
    
    # For regular rectangular full image, directly set edges as False
    if np.sum(mask) == mask.size:
        # Full mask case - directly set borders to False
        if mask.ndim >= 1:
            mask[:edge_erosion] = False
            mask[-edge_erosion:] = False
        
        if mask.ndim >= 2:
            mask[:, :edge_erosion] = False
            mask[:, -edge_erosion:] = False
        
        if mask.ndim == 3:
            mask[:, :, :edge_erosion] = False
            mask[:, :, -edge_erosion:] = False
        
        return mask
    
    # For irregular shape image, use morphological erosion
    # Set outer boundary to False first
    if mask.ndim >= 1:
        mask[0] = False
        mask[-1] = False
    
    if mask.ndim >= 2:
        mask[:, 0] = False
        mask[:, -1] = False
    
    if mask.ndim == 2:
        # 2D erosion with disk
        struct = generate_binary_structure(2, 1)
        # Iterate erosion to approximate disk of radius edge_erosion
        for _ in range(edge_erosion - 1):
            mask = binary_erosion(mask, structure=struct)
            
    elif mask.ndim == 3:
        mask[:, :, 0] = False
        mask[:, :, -1] = False
        
        # Erode in XY plane with disk-like structure
        struct_xy = generate_binary_structure(2, 1)
        for z in range(mask.shape[2]):
            slice_2d = mask[:, :, z].copy()
            for _ in range(edge_erosion - 1):
                slice_2d = binary_erosion(slice_2d, structure=struct_xy)
            mask[:, :, z] = slice_2d
        
        # Erode in Z direction with line
        # Transpose to (Z, Y, X) for Z erosion
        mask = np.transpose(mask, (2, 0, 1))
        struct_z = np.ones((edge_erosion * 2 - 1, 1, 1), dtype=bool)
        mask = binary_erosion(mask, structure=struct_z)
        # Transpose back to (Y, X, Z)
        mask = np.transpose(mask, (1, 2, 0))
    
    return mask
