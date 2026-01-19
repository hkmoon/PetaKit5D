"""
Multiscale filtering implementations for blob and ridge detection.

This module provides multi-scale Laplacian of Gaussian (LoG) and Laplacian of 
Bi-Gaussian (LoBG) filters for detecting blobs and ridges at multiple scales.

Functions:
    filter_multiscale_log_nd: Multi-scale LoG filter for ND images
    filter_multiscale_lobg_nd: Multi-scale LoBG filter for ND images  
    filter_lobg_nd: Single-scale Laplacian of Bi-Gaussian filter

References:
    Xiao, C., M. Staring, et al. (2012). 
    "A multiscale bi-Gaussian filter for adjacent curvilinear structures 
    detection with application to vasculature images." 
    IEEE Transactions on Image Processing, PP(99): 1-1.
"""

import numpy as np
from typing import Tuple, Union, Optional
from .filter_log_nd import filter_log_nd
from .convn_fft import convn_fft


def filter_multiscale_log_nd(
    im_input: np.ndarray,
    sigma_values: np.ndarray,
    spacing: Union[float, np.ndarray] = 1.0,
    border_condition: Union[str, float] = 'symmetric',
    use_normalized_gaussian: bool = True,
    debug_mode: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply multiscale Laplacian of Gaussian (LoG) filter to ND image.
    
    Runs LoG filter across multiple scales and returns the optimal response
    and scale map indicating which scale gave the best response at each pixel.
    
    Parameters
    ----------
    im_input : np.ndarray
        Input ND image
    sigma_values : np.ndarray
        Array of scales (standard deviations) to apply LoG filter at
    spacing : float or np.ndarray, optional
        Pixel spacing. Can be scalar or array matching image dimensions.
        Default: 1.0 (isotropic spacing)
    border_condition : str or float, optional
        Border padding method: 'symmetric', 'replicate', 'wrap', or constant value.
        Default: 'symmetric'
    use_normalized_gaussian : bool, optional
        Whether to normalize the Gaussian kernel. Default: True
    debug_mode : bool, optional
        Print debug information. Default: False
    use_gpu : bool, optional
        Use GPU for convolution if available. Default: False
        
    Returns
    -------
    im_multiscale_log_response : np.ndarray
        Response of the multiscale LoG filter (minimum across scales)
    pixel_scale_map : np.ndarray, optional
        Map indicating optimal scale index for each pixel (1-based indexing)
        
    Examples
    --------
    >>> # Detect blobs at multiple scales
    >>> import numpy as np
    >>> img = np.random.rand(100, 100)
    >>> sigmas = np.array([1.0, 2.0, 4.0])
    >>> response, scale_map = filter_multiscale_log_nd(img, sigmas)
    
    Notes
    -----
    - For blobs of diameter d, optimal sigma ≈ d / (2 * sqrt(ndims))
    - LoG response is negative at blob centers
    - Scale normalization helps compare responses across scales
    """
    # Validate inputs
    if not isinstance(im_input, np.ndarray):
        raise TypeError("im_input must be a numpy array")
    if not isinstance(sigma_values, np.ndarray):
        sigma_values = np.array(sigma_values)
    if sigma_values.ndim != 1:
        raise ValueError("sigma_values must be 1D array")
        
    # Ensure spacing is array
    if np.isscalar(spacing):
        spacing = np.full(im_input.ndim, spacing)
    else:
        spacing = np.asarray(spacing)
        if spacing.size == 1:
            spacing = np.full(im_input.ndim, spacing[0])
            
    if debug_mode:
        print(f"\nRunning LoG filter at multiple scales on image of size {im_input.shape}...")
        
    # Initialize response and scale map
    im_multiscale_log_response = None
    pixel_scale_map = None
    
    # Run LoG filter across scale space
    for i, sigma in enumerate(sigma_values):
        if debug_mode:
            blob_diameter = sigma * 2 * np.sqrt(im_input.ndim)
            print(f"\t{i+1}/{len(sigma_values)}: "
                  f"Trying sigma={sigma:.2f} for blobs of diameter {blob_diameter:.2f}...")
            
        # Compute LoG response at current scale
        im_cur_log_response = filter_log_nd(
            im_input,
            sigma,
            spacing=spacing,
            border_condition=border_condition,
            use_normalized_derivatives=True,
            use_normalized_gaussian=use_normalized_gaussian
        )
        
        if i == 0:
            # Initialize with first scale
            im_multiscale_log_response = im_cur_log_response.copy()
            pixel_scale_map = np.ones(im_input.shape, dtype=np.int32)
        else:
            # Update where current scale gives better (more negative) response
            im_better_mask = im_cur_log_response < im_multiscale_log_response
            im_multiscale_log_response[im_better_mask] = im_cur_log_response[im_better_mask]
            pixel_scale_map[im_better_mask] = i + 1  # 1-based indexing
            
    return im_multiscale_log_response, pixel_scale_map


def filter_lobg_nd(
    im_input: np.ndarray,
    sigma: float,
    rho: float,
    spacing: Union[float, np.ndarray] = 1.0,
    border_condition: Union[str, float] = 'symmetric',
    use_normalized_derivatives: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply Laplacian of Bi-Gaussian (LoBG) filter to ND image.
    
    The LoBG filter is more robust than LoG for detecting blobs under heavy
    overlap. It uses two Gaussians with different scales.
    
    Parameters
    ----------
    im_input : np.ndarray
        Input ND image
    sigma : float
        Standard deviation of the Gaussian
    rho : float
        Ratio of background sigma to foreground sigma (sigmaBackground/sigma)
    spacing : float or np.ndarray, optional
        Pixel spacing. Default: 1.0 (isotropic spacing)
    border_condition : str or float, optional
        Border padding method. Default: 'symmetric'
    use_normalized_derivatives : bool, optional
        Whether to scale-normalize derivatives. Default: False
    use_gpu : bool, optional
        Use GPU for convolution if available. Default: False
        
    Returns
    -------
    im_lobg_response : np.ndarray
        LoBG filtered image
    lobg_kernel : np.ndarray, optional
        The LoBG kernel used (if requested)
        
    Examples
    --------
    >>> # Detect adjacent blobs with rho < 1
    >>> img = np.random.rand(100, 100)
    >>> response, kernel = filter_lobg_nd(img, sigma=2.0, rho=0.2)
    
    Notes
    -----
    - For rho = 1.0, LoBG reduces to standard LoG
    - Smaller rho values (e.g., 0.1-0.3) help separate adjacent structures
    - The filter uses foreground Gaussian for r < sigma and background for r >= sigma
    
    References
    ----------
    Xiao, C., M. Staring, et al. (2012). 
    "A multiscale bi-Gaussian filter for adjacent curvilinear structures 
    detection with application to vasculature images."
    """
    # Validate inputs
    if not isinstance(im_input, np.ndarray):
        raise TypeError("im_input must be a numpy array")
    if not np.isscalar(sigma) or sigma <= 0:
        raise ValueError("sigma must be a positive scalar")
    if not np.isscalar(rho) or rho <= 0:
        raise ValueError("rho must be a positive scalar")
        
    # Ensure spacing is array
    if np.isscalar(spacing):
        spacing = np.full(im_input.ndim, spacing)
    else:
        spacing = np.asarray(spacing)
        if spacing.size == 1:
            spacing = np.full(im_input.ndim, spacing[0])
            
    dims = im_input.ndim
    
    # Adjust sigma according to pixel spacing
    sigma_imsp = sigma / spacing
    
    # Compute the bi-gaussian kernel
    w = np.ceil(4 * sigma_imsp).astype(int)
    
    # Create coordinate grids
    xrange = [np.arange(-w[i], w[i] + 1) for i in range(dims)]
    
    if dims == 1:
        x = [xrange[0]]
    else:
        x = np.meshgrid(*xrange, indexing='ij')
        
    # Compute radial distance
    rad = np.zeros(x[0].shape if dims > 1 else len(x[0]))
    for i in range(dims):
        xi = x[i] if dims > 1 else x[0]
        rad = rad + (xi * spacing[i])**2
    rad = np.sqrt(rad)
    
    # Helper functions
    def normalize_kernel(k):
        return k / np.sum(k)
    
    def gauss_kernel(r, s):
        return normalize_kernel(np.exp(-r**2 / (2 * s**2)))
    
    def log_kernel(r, s):
        return ((r**2 - s**2) / s**4) * gauss_kernel(r, s)
    
    # Compute foreground and background kernels
    fg_kernel = log_kernel(rad, sigma)
    bg_kernel = log_kernel(rad + rho*sigma - sigma, rho*sigma)
    
    # Combine into LoBG kernel
    lobg_kernel = fg_kernel.copy()
    lobg_kernel[rad >= sigma] = rho**2 * bg_kernel[rad >= sigma]
    
    # Apply scale normalization if requested
    if use_normalized_derivatives:
        lobg_kernel = sigma**2 * lobg_kernel
        
    # Remove DC component
    lobg_kernel = lobg_kernel - np.mean(lobg_kernel)
    
    # Pad image and apply convolution in Fourier domain
    if dims > 1:
        pad_size = w
    else:
        pad_size = w
        
    # Pad the image
    if isinstance(border_condition, str):
        if border_condition == 'symmetric':
            mode = 'reflect'
        elif border_condition == 'replicate':
            mode = 'edge'
        elif border_condition == 'antisymmetric':
            mode = 'reflect'
        else:
            mode = border_condition
        pad_width = [(int(p), int(p)) for p in pad_size]
        im_padded = np.pad(im_input, pad_width, mode=mode)
    else:
        # Constant padding
        pad_width = [(int(p), int(p)) for p in pad_size]
        im_padded = np.pad(im_input, pad_width, constant_values=border_condition)
        
    # Apply convolution using FFT
    im_lobg_response = convn_fft(im_padded, lobg_kernel, shape='valid')
    
    return im_lobg_response, lobg_kernel


def filter_multiscale_lobg_nd(
    im_input: np.ndarray,
    sigma_values: np.ndarray,
    rho: float,
    spacing: Union[float, np.ndarray] = 1.0,
    border_condition: Union[str, float] = 'symmetric',
    debug_mode: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply multiscale Laplacian of Bi-Gaussian (LoBG) filter to ND image.
    
    Runs LoBG filter across multiple scales and returns the optimal response
    and scale map.
    
    Parameters
    ----------
    im_input : np.ndarray
        Input ND image
    sigma_values : np.ndarray
        Array of scales (standard deviations) to apply LoBG filter at
    rho : float
        Ratio of background sigma to foreground sigma (typically 0.1-0.3)
    spacing : float or np.ndarray, optional
        Pixel spacing. Default: 1.0 (isotropic spacing)
    border_condition : str or float, optional
        Border padding method. Default: 'symmetric'
    debug_mode : bool, optional
        Print debug information. Default: False
    use_gpu : bool, optional
        Use GPU for convolution if available. Default: False
        
    Returns
    -------
    im_multiscale_lobg_response : np.ndarray
        Response of the multiscale LoBG filter
    pixel_scale_map : np.ndarray, optional
        Map indicating optimal scale index for each pixel (1-based indexing)
        
    Examples
    --------
    >>> # Detect adjacent ridges at multiple scales
    >>> img = np.random.rand(100, 100)
    >>> sigmas = np.array([1.0, 2.0, 4.0])
    >>> response, scale_map = filter_multiscale_lobg_nd(img, sigmas, rho=0.2)
    
    Notes
    -----
    - LoBG is more robust than LoG for overlapping structures
    - Typical rho values: 0.1-0.3 for adjacent structures
    - For rho = 1.0, equivalent to multiscale LoG
    """
    # Validate inputs
    if not isinstance(im_input, np.ndarray):
        raise TypeError("im_input must be a numpy array")
    if not isinstance(sigma_values, np.ndarray):
        sigma_values = np.array(sigma_values)
    if sigma_values.ndim != 1:
        raise ValueError("sigma_values must be 1D array")
    if not np.isscalar(rho) or rho <= 0:
        raise ValueError("rho must be a positive scalar")
        
    # Ensure spacing is array
    if np.isscalar(spacing):
        spacing = np.full(im_input.ndim, spacing)
    else:
        spacing = np.asarray(spacing)
        if spacing.size == 1:
            spacing = np.full(im_input.ndim, spacing[0])
            
    if debug_mode:
        print(f"\nRunning LoBG filter at multiple scales on image of size {im_input.shape}...")
        
    # Initialize response and scale map
    im_multiscale_lobg_response = None
    pixel_scale_map = None
    
    # Run LoBG filter across scale space
    for i, sigma in enumerate(sigma_values):
        if debug_mode:
            print(f"\t{i+1}/{len(sigma_values)}: Trying sigma={sigma:.2f}...")
            
        # Compute LoBG response at current scale
        im_cur_lobg_response, _ = filter_lobg_nd(
            im_input,
            sigma,
            rho,
            spacing=spacing,
            border_condition=border_condition,
            use_normalized_derivatives=True
        )
        
        if i == 0:
            # Initialize with first scale
            im_multiscale_lobg_response = im_cur_lobg_response.copy()
            pixel_scale_map = np.ones(im_input.shape, dtype=np.int32)
        else:
            # Update where current scale gives better (more negative) response
            im_better_mask = im_cur_lobg_response < im_multiscale_lobg_response
            im_multiscale_lobg_response[im_better_mask] = im_cur_lobg_response[im_better_mask]
            pixel_scale_map[im_better_mask] = i + 1  # 1-based indexing
            
    return im_multiscale_lobg_response, pixel_scale_map
