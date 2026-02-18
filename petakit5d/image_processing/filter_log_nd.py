"""
N-dimensional Laplacian of Gaussian (LoG) filter implementation.

Ported from MATLAB PetaKit5D imageProcessing/filterLoGND.m
"""

import numpy as np
from typing import Union, Tuple, Optional, Literal
from .convn_fft import convn_fft


def filter_log_nd(
    image: np.ndarray,
    sigma: float,
    spacing: Union[float, Tuple[float, ...]] = 1.0,
    border_condition: Union[str, float] = 'symmetric',
    use_normalized_derivatives: bool = False,
    use_normalized_gaussian: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply N-dimensional Laplacian of Gaussian (LoG) filter to an image.
    
    The LoG filter combines Gaussian smoothing with the Laplacian operator,
    which is useful for blob detection and edge detection across multiple scales.
    
    Parameters
    ----------
    image : np.ndarray
        Input N-D image to filter
    sigma : float
        Standard deviation of the Gaussian kernel in physical units.
        If spacing is not specified, sigma is in pixels.
    spacing : float or tuple of float, optional
        Pixel spacing of the input image. Can be a scalar (isotropic) or
        tuple with length equal to image dimensions (anisotropic).
        Default: 1.0 (unit isotropic spacing)
    border_condition : str or float, optional
        Method for handling borders. Options:
        - 'symmetric': mirror reflection at borders (default)
        - 'replicate': replicate edge values
        - 'constant': use constant value (pad with zeros)
        - 'wrap': circular wrap-around
        - float: pad with specified constant value
        Default: 'symmetric'
    use_normalized_derivatives : bool, optional
        If True, use scale-normalized derivatives (multiply by sigma^2).
        This compensates for amplitude decrease at larger scales.
        Default: False
    use_normalized_gaussian : bool, optional
        If True, normalize the Gaussian kernel to sum to 1.
        Default: True
        
    Returns
    -------
    log_response : np.ndarray
        LoG filtered image of same shape as input
    log_kernel : np.ndarray, optional
        The LoG kernel used (returned if function called with 2 return values)
        
    Examples
    --------
    >>> import numpy as np
    >>> # 2D example with blob
    >>> img = np.zeros((50, 50))
    >>> img[20:30, 20:30] = 1.0
    >>> log_response = filter_log_nd(img, sigma=2.0)
    >>> 
    >>> # 3D example with anisotropic spacing
    >>> vol = np.random.rand(40, 40, 20)
    >>> log_response = filter_log_nd(vol, sigma=2.0, spacing=(1.0, 1.0, 2.0))
    >>> 
    >>> # Scale-normalized for multi-scale blob detection
    >>> log_response = filter_log_nd(img, sigma=3.0, use_normalized_derivatives=True)
    
    Notes
    -----
    - The LoG operator is the Laplacian of a Gaussian: ∇²G(x, σ)
    - For scale-normalized derivatives, multiply by σ² to make response
      independent of scale
    - Useful for blob detection at scale σ/√n_dims
    - The kernel is zero-meaned to avoid DC component
    - Uses FFT-based convolution for efficiency
    
    References
    ----------
    Lindeberg, T. (1998). "Feature Detection with Automatic Scale Selection."
    International Journal of Computer Vision, 30(2), 79-116.
    
    See Also
    --------
    filter_log : 2D Laplacian of Gaussian filter
    convn_fft : FFT-based N-dimensional convolution
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    
    if not isinstance(sigma, (int, float)) or sigma <= 0:
        raise ValueError("sigma must be a positive scalar")
    
    # Get image dimensions
    dims = image.ndim
    if image.size == max(image.shape):
        # 1D array
        dims = 1
    
    # Handle spacing parameter
    if isinstance(spacing, (int, float)):
        spacing = np.full(dims, float(spacing))
    else:
        spacing = np.array(spacing, dtype=float)
        if spacing.size == 1:
            spacing = np.full(dims, spacing[0])
        elif spacing.size != dims:
            raise ValueError(f"spacing must be scalar or have {dims} elements")
    
    # Create sigma array for all dimensions
    sigma_vec = np.full(dims, sigma)
    
    # Adjust sigma according to pixel spacing
    sigma_vec = sigma_vec / spacing
    
    # Compute LoG kernel
    w = np.ceil(4 * sigma_vec).astype(int)
    
    # Create coordinate grids
    xrange = [np.arange(-w[i], w[i] + 1) for i in range(dims)]
    
    if dims > 1:
        x = np.meshgrid(*xrange, indexing='ij')
    else:
        x = [xrange[0]]
    
    # Compute Gaussian and curvature components
    G = np.ones(x[0].shape)
    C = np.zeros(x[0].shape)
    
    for i in range(dims):
        x2 = x[i] ** 2
        sigma_i = sigma_vec[i]
        
        # Gaussian component
        G *= np.exp(-x2 / (2 * sigma_i ** 2))
        
        # Curvature (Laplacian) component
        if use_normalized_derivatives:
            # Scale-normalized: multiply by sigma^2
            C += (x2 - sigma_i ** 2) / (sigma_i ** 2)
        else:
            # Standard form
            C += (x2 - sigma_i ** 2) / (sigma_i ** 4)
    
    # Normalize Gaussian if requested
    if use_normalized_gaussian:
        G = G / np.sum(G)
    
    # Compute LoG kernel
    log_kernel = C * G
    
    # Zero-mean the kernel to remove DC component
    log_kernel = log_kernel - np.mean(log_kernel)
    
    # Pad image
    if dims > 1:
        padsize = [(w[i], w[i]) for i in range(dims)]
    else:
        padsize = [(w[0], w[0])]
    
    # Handle border condition
    if isinstance(border_condition, str):
        if border_condition == 'symmetric':
            mode = 'reflect'
        elif border_condition == 'replicate':
            mode = 'edge'
        elif border_condition == 'antisymmetric':
            mode = 'reflect'
        elif border_condition == 'constant':
            mode = 'constant'
        elif border_condition == 'wrap':
            mode = 'wrap'
        else:
            raise ValueError(f"Unknown border condition: {border_condition}")
        
        im_padded = np.pad(image, padsize, mode=mode)
    else:
        # Constant value padding
        im_padded = np.pad(image, padsize, mode='constant', 
                          constant_values=border_condition)
    
    # Apply convolution using FFT
    log_response = convn_fft(im_padded, log_kernel, shape='valid')
    
    return log_response


def filter_log_nd_with_kernel(
    image: np.ndarray,
    sigma: float,
    spacing: Union[float, Tuple[float, ...]] = 1.0,
    border_condition: Union[str, float] = 'symmetric',
    use_normalized_derivatives: bool = False,
    use_normalized_gaussian: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply LoG filter and return both response and kernel.
    
    This is a convenience function that returns both the filtered image
    and the kernel used.
    
    Parameters
    ----------
    Same as filter_log_nd
    
    Returns
    -------
    log_response : np.ndarray
        LoG filtered image
    log_kernel : np.ndarray
        The LoG kernel used for filtering
        
    See Also
    --------
    filter_log_nd : Main LoG filtering function
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    
    if not isinstance(sigma, (int, float)) or sigma <= 0:
        raise ValueError("sigma must be a positive scalar")
    
    # Get image dimensions
    dims = image.ndim
    if image.size == max(image.shape):
        dims = 1
    
    # Handle spacing parameter
    if isinstance(spacing, (int, float)):
        spacing = np.full(dims, float(spacing))
    else:
        spacing = np.array(spacing, dtype=float)
        if spacing.size == 1:
            spacing = np.full(dims, spacing[0])
        elif spacing.size != dims:
            raise ValueError(f"spacing must be scalar or have {dims} elements")
    
    # Create sigma array
    sigma_vec = np.full(dims, sigma)
    sigma_vec = sigma_vec / spacing
    
    # Compute LoG kernel
    w = np.ceil(4 * sigma_vec).astype(int)
    xrange = [np.arange(-w[i], w[i] + 1) for i in range(dims)]
    
    if dims > 1:
        x = np.meshgrid(*xrange, indexing='ij')
    else:
        x = [xrange[0]]
    
    G = np.ones(x[0].shape)
    C = np.zeros(x[0].shape)
    
    for i in range(dims):
        x2 = x[i] ** 2
        sigma_i = sigma_vec[i]
        G *= np.exp(-x2 / (2 * sigma_i ** 2))
        
        if use_normalized_derivatives:
            C += (x2 - sigma_i ** 2) / (sigma_i ** 2)
        else:
            C += (x2 - sigma_i ** 2) / (sigma_i ** 4)
    
    if use_normalized_gaussian:
        G = G / np.sum(G)
    
    log_kernel = C * G
    log_kernel = log_kernel - np.mean(log_kernel)
    
    # Pad and convolve
    if dims > 1:
        padsize = [(w[i], w[i]) for i in range(dims)]
    else:
        padsize = [(w[0], w[0])]
    
    if isinstance(border_condition, str):
        if border_condition == 'symmetric':
            mode = 'reflect'
        elif border_condition == 'replicate':
            mode = 'edge'
        elif border_condition == 'antisymmetric':
            mode = 'reflect'
        elif border_condition == 'constant':
            mode = 'constant'
        elif border_condition == 'wrap':
            mode = 'wrap'
        else:
            raise ValueError(f"Unknown border condition: {border_condition}")
        
        im_padded = np.pad(image, padsize, mode=mode)
    else:
        im_padded = np.pad(image, padsize, mode='constant', 
                          constant_values=border_condition)
    
    log_response = convn_fft(im_padded, log_kernel, shape='valid')
    
    return log_response, log_kernel
