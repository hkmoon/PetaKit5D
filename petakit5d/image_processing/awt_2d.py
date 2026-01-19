"""
2D A Trou Wavelet Transform implementation.

Ported from MATLAB PetaKit5D imageProcessing/awt.m
"""

import numpy as np
from typing import Optional, Tuple


def awt(image: np.ndarray, n_bands: Optional[int] = None) -> np.ndarray:
    """
    Compute the A Trou Wavelet Transform of a 2D image.
    
    The A Trou algorithm uses progressively dilated convolution kernels
    to decompose an image into multiple scales. The convolution kernel is
    [1, 4, 6, 4, 1]/16 with spacing 2^(k-1) at scale k.
    
    Reference:
    J.-L. Starck, F. Murtagh, A. Bijaoui, "Image Processing and Data
    Analysis: The Multiscale Approach", Cambridge Press, Cambridge, 2000.
    
    Parameters
    ----------
    image : np.ndarray
        Input 2D image (N x M)
    n_bands : int, optional
        Number of scales to decompose. Default is ceil(max(log2(N), log2(M)))
        where (N, M) = image.shape
    
    Returns
    -------
    W : np.ndarray
        Wavelet coefficients array of size (N, M, n_bands+1).
        - W[:, :, 0:n_bands] contains the detail images (wavelet coefficients)
          at scales k = 1...n_bands
        - W[:, :, n_bands] contains the last approximation image A_K
    
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(128, 128)
    >>> W = awt(img)  # Uses default n_bands
    >>> W = awt(img, n_bands=5)  # Decompose to 5 scales
    >>> # Perfect reconstruction (approximately due to numerical precision)
    >>> reconstructed = np.sum(W[:, :, :-1], axis=2) + W[:, :, -1]
    
    Notes
    -----
    - The transform maintains perfect reconstruction: 
      image ≈ Σ(detail_k) + approximation
    - Uses separable convolution for efficiency
    - Symmetric padding is used at borders
    - The algorithm is translation-invariant (unlike standard DWT)
    
    See Also
    --------
    awt_1d : 1D A Trou Wavelet Transform
    awt_denoising : Denoising using A Trou wavelets
    """
    if image.ndim != 2:
        raise ValueError(f"Input must be a 2D image, got shape {image.shape}")
    
    N, M = image.shape
    
    # Default n_bands
    K = int(np.ceil(max(np.log2(N), np.log2(M))))
    
    if n_bands is None:
        n_bands = K
    else:
        if n_bands < 1 or n_bands > K:
            raise ValueError(f"n_bands must be in range [1, {K}], got {n_bands}")
    
    # Initialize output
    W = np.zeros((N, M, n_bands + 1), dtype=np.float64)
    
    # Convert to float
    image = image.astype(np.float64)
    last_approx = image.copy()
    
    # Compute wavelet decomposition
    for k in range(1, n_bands + 1):
        new_approx = _convolve_awt(last_approx, k)
        W[:, :, k - 1] = last_approx - new_approx
        last_approx = new_approx
    
    # Store final approximation
    W[:, :, n_bands] = last_approx
    
    return W


def _convolve_awt(image: np.ndarray, scale: int) -> np.ndarray:
    """
    Convolve image with dilated A Trou kernel at given scale.
    
    Uses separable convolution with [1, 4, 6, 4, 1]/16 kernel
    with dilation 2^(scale-1).
    
    Parameters
    ----------
    image : np.ndarray
        Input 2D image
    scale : int
        Scale level (k), determines dilation factor 2^(k-1)
    
    Returns
    -------
    filtered : np.ndarray
        Convolved image
    """
    N, M = image.shape
    k1 = 2 ** (scale - 1)  # Current dilation
    k2 = 2 ** scale         # Next dilation level for padding
    
    # Pad image for column convolution
    tmp = np.pad(image, ((k2, k2), (0, 0)), mode='edge')
    
    # Convolve columns with dilated kernel [1, 4, 6, 4, 1]/16
    result = np.zeros_like(image)
    for i in range(N):
        idx = i + k2
        result[i, :] = (6 * tmp[idx, :] + 
                       4 * (tmp[idx + k1, :] + tmp[idx - k1, :]) +
                       tmp[idx + k2, :] + tmp[idx - k2, :])
    
    # Scale by 1/16
    result *= 0.0625
    
    # Pad for row convolution
    tmp = np.pad(result, ((0, 0), (k2, k2)), mode='edge')
    
    # Convolve rows with dilated kernel
    for j in range(M):
        idx = j + k2
        result[:, j] = (6 * tmp[:, idx] + 
                       4 * (tmp[:, idx + k1] + tmp[:, idx - k1]) +
                       tmp[:, idx + k2] + tmp[:, idx - k2])
    
    # Scale by 1/16
    result *= 0.0625
    
    return result
