"""
A Trou Wavelet Denoising implementation.

Ported from MATLAB PetaKit5D imageProcessing/awtDenoising.m
"""

import numpy as np
from typing import Optional
from scipy.stats import norm

from .awt_2d import awt


def awt_denoising(
    image: np.ndarray,
    n_bands: Optional[int] = None,
    include_low_band: bool = True,
    n_sigma: float = 3.0
) -> np.ndarray:
    """
    Reconstruct an image from soft thresholding of its A Trou wavelet coefficients.
    
    This function performs wavelet-based denoising by decomposing the image
    using the A Trou Wavelet Transform, applying soft thresholding to the
    detail coefficients, and reconstructing the denoised image.
    
    Reference:
    "Olivo-Marin J.C. 2002. Extraction of spots in biological images using
    multiscale products. Pattern Recognit. 35: 1989–1996."
    
    Parameters
    ----------
    image : np.ndarray
        Input 2D image to denoise (N x M)
    n_bands : int, optional
        Number of wavelet scales to use. Default is ceil(max(log2(N), log2(M)))
        where (N, M) = image.shape
    include_low_band : bool, optional
        Whether to add the approximation image A_K (lowest frequency band)
        to the reconstructed image. Default is True.
    n_sigma : float, optional
        Number of standard deviations to use in the soft threshold.
        Default is 3.0. Higher values preserve more detail but less denoising.
    
    Returns
    -------
    reconstructed : np.ndarray
        Denoised/reconstructed image of same shape as input
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create noisy image
    >>> clean = np.random.rand(128, 128)
    >>> noisy = clean + 0.1 * np.random.randn(128, 128)
    >>> # Denoise with default parameters
    >>> denoised = awt_denoising(noisy)
    >>> # More aggressive denoising (lower threshold)
    >>> denoised_aggressive = awt_denoising(noisy, n_sigma=2.0)
    >>> # Preserve more detail (higher threshold)
    >>> denoised_detail = awt_denoising(noisy, n_sigma=4.0)
    
    Notes
    -----
    - Uses Median Absolute Deviation (MAD) for robust noise estimation
    - Soft thresholding zeros out coefficients below threshold
    - The threshold is computed as: mad_factor * MAD(coefficients)
      where mad_factor = n_sigma / norminv(0.75)
    - Works best for Gaussian noise
    - Higher n_sigma values preserve more signal but less denoising
    - Lower n_sigma values provide more denoising but may remove signal
    
    See Also
    --------
    awt : 2D A Trou Wavelet Transform
    """
    if image.ndim != 2:
        raise ValueError(f"Input must be a 2D image, got shape {image.shape}")
    
    N, M = image.shape
    
    # Default n_bands
    if n_bands is None:
        n_bands = int(np.ceil(max(np.log2(N), np.log2(M))))
    else:
        K = int(np.ceil(max(np.log2(N), np.log2(M))))
        if n_bands < 1 or n_bands > K:
            raise ValueError(f"n_bands must be in range [1, {K}], got {n_bands}")
    
    # Compute A Trou Wavelet Transform
    W = awt(image, n_bands)
    
    # Initialize reconstruction
    if include_low_band:
        reconstructed = W[:, :, n_bands].copy()
    else:
        reconstructed = np.zeros((N, M), dtype=np.float64)
    
    # Compute MAD factor for soft thresholding
    # norminv(0.75, 0, 1) ≈ 0.6745 (75th percentile of standard normal)
    norminv_075 = norm.ppf(0.75)  # ≈ 0.6745
    mad_factor = n_sigma / norminv_075
    
    # Apply soft thresholding to each scale
    for k in range(n_bands):
        S = W[:, :, k].copy()
        
        # Compute MAD (Median Absolute Deviation)
        # MAD = median(|x - median(x)|)
        # flag=1 in MATLAB mad() means: MAD = median(|x|) assuming median is 0
        mad_val = np.median(np.abs(S))
        
        # Compute threshold
        threshold = mad_factor * mad_val
        
        # Soft thresholding: zero out coefficients below threshold
        S[np.abs(S) < threshold] = 0
        
        # Add thresholded coefficients to reconstruction
        reconstructed += S
    
    return reconstructed
