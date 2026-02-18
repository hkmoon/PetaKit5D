"""
Bilateral filtering for edge-preserving smoothing.

This module implements bilateral filtering based on trigonometric expansion
of the Gaussian range kernel as described in Chaudhury et al., IEEE Trans. 
Imag. Proc. 2011.

Author: Converted from MATLAB (Francois Aguet, 07/12/2011)
Date: 2026-01-09
"""

import numpy as np
from scipy import stats
from typing import Tuple
from .filters import filter_gauss_2d


def bilateral_filter(img: np.ndarray, sigma_s: float, sigma_r: float) -> np.ndarray:
    """
    Apply bilateral filter to an image for edge-preserving smoothing.
    
    The bilateral filter is a non-linear filter that smooths images while
    preserving edges. It combines spatial Gaussian filtering with range
    (intensity) filtering. The implementation uses a trigonometric expansion
    of the Gaussian range kernel for computational efficiency.
    
    Parameters
    ----------
    img : np.ndarray
        Input 2D image array
    sigma_s : float
        Spatial sigma (controls Gaussian blur in spatial domain)
    sigma_r : float
        Range/intensity sigma (controls how much intensity differences matter)
        
    Returns
    -------
    np.ndarray
        Bilaterally filtered image
        
    Notes
    -----
    Based on Chaudhury et al., "Fast bilateral filtering using trigonometric
    range kernels," IEEE Trans. Image Processing, 2011.
    
    The bilateral filter is particularly useful for:
    - Noise reduction while preserving edges
    - Detail enhancement
    - HDR tone mapping
    - Flash/no-flash photography merging
    
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(100, 100)
    >>> filtered = bilateral_filter(img, sigma_s=2.0, sigma_r=0.1)
    >>> filtered.shape
    (100, 100)
    """
    if img.ndim != 2:
        raise ValueError("bilateral_filter only supports 2D images")
    
    T = np.max(img)
    
    # Handle constant-zero or near-zero images
    if T == 0 or T < 1e-10:
        return img.copy()
    
    ny, nx = img.shape
    
    gamma = np.pi / (2 * T)
    rho = gamma * sigma_r
    
    # Lookup table for optimal number of coefficients (see ref. [1])
    sigma_r_threshold = np.array([200, 150, 100, 80, 60, 50, 40]) / 255 * T
    N0 = np.array([1, 2, 3, 4, 5, 7, 9])
    
    # Determine number of terms N in the expansion
    if sigma_r >= sigma_r_threshold[-1]:
        idx = np.where(sigma_r >= sigma_r_threshold)[0]
        if len(idx) > 0:
            N = N0[idx[0]]
        else:
            N = N0[-1]
    elif sigma_r > 1 / gamma**2:
        N = 50  # arbitrary high N
    else:
        N = int(np.ceil(1 / (gamma * sigma_r)**2))
    
    # Determine cutoff for insignificant coefficients
    if N > 20:
        # Use normal approximation for binomial distribution
        bounds = stats.norm.ppf([0.025, 0.975], loc=N/2, scale=np.sqrt(N)/2)
        bounds = [int(np.floor(bounds[0])), int(np.ceil(bounds[1]))]
    else:
        bounds = [0, N]
    
    # Binomial coefficients/weights
    c = stats.binom.pmf(np.arange(N + 1), N, 0.5)
    
    h = np.zeros((ny, nx), dtype=complex)
    g = np.zeros((ny, nx), dtype=complex)
    
    for n in range(bounds[0], bounds[1] + 1):
        tmp = 1j * gamma * (2*n - N) * img / (rho * np.sqrt(N))
        hx = np.exp(tmp)
        d = c[n] * np.exp(-tmp)
        
        # Apply Gaussian filter
        h_filtered, _ = filter_gauss_2d(hx, sigma_s)
        g_filtered, _ = filter_gauss_2d(img * hx, sigma_s)
        
        h = h + d * h_filtered
        g = g + d * g_filtered
    
    bf = np.real(g / h)
    
    return bf
