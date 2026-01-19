"""
A Trou Wavelet Transform for 1D signal processing.

Based on algorithm from:
J.-L. Starck, F. Murtagh, A. Bijaoui, "Image Processing and Data
Analysis: The Multiscale Approach", Cambridge Press, Cambridge, 2000.

Authors: Sylvain Berlemont (2009), Francois Aguet (2010)
Python port: GitHub Copilot
"""

import numpy as np
from typing import Optional, Tuple
from scipy import ndimage


def awt_1d(signal: np.ndarray, n_bands: Optional[int] = None) -> np.ndarray:
    """
    Compute the A Trou Wavelet Transform of a 1D signal.
    
    The A Trou algorithm computes a wavelet decomposition using a specific
    convolution kernel that progressively dilates by powers of 2. This creates
    a multi-scale representation useful for denoising and feature detection.
    
    Parameters
    ----------
    signal : ndarray
        1D input signal
    n_bands : int, optional
        Number of wavelet bands to compute (inclusive).
        Default is ceil(log2(N)) where N is the signal length.
        Must be in range [1, ceil(log2(N))]
        
    Returns
    -------
    W : ndarray, shape (N, n_bands+1)
        Wavelet coefficients organized as:
        - W[:, 0:n_bands]: Wavelet coefficients (detail images) at scales k=1...n_bands
        - W[:, n_bands]: Last approximation image A_K
        
    Raises
    ------
    ValueError
        If n_bands is not in valid range [1, ceil(log2(N))]
        
    Examples
    --------
    >>> import numpy as np
    >>> signal = np.sin(np.linspace(0, 10*np.pi, 256))
    >>> W = awt_1d(signal, n_bands=4)
    >>> W.shape
    (256, 5)
    
    >>> # First 4 columns are detail coefficients
    >>> # Last column is final approximation
    >>> detail_1 = W[:, 0]  # Finest scale details
    >>> approx_final = W[:, -1]  # Coarsest approximation
    
    Notes
    -----
    The convolution kernel used is [1, 4, 6, 4, 1]/16, applied with
    progressively increasing dilation (spacing) by powers of 2.
    
    For scale k, the kernel has spacing 2^(k-1) between coefficients.
    
    The symmetric padding ensures proper boundary handling.
    
    See Also
    --------
    awt_denoising : Denoise signal using soft thresholding of AWT coefficients
    """
    N = len(signal)
    K = int(np.ceil(np.log2(N)))
    
    if n_bands is None:
        n_bands = K
    else:
        if n_bands < 1 or n_bands > K:
            raise ValueError(
                f"Invalid range for n_bands parameter. Must be in [1, {K}], got {n_bands}"
            )
    
    # Initialize output array
    W = np.zeros((N, n_bands + 1))
    
    # Ensure signal is 1D float array
    signal = np.asarray(signal, dtype=float).ravel()
    last_A = signal.copy()
    
    # Compute wavelet decomposition
    for k in range(1, n_bands + 1):
        new_A = _convolve_awt(last_A, k)
        W[:, k - 1] = last_A - new_A  # Detail coefficients
        last_A = new_A
    
    W[:, n_bands] = last_A  # Final approximation
    
    return W


def _convolve_awt(signal: np.ndarray, k: int) -> np.ndarray:
    """
    Convolve signal with dilated A Trou kernel at scale k.
    
    Parameters
    ----------
    signal : ndarray
        1D signal to convolve
    k : int
        Scale index (dilation = 2^(k-1))
        
    Returns
    -------
    F : ndarray
        Filtered signal
        
    Notes
    -----
    Kernel coefficients: [1, 4, 6, 4, 1] / 16
    Dilation at scale k: 2^(k-1)
    """
    N = len(signal)
    k1 = 2 ** (k - 1)  # Dilation factor
    k2 = 2 * k1  # Padding size
    
    # Symmetric padding
    # padarrayXT with 'symmetric' mode
    tmp = np.pad(signal, (k2, k2), mode='symmetric')
    
    # Apply dilated convolution manually
    # Kernel: [1, 4, 6, 4, 1] / 16 with spacing k1
    result = np.zeros(N)
    for i in range(k2, k2 + N):
        result[i - k2] = (
            6 * tmp[i] +
            4 * (tmp[i + k1] + tmp[i - k1]) +
            tmp[i + k2] + tmp[i - k2]
        )
    
    return result / 16.0
