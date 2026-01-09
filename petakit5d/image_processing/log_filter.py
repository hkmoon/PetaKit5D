"""
Laplacian of Gaussian (LoG) filtering using FFT for exact convolution.

This module provides FFT-based LoG filtering for 1D, 2D, and 3D signals,
computing the exact convolution unlike approximate methods.

Author: Converted from MATLAB (Francois Aguet, 11/11/2010)
Date: 2026-01-09
"""

import numpy as np
from typing import Union


def filter_log(signal: np.ndarray, sigma: float) -> np.ndarray:
    """
    Filter a signal with a Laplacian of Gaussian filter using FFT.
    
    Unlike the built-in Matlab function fspecial('Laplacian'), this function
    computes the exact convolution using FFTs for better accuracy and speed.
    
    The Laplacian of Gaussian is computed in the Fourier domain as:
    LoG(ω) = |ω|² · exp(-σ²|ω|²/2)
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal, can be 1D, 2D, or 3D array.
    sigma : float
        Standard deviation of the Gaussian kernel.
        Larger values result in more smoothing.
        
    Returns
    -------
    filtered : np.ndarray
        LoG filtered signal with the same shape as input.
        
    Examples
    --------
    >>> # 1D signal filtering
    >>> signal_1d = np.random.rand(100)
    >>> filtered_1d = filter_log(signal_1d, sigma=2.0)
    
    >>> # 2D image filtering
    >>> image = np.random.rand(100, 100)
    >>> filtered_2d = filter_log(image, sigma=1.5)
    
    >>> # 3D volume filtering
    >>> volume = np.random.rand(50, 50, 50)
    >>> filtered_3d = filter_log(volume, sigma=2.5)
    
    Notes
    -----
    The Laplacian of Gaussian is useful for:
    - Edge detection (zero-crossings indicate edges)
    - Blob detection (local extrema indicate blobs)
    - Scale-space analysis
    
    The function automatically determines dimensionality from the input shape.
    For 1D arrays, it treats them as vectors regardless of their shape.
    
    The FFT-based approach is exact and generally faster than spatial
    convolution for sigma >= 2.
    """
    # Determine dimensionality
    if signal.size == max(signal.shape):
        # 1D signal (vector)
        dims = 1
    else:
        dims = signal.ndim
    
    if dims == 1:
        # 1D filtering
        signal_flat = signal.ravel()
        nx = len(signal_flat)
        
        # Create frequency grid
        w1 = np.arange(-nx//2, nx//2)
        w1 = np.fft.fftshift(w1)
        w1 = w1 * 2 * np.pi / nx
        
        # Compute FFT
        I = np.fft.fft(signal_flat)
        
        # Apply LoG filter in frequency domain
        LoG = w1**2 * np.exp(-0.5 * sigma**2 * w1**2)
        
        # Inverse FFT and take real part
        result = np.real(np.fft.ifft(I * LoG))
        
        # Reshape to original shape
        return result.reshape(signal.shape)
        
    elif dims == 2:
        # 2D filtering
        ny, nx = signal.shape
        
        # Create frequency grids
        w1, w2 = np.meshgrid(
            np.arange(-nx//2, nx//2),
            np.arange(-ny//2, ny//2)
        )
        w1 = np.fft.fftshift(w1)
        w2 = np.fft.fftshift(w2)
        
        w1 = w1 * 2 * np.pi / nx
        w2 = w2 * 2 * np.pi / ny
        
        # Compute FFT
        I = np.fft.fft2(signal)
        
        # Apply LoG filter in frequency domain
        w_squared = w1**2 + w2**2
        LoG = w_squared * np.exp(-0.5 * sigma**2 * w_squared)
        
        # Inverse FFT and take real part
        return np.real(np.fft.ifft2(I * LoG))
        
    elif dims == 3:
        # 3D filtering
        ny, nx, nz = signal.shape
        
        # Create frequency grids
        w1, w2, w3 = np.meshgrid(
            np.arange(-nx//2, nx//2),
            np.arange(-ny//2, ny//2),
            np.arange(-nz//2, nz//2),
            indexing='xy'
        )
        w1 = np.fft.fftshift(w1)
        w2 = np.fft.fftshift(w2)
        w3 = np.fft.fftshift(w3)
        
        w1 = w1 * 2 * np.pi / nx
        w2 = w2 * 2 * np.pi / ny
        w3 = w3 * 2 * np.pi / nz
        
        # Compute FFT
        I = np.fft.fftn(signal)
        
        # Apply LoG filter in frequency domain
        w_squared = w1**2 + w2**2 + w3**2
        LoG = w_squared * np.exp(-0.5 * sigma**2 * w_squared)
        
        # Inverse FFT and take real part
        return np.real(np.fft.ifftn(I * LoG))
    
    else:
        raise ValueError(f"Unsupported number of dimensions: {dims}. Must be 1, 2, or 3.")
