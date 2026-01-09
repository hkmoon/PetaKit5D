"""
FFT-based N-dimensional convolution.

This module provides efficient N-dimensional convolution using FFT,
which is faster than spatial convolution for large kernels.

Author: Converted from MATLAB (Deepak Roy Chittajallu)
Date: 2026-01-09
"""

import numpy as np
from typing import Union, Tuple, Optional, Literal


def convn_fft(
    A: np.ndarray,
    B: np.ndarray,
    shape: Literal['full', 'same', 'valid'] = 'full',
    dims: Optional[Union[int, Tuple[int, ...]]] = None,
    use_power_of_two: bool = False
) -> np.ndarray:
    """
    Perform FFT-based N-dimensional convolution.
    
    This function uses the Fourier transform convolution theorem: the FT of 
    the convolution equals the product of the FTs of the input functions.
    It's particularly efficient for large kernels.
    
    Parameters
    ----------
    A : np.ndarray
        First input array
    B : np.ndarray
        Second input array (convolution kernel)
    shape : {'full', 'same', 'valid'}, optional
        Controls the size of the output:
        - 'full': (default) returns the full N-D convolution
        - 'same': returns central part same size as A
        - 'valid': returns only parts computed without zero-padding
    dims : int or tuple of int, optional
        Dimensions along which to perform convolution.
        By default, all dimensions are used.
    use_power_of_two : bool, optional
        If True, rounds dimensions to nearest power of 2 by zero-padding
        during FFT. Faster but requires more memory. Default: False
        
    Returns
    -------
    np.ndarray
        Convolved array
        
    Notes
    -----
    Complexity: O((na+nb)*log(na+nb)) where na/nb are lengths of A and B
    
    Usage recommendations:
    - 1D: faster than np.convolve for nA, nB > 1000
    - 2D: faster than scipy.signal.convolve2d for nA, nB > 20
    - 3D: faster than scipy.signal.convolve for nA, nB > 5
    
    Examples
    --------
    >>> import numpy as np
    >>> A = np.random.rand(100, 100, 50)
    >>> B = np.random.rand(5, 5, 5)
    >>> C = convn_fft(A, B, shape='same')
    >>> C.shape
    (100, 100, 50)
    """
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise TypeError("A and B must be numpy arrays")
    
    nd = max(A.ndim, B.ndim)
    
    # Handle dims parameter
    if dims is None:
        dims = tuple(range(nd))
    elif isinstance(dims, int):
        dims = (dims,)
    else:
        dims = tuple(sorted(set(dims)))
    
    # Validate dims
    for dim in dims:
        if dim < 0 or dim >= nd:
            raise ValueError(f"Dimension {dim} out of range for {nd}D arrays")
    
    # Define truncation function based on shape
    if shape == 'full':
        def ifun(m, n):
            return slice(0, m + n - 1)
    elif shape == 'same':
        def ifun(m, n):
            start = int(np.ceil((n - 1) / 2))
            return slice(start, start + m)
    elif shape == 'valid':
        def ifun(m, n):
            return slice(n - 1, m)
    else:
        raise ValueError(f"Unknown shape: {shape}")
    
    # Check if inputs are real
    ab_real = np.isrealobj(A) and np.isrealobj(B)
    
    # Function to determine FFT length
    if use_power_of_two:
        def lfftfun(l):
            return 2 ** int(np.ceil(np.log2(l)))
    else:
        def lfftfun(l):
            return l
    
    # Compute FFTs
    slices = [slice(None)] * nd
    
    for dim in dims:
        m = A.shape[dim] if dim < A.ndim else 1
        n = B.shape[dim] if dim < B.ndim else 1
        
        l = lfftfun(m + n - 1)
        
        A = np.fft.fft(A, n=l, axis=dim)
        B = np.fft.fft(B, n=l, axis=dim)
        
        slices[dim] = ifun(m, n)
    
    # Multiply FFTs element-wise
    C = A * B
    
    # Transform back from frequency domain
    for dim in dims:
        C = np.fft.ifft(C, axis=dim)
    
    # Truncate results
    C = C[tuple(slices)]
    
    # Make sure result is real if inputs were real
    if ab_real:
        C = np.real(C)
    
    return C
