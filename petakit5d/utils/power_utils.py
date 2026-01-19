"""
Fast power computation utility.

Author: Xiongtao Ruan
Python port: GitHub Copilot
"""

import numpy as np
from typing import Union


def fast_power(base: Union[float, np.ndarray], exponent: int) -> Union[float, np.ndarray]:
    """
    Compute base raised to integer exponent using fast exponentiation.
    
    Uses binary exponentiation algorithm (exponentiation by squaring) for
    efficient computation with O(log n) multiplications instead of O(n).
    
    Parameters
    ----------
    base : float or ndarray
        Base value(s) to raise to power
    exponent : int
        Integer exponent (can be negative)
        
    Returns
    -------
    result : float or ndarray
        base raised to exponent
        
    Examples
    --------
    >>> fast_power(2, 10)
    1024.0
    
    >>> fast_power(2, -3)
    0.125
    
    >>> import numpy as np
    >>> fast_power(np.array([2, 3, 4]), 3)
    array([ 8., 27., 64.])
    
    Notes
    -----
    This implementation uses bitwise operations for efficiency:
    - Checks least significant bit to determine if current base should multiply
    - Squares base and right-shifts exponent in each iteration
    - Only O(log n) iterations required
    
    For negative exponents, takes reciprocal of base first.
    """
    # Convert to numpy array for consistent handling
    base_arr = np.asarray(base, dtype=float)
    
    # Handle negative exponent
    if exponent < 0:
        base_arr = 1.0 / base_arr
        exponent = -exponent
    
    # Quick return for exponent == 1
    if exponent == 1:
        if np.isscalar(base):
            return float(base_arr)
        return base_arr
    
    # Initialize result
    result = np.ones_like(base_arr, dtype=float)
    
    # Binary exponentiation
    while exponent > 0:
        # Check least significant bit
        if exponent & 1:  # Same as bitget(exponent, 1) in MATLAB
            result = result * base_arr
        
        # Square the base and right-shift exponent
        base_arr = base_arr * base_arr
        exponent = exponent >> 1  # Same as bitshift(exponent, -1)
    
    # Return scalar if input was scalar
    if np.isscalar(base):
        return float(result)
    
    return result
