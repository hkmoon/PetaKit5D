"""
String formatting utilities.

Ported from MATLAB mat2str_comma.m
"""

import numpy as np
from typing import Union, Optional


def mat2str_comma(A: Union[list, np.ndarray], sn: Optional[int] = None) -> str:
    """
    Convert matrix/array to string with comma separation.
    
    Args:
        A: Input vector/array
        sn: Significant number of decimal places. If None, treats as integers.
        
    Returns:
        str: String representation with comma-separated values in brackets
        
    Examples:
        >>> mat2str_comma([1, 2, 3])
        '[1,2,3]'
        >>> mat2str_comma([1.5, 2.7, 3.9], sn=2)
        '[1.50,2.70,3.90]'
        
    Original MATLAB function: mat2str_comma.m
    Author: Xiongtao Ruan
    """
    # Convert to numpy array for consistent handling
    if not isinstance(A, np.ndarray):
        A = np.array(A)
    
    # Flatten to 1D
    A_flat = A.flatten()
    
    if sn is None:
        # Integer format
        values = ','.join(str(int(x)) for x in A_flat)
    else:
        # Fixed decimal format
        format_str = f'{{:.{sn}f}}'
        values = ','.join(format_str.format(x) for x in A_flat)
    
    return f'[{values}]'
