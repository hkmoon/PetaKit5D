"""
Contrast adjustment utilities.

Ported from MATLAB imageProcessing/ directory.
"""

import numpy as np
from typing import Optional, Tuple, Union


def scale_contrast(
    input_array: np.ndarray,
    range_in: Optional[Tuple[float, float]] = None,
    range_out: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Adjust the contrast of the input signal.
    
    Args:
        input_array: Input signal/image array
        range_in: Input range as (min, max). If None, uses [min(array), max(array)]
        range_out: Output range as (min, max). Default is (0, 255)
        
    Returns:
        np.ndarray: Contrast-adjusted output array
        
    Examples:
        >>> img = np.array([[0, 50], [100, 200]])
        >>> scaled = scale_contrast(img, range_out=(0, 1))
        
    Original MATLAB function: scaleContrast.m
    Author: Francois Aguet (Last modified: 03/22/2011)
    """
    if range_in is None:
        range_in = (float(np.min(input_array)), float(np.max(input_array)))
    
    if range_out is None:
        range_out = (0.0, 255.0)
    
    range_in_diff = range_in[1] - range_in[0]
    
    if range_in_diff != 0:
        output = ((input_array.astype(float) - range_in[0]) / range_in_diff * 
                 (range_out[1] - range_out[0]) + range_out[0])
    else:
        output = np.zeros_like(input_array, dtype=float)
    
    return output


def invert_contrast(
    input_array: np.ndarray,
    range_in: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Invert the input signal, preserving the dynamic range.
    
    If a dynamic range is provided, the input first gets truncated to that range.
    This is different from simple negation - it maps [a, b] to [b, a] while
    preserving the dynamic range.
    
    Args:
        input_array: Input signal/image array
        range_in: Input range as (min, max). If None, uses [min(array), max(array)]
        
    Returns:
        np.ndarray: Contrast-inverted output array
        
    Examples:
        >>> img = np.array([[0, 50], [100, 200]])
        >>> inverted = invert_contrast(img, range_in=(0, 200))
        
    Original MATLAB function: invertContrast.m
    Author: Francois Aguet (08/09/2012)
    """
    if range_in is None:
        range_in = (float(np.min(input_array)), float(np.max(input_array)))
    
    # Create a copy to avoid modifying the input
    output = input_array.copy().astype(float)
    
    # Truncate to range if provided
    output = np.clip(output, range_in[0], range_in[1])
    
    # Invert: -input + sum(range)
    output = -output + sum(range_in)
    
    return output
