"""
Color conversion utilities.

Ported from MATLAB imageProcessing/ directory.
"""

import numpy as np
from typing import Optional
from .contrast import scale_contrast


def ch2rgb(
    R: Optional[np.ndarray] = None,
    G: Optional[np.ndarray] = None,
    B: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate an RGB image from grayscale input channels.
    
    Channels are normalized to [0..255] and converted to uint8.
    Input channels can be None (will be treated as zeros).
    
    Args:
        R: Red channel (2D array) or None
        G: Green channel (2D array) or None
        B: Blue channel (2D array) or None
        
    Returns:
        np.ndarray: RGB image as uint8 array with shape (height, width, 3)
        
    Examples:
        >>> red = np.random.rand(100, 100) * 255
        >>> green = np.random.rand(100, 100) * 255
        >>> rgb = ch2rgb(red, green, None)
        
    Original MATLAB function: ch2rgb.m
    Author: Francois Aguet (June 2010)
    """
    # Determine image dimensions from non-None channel
    channels = [ch for ch in [R, G, B] if ch is not None]
    
    if not channels:
        raise ValueError("At least one channel must be provided")
    
    ny, nx = channels[0].shape
    
    # Create zero arrays for None channels
    if R is None:
        R = np.zeros((ny, nx))
    if G is None:
        G = np.zeros((ny, nx))
    if B is None:
        B = np.zeros((ny, nx))
    
    # Scale each channel to [0, 255] and stack
    R_scaled = scale_contrast(R, range_out=(0, 255))
    G_scaled = scale_contrast(G, range_out=(0, 255))
    B_scaled = scale_contrast(B, range_out=(0, 255))
    
    # Stack and convert to uint8
    im_rgb = np.stack([R_scaled, G_scaled, B_scaled], axis=2).astype(np.uint8)
    
    return im_rgb
