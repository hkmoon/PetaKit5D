"""
Visualization utilities for image overlay and display.

This module provides functions for creating RGB overlays and visualizations.
"""

import numpy as np
from typing import Union, List, Tuple, Optional
from .contrast import scale_contrast


def rgb_overlay(
    img: np.ndarray,
    masks: Union[np.ndarray, List[np.ndarray]],
    colors: Union[Tuple[float, float, float], List[Tuple[float, float, float]]],
    i_range: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Overlay color masks on a grayscale input image.
    
    Creates an RGB image by overlaying colored masks on a grayscale background.
    Nonzero elements in the masks are colored according to the specified colors.
    
    Parameters
    ----------
    img : np.ndarray
        Grayscale input image (2D array)
    masks : np.ndarray or list of np.ndarray
        Overlay masks. Can be a single mask or list of masks.
        Nonzero elements are colored.
    colors : tuple or list of tuples
        RGB color specifications. Each color is a 3-element tuple with
        values in [0, 1]. Can be a single color or list of colors.
    i_range : tuple of 2 floats, optional
        Dynamic range of input image [min, max]. If None, uses full range.
    
    Returns
    -------
    img_rgb : np.ndarray
        uint8 RGB image with shape (M, N, 3)
    
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(100, 100) * 255
    >>> mask1 = np.zeros((100, 100), dtype=bool)
    >>> mask1[20:40, 20:40] = True
    >>> mask2 = np.zeros((100, 100), dtype=bool)
    >>> mask2[60:80, 60:80] = True
    >>> # Red and green overlays
    >>> rgb = rgb_overlay(img, [mask1, mask2], [(1, 0, 0), (0, 1, 0)])
    
    Notes
    -----
    - Based on MATLAB rgbOverlay by Francois Aguet (April 2011)
    - Masked regions replace the grayscale background with colored overlays
    - Multiple overlays are combined additively, clipped to maximum 1.0
    - Where overlays exist, grayscale information is suppressed
    
    References
    ----------
    Francois Aguet, April 2011
    """
    # Normalize inputs
    if not isinstance(masks, list):
        masks = [masks]
    if not isinstance(colors, list):
        colors = [colors]
    
    nm = len(masks)
    
    if len(colors) != nm:
        raise ValueError(f"Number of masks ({nm}) must match number of colors ({len(colors)})")
    
    # Ensure all masks are boolean
    masks = [np.asarray(m, dtype=bool) for m in masks]
    
    # Scale contrast of grayscale image to [0, 1]
    img = scale_contrast(img, range_in=i_range, range_out=(0, 1))
    
    # Find combined mask (any overlay)
    comb_idx = np.zeros(img.shape, dtype=bool)
    for mask in masks:
        comb_idx |= mask
    
    # Start with grayscale, but suppress where overlays exist
    img_rgb = img.copy()
    img_rgb[comb_idx] = 0
    
    # Replicate to create RGB image
    img_rgb = np.stack([img_rgb] * 3, axis=2)
    
    # Add each color channel
    for c in range(3):
        # Accumulate masks weighted by their colors for this channel
        c_mask = np.zeros(img.shape, dtype=np.float64)
        for k in range(nm):
            c_mask += colors[k][c] * masks[k].astype(np.float64)
        
        # Clip to [0, 1]
        c_mask = np.minimum(c_mask, 1.0)
        
        # Where mask is 1, show original grayscale; otherwise show colored overlay
        idx = (c_mask == 1.0)
        tmp = img_rgb[:, :, c].copy()
        tmp[idx] = img[idx]
        img_rgb[:, :, c] = tmp
    
    # Convert to uint8
    img_rgb = np.clip(img_rgb * 255, 0, 255).astype(np.uint8)
    
    return img_rgb


def z_proj_image(
    images: np.ndarray,
    proj_type: str = 'max'
) -> np.ndarray:
    """
    Perform Z-projection of a 3D image stack.
    
    Projects a 3D stack along the Z-axis (first dimension) using various
    projection methods.
    
    Parameters
    ----------
    images : np.ndarray
        3D array with shape (Z, Y, X) or (Z, Y, X, C) for multi-channel
    proj_type : str, optional
        Projection type:
        - 'max': Maximum intensity projection (default)
        - 'mean' or 'ave': Average intensity projection
        - 'median' or 'med': Median intensity projection
        - 'min': Minimum intensity projection
    
    Returns
    -------
    projection : np.ndarray
        2D projected image with shape (Y, X) or (Y, X, C)
    
    Examples
    --------
    >>> import numpy as np
    >>> stack = np.random.rand(50, 100, 100) * 255
    >>> mip = z_proj_image(stack, 'max')
    >>> avg = z_proj_image(stack, 'mean')
    
    Notes
    -----
    - Based on MATLAB zProjImage
    - Simplified version that works with numpy arrays directly
    - For file I/O, use separate functions
    
    See Also
    --------
    project_3d_to_2d : More general 3D to 2D projection function
    """
    if images.ndim < 3:
        raise ValueError("Input must be at least 3-dimensional")
    
    proj_type = proj_type.lower()
    
    if proj_type == 'max':
        projection = np.max(images, axis=0)
    elif proj_type in ['mean', 'ave', 'average']:
        projection = np.nanmean(images, axis=0)
    elif proj_type in ['median', 'med']:
        projection = np.nanmedian(images, axis=0)
    elif proj_type == 'min':
        projection = np.min(images, axis=0)
    else:
        raise ValueError(f"Unknown projection type: {proj_type}. "
                         f"Must be 'max', 'mean'/'ave', 'median'/'med', or 'min'")
    
    return projection
