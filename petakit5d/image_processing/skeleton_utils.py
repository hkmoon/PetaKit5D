"""
Skeleton utilities for binary image skeletonization in 2D and 3D.

This module provides morphological binary skeletonization using various methods:
- Thinning: Iterative thinning until no more points can be removed
- Erosion & Opening: Skeletonization via erosion and opening operations
- Divergence: Skeletonization using divergence of distance transform gradient

References:
    [1] A.K. Jain - "Fundamentals of Digital Image Processing", Prentice-Hall, 1989
    [2] K. Palagyi & A. Kuba, A Parallel 3D 12-Subiteration Thinning Algorithm,
        Graphical Models and Image Processing, 61, p. 199, (1999)
"""

import numpy as np
from scipy import ndimage
from typing import Union, Literal
import warnings

from .morphology import binary_sphere
from .morphology_thin import bw_thin


def skeleton(
    bw: np.ndarray,
    method: Literal['t', 'e', 'd'] = 't',
    div_threshold: float = -1.0
) -> np.ndarray:
    """
    Morphological binary skeletonization in 2D or 3D.
    
    This function performs skeletonization on the input (2D or 3D) binary
    matrix, and supports a variety of methods for performing this operation.
    
    Parameters
    ----------
    bw : np.ndarray
        The 2D or 3D binary matrix to skeletonize. Will be converted to boolean.
    method : {'t', 'e', 'd'}, optional
        Method to use for skeletonization:
        
        - 't' : Thinning (default). The skeleton is produced by repeated thinning
          of the input binary image until no more points are removed. For 2D
          images this uses scipy's morphological operations, while for 3D images
          it uses the bw_thin algorithm.
        
        - 'e' : Erosion & opening. As described in [1] p387. NOTE: Due to
          inherent discretization, this method DOES NOT GUARANTEE connectedness
          of the resulting skeleton [1] p. 386 Fig. 9.33, even if the original
          object is connected. However, it IS guaranteed that the returned points
          will be on the medial axis/medial surface of the input mask.
        
        - 'd' : Divergence. This method uses the divergence of the gradient of
          the distance transform of the input binary image to find skeleton
          points. Connectivity is NOT guaranteed to be preserved, and the
          resulting points are not guaranteed to be on the true medial axis.
          HOWEVER, this method is fast and is much less sensitive to minor
          variations in the shape of the input binary object than the methods
          above. The skeleton lies on ridge-lines of the distance transform.
          
    div_threshold : float, optional
        The value to threshold the divergence at to produce the skeleton
        (should be negative!). Lower (more negative) values will give fewer
        points which are more likely to lie on the true skeleton, but less
        likely to preserve connectivity. Only used when method='d'.
        Default is -1.0.
    
    Returns
    -------
    skel : np.ndarray
        The 2D or 3D binary matrix with the skeleton points.
    
    Raises
    ------
    ValueError
        If input is not 2D or 3D, or if method is not recognized.
    
    Warnings
    --------
    If div_threshold >= 0 when method='d', warns that result is not a skeleton.
    
    References
    ----------
    [1] A.K. Jain - "Fundamentals of Digital Image Processing", Prentice-Hall, 1989
    [2] K. Palagyi & A. Kuba, A Parallel 3D 12-Subiteration Thinning Algorithm,
        Graphical Models and Image Processing, 61, p. 199, (1999)
    
    Examples
    --------
    >>> import numpy as np
    >>> from petakit5d.image_processing import skeleton
    >>> 
    >>> # Create a simple 2D binary image
    >>> bw = np.zeros((50, 50), dtype=bool)
    >>> bw[10:40, 20:30] = True
    >>> 
    >>> # Skeletonize using thinning (default)
    >>> skel_t = skeleton(bw, method='t')
    >>> 
    >>> # Skeletonize using erosion & opening
    >>> skel_e = skeleton(bw, method='e')
    >>> 
    >>> # Skeletonize using divergence
    >>> skel_d = skeleton(bw, method='d', div_threshold=-2.0)
    """
    # Input validation
    if bw.ndim < 2 or bw.ndim > 3:
        raise ValueError('Must input a 2D or 3D matrix for skeletonization!')
    
    if div_threshold >= 0 and method == 'd':
        warnings.warn(
            "Non-negative threshold values are not recommended! "
            "What you're getting is NOT a skeleton!",
            UserWarning
        )
    
    # Make sure the matrix is logical
    bw = np.asarray(bw, dtype=bool)
    
    if method == 'e':
        # Skeletonization by Erosion & Opening
        
        if bw.ndim == 3:
            # 3D structuring element
            nH_1 = binary_sphere(1.0)
        else:
            # 2D disk structuring element (approximation with cross)
            nH_1 = ndimage.generate_binary_structure(2, 1)
        
        # Initialize the skeleton matrix
        skel = np.zeros_like(bw, dtype=bool)
        
        j = 1
        while np.any(bw):
            if bw.ndim == 3:
                nH = binary_sphere(float(j))
            else:
                # For 2D, create disk-like structuring element
                # using distance from center
                size = 2 * j + 1
                y, x = np.ogrid[-j:j+1, -j:j+1]
                nH = (x**2 + y**2 <= j**2)
            
            # Perform erosion at the current radius
            bw = ndimage.binary_erosion(bw, structure=nH)
            
            # Subtract the opening with unit radius
            opened = ndimage.binary_opening(bw, structure=nH_1)
            skel = skel | (bw != opened)
            
            j += 1
    
    elif method == 't':
        # Skeletonization by Thinning
        
        if bw.ndim == 2:
            # For 2D, use scipy's morphological thinning
            from skimage.morphology import skeletonize
            skel = skeletonize(bw)
        
        elif bw.ndim == 3:
            # For 3D, use our bw_thin implementation
            skel = bw_thin(bw)
    
    elif method == 'd':
        # Skeletonization by Divergence
        
        # Calculate the gradient of the distance transform of the inverse
        # of the input object
        dist = ndimage.distance_transform_edt(~bw)
        
        if bw.ndim == 2:
            # 2D gradient
            gy, gx = np.gradient(dist)
            
            # Calculate the divergence of this gradient vector field
            # and threshold it
            div = np.gradient(gx, axis=1) + np.gradient(gy, axis=0)
            skel = div < div_threshold
        
        elif bw.ndim == 3:
            # 3D gradient
            gz, gy, gx = np.gradient(dist)
            
            # Calculate the divergence
            div = (np.gradient(gx, axis=2) + 
                   np.gradient(gy, axis=1) + 
                   np.gradient(gz, axis=0))
            skel = div < div_threshold
    
    else:
        raise ValueError(f'The input "{method}" is not a recognized method!')
    
    return skel
