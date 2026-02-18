"""
Image utility functions for metadata extraction and bounding box computation.

Author: Xiongtao Ruan
Python port: GitHub Copilot
"""

import numpy as np
from typing import Tuple, Union
import os


def get_image_bounding_box(image: np.ndarray) -> Tuple[int, ...]:
    """
    Get the bounding box of non-zero regions in a 2D or 3D image.
    
    Computes the minimal bounding box that contains all non-zero pixels/voxels
    in the image. Returns coordinates in MATLAB-style 1-based indexing.
    
    Parameters
    ----------
    image : ndarray
        2D or 3D image array
        
    Returns
    -------
    bbox : tuple of int
        For 2D: (y1, x1, y2, x2) - min and max row/column indices
        For 3D: (y1, x1, z1, y2, x2, z2) - min and max indices for each dimension
        All indices are 1-based (MATLAB convention)
        
    Examples
    --------
    >>> import numpy as np
    >>> img = np.zeros((10, 10))
    >>> img[3:7, 2:8] = 1
    >>> get_image_bounding_box(img)
    (4, 3, 7, 8)
    
    >>> img_3d = np.zeros((10, 10, 10))
    >>> img_3d[2:8, 3:7, 4:6] = 1
    >>> get_image_bounding_box(img_3d)
    (3, 4, 5, 8, 7, 6)
    
    Notes
    -----
    - Returns 1-based indices for MATLAB compatibility
    - For 2D images: bbox = [y1, x1, y2, x2]
    - For 3D images: bbox = [y1, x1, z1, y2, x2, z2]
    - Assumes image axes are (Y, X) or (Y, X, Z) following MATLAB convention
    """
    nd = image.ndim
    
    if nd not in [2, 3]:
        raise ValueError(f"Image must be 2D or 3D, got {nd}D")
    
    # Use boolean projection to robustly detect non-zero support
    if nd == 3:
        # For 3D: check for non-zero values along axes
        I_y = np.any(image != 0, axis=(1, 2))
        I_x = np.any(image != 0, axis=(0, 2))
        I_z = np.any(image != 0, axis=(0, 1))
    else:
        # For 2D: check for non-zero values along axes
        I_y = np.any(image != 0, axis=1)
        I_x = np.any(image != 0, axis=0)
    
    # Find first and last non-zero indices (0-based)
    y_nz = np.where(I_y)[0]
    x_nz = np.where(I_x)[0]
    
    if len(y_nz) == 0 or len(x_nz) == 0:
        # Empty image - return zeros
        if nd == 3:
            return (0, 0, 0, 0, 0, 0)
        return (0, 0, 0, 0)
    
    y1 = y_nz[0]
    y2 = y_nz[-1]
    x1 = x_nz[0]
    x2 = x_nz[-1]
    
    if nd == 3:
        z_nz = np.where(I_z)[0]
        if len(z_nz) == 0:
            return (0, 0, 0, 0, 0, 0)
        z1 = z_nz[0]
        z2 = z_nz[-1]
        # Convert to 1-based indexing
        return (y1 + 1, x1 + 1, z1 + 1, y2 + 1, x2 + 1, z2 + 1)
    else:
        # Convert to 1-based indexing
        return (y1 + 1, x1 + 1, y2 + 1, x2 + 1)


def get_image_data_type(file_path: str, zarr_file: bool = False) -> str:
    """
    Get image data type without loading the entire image.
    
    Reads metadata to determine the underlying data type of TIFF or Zarr files
    without loading the full image into memory.
    
    Parameters
    ----------
    file_path : str
        Path to image file (TIFF or Zarr)
    zarr_file : bool, optional
        If True, treat as Zarr file even if extension doesn't indicate it
        Default is False
        
    Returns
    -------
    dtype : str
        NumPy dtype string (e.g., 'uint8', 'uint16', 'float32')
        
    Raises
    ------
    ValueError
        If file format is not supported (only TIFF and Zarr supported)
    FileNotFoundError
        If file does not exist
        
    Examples
    --------
    >>> dtype = get_image_data_type('/path/to/image.tif')
    'uint16'
    
    >>> dtype = get_image_data_type('/path/to/data.zarr')
    'float32'
    
    Notes
    -----
    Supported formats:
    - TIFF (.tif, .tiff): Uses tifffile library
    - Zarr (.zarr): Uses zarr library
    
    This function is useful for determining appropriate memory allocation
    before loading large images.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Check if it's a TIFF file
    if file_path.lower().endswith(('.tif', '.tiff')):
        try:
            import tifffile
            with tifffile.TiffFile(file_path) as tif:
                # Get dtype from first page
                dtype = str(tif.pages[0].dtype)
                return dtype
        except ImportError:
            raise ImportError("tifffile library required for TIFF files. Install with: pip install tifffile")
    
    # Check if it's a Zarr file
    elif file_path.lower().endswith('.zarr') or zarr_file:
        try:
            import zarr
            z = zarr.open(file_path, mode='r')
            dtype = str(z.dtype)
            return dtype
        except ImportError:
            raise ImportError("zarr library required for Zarr files. Install with: pip install zarr")
    
    else:
        raise ValueError(
            f"Unknown file format: {file_path}\n"
            "Currently only TIFF (.tif, .tiff) and Zarr (.zarr) files are supported"
        )
