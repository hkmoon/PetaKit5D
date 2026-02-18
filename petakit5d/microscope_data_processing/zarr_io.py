"""
Zarr file I/O utilities.

This module provides functions for reading Zarr format files.
Note: This is a simplified Python version. The full MATLAB implementation
includes parallel reading and various adapters that are Python-ecosystem specific.
"""

import numpy as np
from typing import Optional, Tuple
import warnings


def read_zarr(
    filepath: str,
    input_bbox: Optional[Tuple[int, ...]] = None,
    sparse_data: bool = True
) -> np.ndarray:
    """
    Read data from a Zarr file.
    
    Wrapper for zarr reader with support for bounding box selection.
    This is a simplified version of the MATLAB readzarr function,
    using the zarr-python library.
    
    Parameters
    ----------
    filepath : str
        Path to the Zarr file/directory
    input_bbox : tuple of ints, optional
        Region bounding box to read. For 3D data, provide as:
        (ymin, xmin, zmin, ymax, xmax, zmax) using 1-based MATLAB indexing.
        If None, reads entire array.
    sparse_data : bool, optional
        Whether to handle sparse data efficiently (default: True).
        Note: In Python zarr, sparsity is handled automatically.
    
    Returns
    -------
    data : np.ndarray
        The data read from the Zarr file
    
    Examples
    --------
    >>> # Read entire zarr array
    >>> data = read_zarr('/path/to/data.zarr')
    >>> 
    >>> # Read a bounding box (1-based MATLAB indexing)
    >>> bbox = (10, 20, 5, 50, 60, 25)  # ymin, xmin, zmin, ymax, xmax, zmax
    >>> data = read_zarr('/path/to/data.zarr', input_bbox=bbox)
    
    Notes
    -----
    - Based on MATLAB readzarr by Xiongtao Ruan (01/25/2022)
    - Requires zarr-python package: pip install zarr
    - Python zarr handles chunking and compression automatically
    - For distributed/parallel reading, consider dask with zarr
    
    References
    ----------
    Xiongtao Ruan (01/25/2022, 02/01/2022)
    """
    try:
        import zarr
    except ImportError:
        raise ImportError(
            "zarr package is required for read_zarr. "
            "Install it with: pip install zarr"
        )
    
    try:
        # Open zarr array
        z = zarr.open(filepath, mode='r')
        
        if input_bbox is None:
            # Read entire array
            data = z[:]
        else:
            # Convert MATLAB 1-based indexing to Python 0-based
            if len(input_bbox) == 6:
                # 3D bounding box: (ymin, xmin, zmin, ymax, xmax, zmax)
                ymin, xmin, zmin, ymax, xmax, zmax = input_bbox
                # Convert to 0-based and create slices
                data = z[ymin-1:ymax, xmin-1:xmax, zmin-1:zmax]
            elif len(input_bbox) == 4:
                # 2D bounding box: (ymin, xmin, ymax, xmax)
                ymin, xmin, ymax, xmax = input_bbox
                data = z[ymin-1:ymax, xmin-1:xmax]
            else:
                raise ValueError(
                    f"input_bbox must have 4 (2D) or 6 (3D) elements, got {len(input_bbox)}"
                )
        
        return np.asarray(data)
        
    except Exception as e:
        import os
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Zarr file {filepath} does not exist") from e
        else:
            raise RuntimeError(f"Error reading Zarr file {filepath}: {str(e)}") from e


def write_zarr(
    filepath: str,
    data: np.ndarray,
    chunks: Optional[Tuple[int, ...]] = None,
    compressor: str = 'default',
    overwrite: bool = False
) -> None:
    """
    Write data to a Zarr file.
    
    Parameters
    ----------
    filepath : str
        Path where Zarr file/directory will be created
    data : np.ndarray
        Data to write
    chunks : tuple of ints, optional
        Chunk size for each dimension. If None, uses automatic chunking.
    compressor : str, optional
        Compression algorithm. Options: 'default', 'blosc', 'zstd', 'gzip', 'bz2', 'none'
        Default is 'default' which uses Blosc with lz4.
    overwrite : bool, optional
        Whether to overwrite existing file (default: False)
    
    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(100, 100, 50)
    >>> write_zarr('/path/to/output.zarr', data, chunks=(50, 50, 25))
    
    Notes
    -----
    Helper function for creating Zarr files with common defaults
    """
    try:
        import zarr
        from numcodecs import Blosc, Zstd, GZip, BZ2
    except ImportError:
        raise ImportError(
            "zarr and numcodecs packages are required. "
            "Install with: pip install zarr numcodecs"
        )
    
    # Select compressor
    if compressor == 'default' or compressor == 'blosc':
        comp = Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
    elif compressor == 'zstd':
        comp = Zstd(level=3)
    elif compressor == 'gzip':
        comp = GZip(level=6)
    elif compressor == 'bz2':
        comp = BZ2(level=5)
    elif compressor == 'none':
        comp = None
    else:
        warnings.warn(f"Unknown compressor '{compressor}', using default")
        comp = Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
    
    # Create zarr array with proper overwrite handling
    # Force zarr_format=2 for compatibility with compressors
    if overwrite:
        z = zarr.open(filepath, mode='w', shape=data.shape, dtype=data.dtype, 
                     chunks=chunks, compressor=comp, zarr_format=2)
        z[:] = data
    else:
        z = zarr.open(filepath, mode='w-', shape=data.shape, dtype=data.dtype,
                     chunks=chunks, compressor=comp, zarr_format=2)
        z[:] = data
