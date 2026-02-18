"""
Zarr file utilities for microscopy data processing.

This module provides utilities for creating and manipulating Zarr files,
including initialization, block writing, and format conversion.

Author: Xiongtao Ruan (original MATLAB), Python port
"""

import os
import shutil
import warnings
from typing import Optional, Tuple, Union
import numpy as np


def create_zarr(
    filepath: str,
    overwrite: bool = False,
    data_size: Tuple[int, ...] = (500, 500, 500),
    block_size: Tuple[int, ...] = (500, 500, 500),
    shard_size: Optional[Tuple[int, ...]] = None,
    dtype: str = 'uint16',
    order: str = 'F',
    expand_2d_dim: bool = True,
    group_write: bool = True,
    compressor: str = 'zstd',
    clevel: int = 1,
    zarr_sub_size: Optional[Tuple[int, ...]] = None,
    dim_separator: str = '.'
) -> None:
    """
    Initialize a Zarr file with specified parameters.
    
    Wrapper for initializing zarr file with chunking, compression, and sharding support.
    
    Parameters
    ----------
    filepath : str
        Path to the zarr file to create
    overwrite : bool, optional
        Whether to overwrite existing zarr file (default: False)
    data_size : tuple of int, optional
        Size of the data array (default: (500, 500, 500))
    block_size : tuple of int, optional
        Chunk size for zarr storage (default: (500, 500, 500))
    shard_size : tuple of int or None, optional
        Shard size within chunks (default: None)
    dtype : str, optional
        Data type ('uint16', 'uint8', 'single', 'double') (default: 'uint16')
    order : str, optional
        Storage order, 'F' for Fortran (column-major) or 'C' for C (row-major) (default: 'F')
    expand_2d_dim : bool, optional
        Expand z dimension for 2D data (default: True)
    group_write : bool, optional
        Set group write permissions on Unix (default: True)
    compressor : str, optional
        Compression algorithm ('zstd', 'gzip', 'lz4', etc.) (default: 'zstd')
    clevel : int, optional
        Compression level (default: 1)
    zarr_sub_size : tuple of int or None, optional
        Subfolder size for very large arrays (default: None)
    dim_separator : str, optional
        Dimension separator character ('.' or '/') (default: '.')
        
    Notes
    -----
    For 2D data, if expand_2d_dim is True, a singleton z dimension is added.
    Shard sizes must be divisors of block sizes.
    Uses zarr library for storage with optional compression.
    """
    # Convert to lists for modification
    data_size = list(data_size)
    block_size = list(block_size)
    
    # Handle 2D data
    if len(data_size) == 2:
        if expand_2d_dim:
            data_size.append(1)
            if len(block_size) == 2:
                block_size.append(1)
        else:
            block_size = block_size[:2]
    
    # Ensure block size doesn't exceed data size
    block_size = [min(d, b) for d, b in zip(data_size, block_size)]
    
    # Validate and adjust shard size
    if shard_size is not None:
        shard_size = list(shard_size)
        shard_size = [min(b, s) for b, s in zip(block_size, shard_size)]
        # Check if shard sizes are divisors of block sizes
        for i, (b, s) in enumerate(zip(block_size, shard_size)):
            if b % s != 0:
                warnings.warn(
                    f"Shard size must be divisor of block size at dimension {i}. "
                    f"Setting shard size to block size."
                )
                shard_size[i] = b
    
    # Overwrite if requested
    if overwrite and os.path.exists(filepath):
        shutil.rmtree(filepath)
    
    # Map dtype to zarr dtype
    dtype_map = {
        'double': np.float64,
        'single': np.float32,
        'uint16': np.uint16,
        'uint8': np.uint8,
    }
    
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported data type: {dtype}")
    
    np_dtype = dtype_map[dtype]
    
    try:
        import zarr
        from numcodecs import Zstd, GZip, LZ4, Blosc
        
        # Select compressor
        if compressor.lower() == 'zstd':
            comp = Zstd(level=clevel)
        elif compressor.lower() in ['gzip', 'gz']:
            comp = GZip(level=clevel)
        elif compressor.lower() == 'lz4':
            comp = LZ4(level=clevel)
        else:
            # Use Blosc with specified compressor
            comp = Blosc(cname=compressor, clevel=clevel)
        
        # Determine storage order
        if order == 'F':
            zarr_order = 'F'  # Fortran order (column-major)
        else:
            zarr_order = 'C'  # C order (row-major)
        
        # Create zarr array
        z = zarr.open(
            filepath,
            mode='w',
            shape=tuple(data_size),
            chunks=tuple(block_size),
            dtype=np_dtype,
            compressor=comp,
            order=zarr_order,
            dimension_separator=dim_separator
        )
        
        # Note: Zarr v2 doesn't have native sharding support
        # Sharding is handled differently in Zarr v3
        if shard_size is not None:
            warnings.warn(
                "Shard size parameter is noted but not directly supported in Zarr v2. "
                "Consider using smaller chunk sizes instead."
            )
        
    except ImportError as e:
        raise ImportError(
            "zarr library is required for create_zarr. "
            "Install it with: pip install zarr numcodecs"
        ) from e
    except Exception as e:
        warnings.warn(f"Error creating zarr file: {e}")
        raise
    
    # Set group write permissions on Unix
    if group_write and os.name != 'nt':  # Not Windows
        try:
            # Change permissions to allow group write
            os.chmod(filepath, 0o775)
            # Recursively set permissions for all files
            for root, dirs, files in os.walk(filepath):
                for d in dirs:
                    os.chmod(os.path.join(root, d), 0o775)
                for f in files:
                    os.chmod(os.path.join(root, f), 0o664)
        except Exception as e:
            warnings.warn(f"Unable to change file attributes for group write: {e}")


def write_zarr_block(
    zarr_array,
    block_sub: Tuple[int, ...],
    data: np.ndarray,
    mode: str = 'w'
) -> None:
    """
    Write a block of data to a Zarr array.
    
    Writes data to specified block coordinates with proper edge handling.
    
    Parameters
    ----------
    zarr_array : zarr.Array or similar
        Zarr array object to write to
    block_sub : tuple of int
        Block subscripts/coordinates (1-based indexing, converted to 0-based)
    data : ndarray
        Data to write to the block
    mode : str, optional
        Write mode: 'w' for direct write, 'r' for read mode with edge trimming
        (default: 'w')
        
    Notes
    -----
    In 'r' mode, data is trimmed if it extends beyond array boundaries.
    Block coordinates use 1-based indexing (MATLAB convention) and are converted
    to 0-based indexing internally.
    
    Examples
    --------
    >>> import zarr
    >>> z = zarr.open('data.zarr', mode='w', shape=(100, 100, 100), 
    ...               chunks=(50, 50, 50), dtype='uint16')
    >>> data = np.random.randint(0, 1000, (50, 50, 50), dtype='uint16')
    >>> write_zarr_block(z, (1, 1, 1), data, mode='w')
    """
    # Convert 1-based to 0-based indexing
    block_sub_0based = tuple(b - 1 for b in block_sub)
    
    if mode == 'w':
        # Direct block write using zarr's chunk-aware indexing
        chunk_shape = zarr_array.chunks
        ndim = zarr_array.ndim
        
        # Calculate starting indices
        start_idx = tuple(b * c for b, c in zip(block_sub_0based, chunk_shape))
        end_idx = tuple(s + c for s, c in zip(start_idx, data.shape[:ndim]))
        
        # Create slice objects
        slices = tuple(slice(s, e) for s, e in zip(start_idx, end_idx))
        
        # Write data
        zarr_array[slices] = data
    else:
        # Read mode with edge trimming
        chunk_shape = zarr_array.chunks
        array_shape = zarr_array.shape
        ndim = zarr_array.ndim
        
        # Calculate starting indices
        start_idx = tuple(b * c for b, c in zip(block_sub_0based, chunk_shape))
        
        # Check if this is an edge block
        is_edge = any(b == (s // c) for b, s, c in 
                     zip(block_sub_0based, array_shape, chunk_shape))
        
        if is_edge:
            # Calculate how much data extends beyond array boundaries
            total_data_end = tuple(s + d for s, d in zip(start_idx, data.shape[:ndim]))
            trim_amount = tuple(max(0, t - a) for t, a in zip(total_data_end, array_shape))
            
            if any(trim_amount):
                # Trim data to fit within array boundaries
                slices = tuple(slice(0, d - t) for d, t in zip(data.shape[:ndim], trim_amount))
                data = data[slices]
        
        # Calculate end indices after potential trimming
        end_idx = tuple(s + d for s, d in zip(start_idx, data.shape[:ndim]))
        
        # Create slice objects
        slices = tuple(slice(s, e) for s, e in zip(start_idx, end_idx))
        
        # Write trimmed data
        zarr_array[slices] = data


def integral_image_3d(
    A: np.ndarray,
    sz_t: Tuple[int, int, int]
) -> np.ndarray:
    """
    Compute 3D integral image (summed area table).
    
    Computes the integral image of a 3D array, which is useful for fast
    computation of sums over rectangular regions.
    
    Parameters
    ----------
    A : ndarray
        Input 3D array
    sz_t : tuple of int
        Size of the template (3 elements for 3D)
        
    Returns
    -------
    int_a : ndarray
        3D integral image array with shape (sz_a[0] + sz_t[0] - 1, 
        sz_a[1] + sz_t[1] - 1, sz_a[2] + sz_t[2] - 1) where sz_a is 
        the shape of A. This size allows computation of local sums for 
        normalized cross-correlation.
        
    Raises
    ------
    ValueError
        If A is smaller than sz_t in any dimension
        
    Notes
    -----
    The integral image allows computation of sums over any rectangular region
    in constant time. This is particularly useful for template matching and
    normalized cross-correlation.
    
    The algorithm uses cumulative sums along each dimension sequentially.
    This is adapted from MATLAB's normxcorr2 integral image computation.
    
    Examples
    --------
    >>> A = np.random.rand(100, 100, 100)
    >>> sz_t = (10, 10, 10)
    >>> int_img = integral_image_3d(A, sz_t)
    """
    sz_a = A.shape[:3]
    original_dtype = A.dtype
    
    # Validate input
    if any(s_a < s_t for s_a, s_t in zip(sz_a, sz_t)):
        raise ValueError(
            f"Size of A {sz_a} must not be smaller than size T {sz_t} in any dimension"
        )
    
    # Create padded array with zeros
    # Padding extends sz_t-1 on each side in each dimension
    B = np.zeros(tuple(s + 2*t - 1 for s, t in zip(sz_a, sz_t)), dtype=A.dtype)
    
    # Place A in the center of B
    slices = tuple(slice(t, t + s) for t, s in zip(sz_t, sz_a))
    B[slices] = A
    
    # Compute cumulative sum along first dimension
    s = np.cumsum(B, axis=0)
    # Subtract to get integral over window: s[i+sz_t[0]] - s[i]
    c = s[sz_t[0]:, :, :] - s[:-sz_t[0], :, :]
    
    # Compute cumulative sum along second dimension
    s = np.cumsum(c, axis=1)
    c = s[:, sz_t[1]:, :] - s[:, :-sz_t[1], :]
    
    # Compute cumulative sum along third dimension
    s = np.cumsum(c, axis=2)
    c = s[:, :, sz_t[2]:] - s[:, :, :-sz_t[2]]
    
    # Preserve original dtype (cumsum may promote integer types)
    if c.dtype != original_dtype:
        c = c.astype(original_dtype)
    
    return c
