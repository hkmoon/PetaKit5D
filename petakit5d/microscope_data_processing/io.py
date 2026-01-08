"""
I/O utilities for microscope data.

Ported from MATLAB microscopeDataProcessing/io/ directory.
"""

import numpy as np
from typing import Optional, Tuple
from pathlib import Path


def read_tiff(
    filepath: str,
    range_indices: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Read TIFF files with optional range specification.
    
    This is a wrapper that attempts to use optimized readers when available,
    falling back to standard libraries.
    
    Args:
        filepath: Path to the TIFF file
        range_indices: Optional tuple of (start, end) for reading specific slices.
                      Uses 1-based indexing to match MATLAB (converted internally)
        
    Returns:
        np.ndarray: Image data from TIFF file
        
    Examples:
        >>> data = read_tiff('image.tif')
        >>> data_range = read_tiff('stack.tif', range_indices=(1, 10))
        
    Original MATLAB function: readtiff.m
    Author: Xiongtao Ruan (09/28/2022)
    
    Note: This implementation uses tifffile library. For high-performance reading,
    consider using parallel readers or specialized libraries.
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError(
            "tifffile library is required for TIFF reading. "
            "Install it with: pip install tifffile"
        )
    
    filepath_obj = Path(filepath)
    if not filepath_obj.exists():
        raise FileNotFoundError(f"TIFF file not found: {filepath}")
    
    try:
        if range_indices is None:
            # Read entire file
            data = tifffile.imread(filepath)
        else:
            # Read specific range
            # Convert from MATLAB 1-based to Python 0-based indexing
            start_idx = range_indices[0] - 1
            end_idx = range_indices[1]  # End is exclusive in Python
            
            # Read with range
            with tifffile.TiffFile(filepath) as tif:
                if len(tif.pages) < end_idx:
                    raise ValueError(
                        f"Requested range {range_indices} exceeds available pages "
                        f"({len(tif.pages)})"
                    )
                
                # Read pages in range
                pages_to_read = list(range(start_idx, end_idx))
                data = tifffile.imread(filepath, key=pages_to_read)
        
        return data
        
    except Exception as e:
        raise IOError(f"Error reading TIFF file {filepath}: {str(e)}")
