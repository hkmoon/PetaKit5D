"""
I/O utilities for microscope data.

Ported from MATLAB microscopeDataProcessing/io/ directory.
"""

import numpy as np
from typing import Optional, Tuple, Literal
from pathlib import Path
import os


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


def write_tiff(
    img: np.ndarray,
    filepath: str,
    compression: Literal['none', 'lzw'] = 'lzw',
    mode: Literal['tifffile', 'imageio'] = 'tifffile',
    metadata: Optional[dict] = None
) -> None:
    """
    Write image data to a TIFF file.

    This is a wrapper for writing TIFF files with compression support.
    Compatible with MATLAB's writetiff function.

    Parameters
    ----------
    img : np.ndarray
        Image data to write. Can be 2D, 3D, or 4D array.
    filepath : str
        Path to the output TIFF file.
    compression : {'none', 'lzw'}, optional
        Compression method. Default is 'lzw'.
        - 'none': No compression
        - 'lzw': LZW compression (lossless)
    mode : {'tifffile', 'imageio'}, optional
        Library to use for writing. Default is 'tifffile'.
        - 'tifffile': Use tifffile library (recommended)
        - 'imageio': Use imageio library (fallback)
    metadata : dict, optional
        Metadata dictionary to include in TIFF file. Common keys:
        - 'axes': String specifying axis order (e.g., 'CZYX', 'TZYX', 'ZYX')
        - 'Channel': Channel information dict or list
        - 'spacing': Physical spacing in microns
        - 'unit': Unit of measurement

    Raises
    ------
    ImportError
        If required library is not installed.
    IOError
        If file writing fails.

    Examples
    --------
    >>> # Write a 2D image
    >>> img = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
    >>> write_tiff(img, 'output.tif')

    >>> # Write a 3D stack without compression
    >>> stack = np.random.randint(0, 255, (50, 100, 100), dtype=np.uint8)
    >>> write_tiff(stack, 'stack.tif', compression='none')

    >>> # Write multichannel with axes metadata
    >>> multichannel = np.random.rand(4, 50, 100, 100)
    >>> write_tiff(multichannel, 'output.ome.tif', metadata={'axes': 'CZYX'})

    Notes
    -----
    The function automatically creates parent directories if they don't exist.

    Original MATLAB function: writetiff.m
    Author: Xiongtao Ruan (07/30/2020, updated 07/21/2022)

    For very large files or high-performance writing, consider using
    specialized libraries or parallel writing methods.
    """
    # Create parent directory if it doesn't exist
    filepath_obj = Path(filepath)
    parent_dir = filepath_obj.parent
    if parent_dir and not parent_dir.exists():
        parent_dir.mkdir(parents=True, exist_ok=True)

    # Map compression option
    if compression.lower() not in ['none', 'lzw']:
        raise ValueError(f"Invalid compression: {compression}. Must be 'none' or 'lzw'")

    try:
        if mode == 'tifffile':
            try:
                import tifffile
            except ImportError:
                raise ImportError(
                    "tifffile library is required for TIFF writing. "
                    "Install it with: pip install tifffile"
                )

            # Check if imagecodecs is available for compression
            try:
                import imagecodecs
                imagecodecs_available = True
            except ImportError:
                imagecodecs_available = False

            # Set compression parameter
            if compression.lower() == 'lzw':
                if imagecodecs_available:
                    compress = 'lzw'
                else:
                    # Fall back to no compression if imagecodecs not available
                    compress = None
                    import warnings
                    warnings.warn(
                        "imagecodecs not available, writing without compression. "
                        "Install imagecodecs for LZW compression support: pip install imagecodecs",
                        UserWarning
                    )
            else:
                compress = None

            # Prepare write parameters
            write_params = {
                'compression': compress,
                'photometric': 'minisblack' if img.ndim >= 2 else None
            }

            # Add metadata if provided
            if metadata is not None:
                # Check if axes metadata is provided
                if 'axes' in metadata:
                    write_params['metadata'] = {'axes': metadata['axes']}
                    # Use OME-TIFF format for proper metadata support
                    if filepath.lower().endswith('.ome.tif') or filepath.lower().endswith('.ome.tiff'):
                        # For OME-TIFF, use more compatible parameters
                        write_params['imagej'] = False
                    else:
                        write_params['imagej'] = True
                else:
                    write_params['imagej'] = True

                # Add other metadata fields if present
                if 'spacing' in metadata or 'unit' in metadata or 'Channel' in metadata:
                    if 'metadata' not in write_params:
                        write_params['metadata'] = {}
                    if 'spacing' in metadata:
                        write_params['metadata']['spacing'] = metadata['spacing']
                    if 'unit' in metadata:
                        write_params['metadata']['unit'] = metadata['unit']
                    if 'Channel' in metadata:
                        write_params['metadata']['Channel'] = metadata['Channel']
            else:
                # ImageJ format doesn't support float64, convert to float32
                if img.dtype == np.float64:
                    write_params['imagej'] = False
                else:
                    write_params['imagej'] = True

            # Write the TIFF file
            tifffile.imwrite(filepath, img, **write_params)

        elif mode == 'imageio':
            try:
                import imageio
            except ImportError:
                raise ImportError(
                    "imageio library is required for TIFF writing. "
                    "Install it with: pip install imageio"
                )

            # imageio uses different compression parameter names
            if compression.lower() == 'lzw':
                compress = 5  # LZW compression code
            else:
                compress = 0  # No compression

            # Write the TIFF file
            imageio.imwrite(filepath, img, format='TIFF', compression=compress)

        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'tifffile' or 'imageio'")

        # Set file permissions on Unix-like systems (simulate groupWrite from MATLAB)
        if not os.name == 'nt':  # Not Windows
            try:
                os.chmod(filepath, 0o664)  # rw-rw-r--
            except (OSError, PermissionError):
                pass  # Ignore permission errors

    except Exception as e:
        raise IOError(f"Error writing TIFF file {filepath}: {str(e)}")
