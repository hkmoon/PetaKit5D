"""
Complete deskewing workflow for light sheet microscopy.

Ported from MATLAB deskewData.m and related functions.
"""

import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict, Any
import warnings

from .deskew_rotate import deskew_frame_3d, rotate_frame_3d
from .io import read_tiff, write_tiff
from .volume_utils import process_flatfield_correction_frame


def scmos_camera_flip(
    image: np.ndarray,
    flip_mode: str = 'none',
    axis: int = 1
) -> np.ndarray:
    """
    Flip image for sCMOS camera orientation correction.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (2D or 3D)
    flip_mode : str, optional
        Flip mode: 'none', 'horizontal', 'vertical', or 'both' (default: 'none')
    axis : int, optional
        Axis to flip for 3D volumes (default: 1 for Y axis)
        
    Returns
    -------
    np.ndarray
        Flipped image
    """
    if flip_mode == 'none':
        return image
    
    flipped = image.copy()
    
    if image.ndim == 2:
        if flip_mode in ['horizontal', 'both']:
            flipped = np.flip(flipped, axis=1)
        if flip_mode in ['vertical', 'both']:
            flipped = np.flip(flipped, axis=0)
    elif image.ndim == 3:
        # For 3D volumes, flip along specified axis
        if flip_mode in ['horizontal', 'both']:
            flipped = np.flip(flipped, axis=2)  # X axis
        if flip_mode in ['vertical', 'both']:
            flipped = np.flip(flipped, axis=axis)  # Specified axis (usually Y)
    
    return flipped


def deskew_data(
    input_paths: Union[str, List[str]],
    output_dir: str,
    dz: float = 0.5,
    angle: float = 32.45,
    pixel_size: float = 0.108,
    reverse: bool = False,
    rotate: bool = True,
    flip_mode: str = 'none',
    flat_field_path: Optional[str] = None,
    interpolation: str = 'linear',
    save_deskew: bool = True,
    save_rotate: bool = True,
    bbox: Optional[Tuple[int, ...]] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Complete deskewing workflow for light sheet microscopy data.
    
    This function orchestrates the complete deskewing pipeline:
    1. Load input images
    2. Apply camera flip (if specified)
    3. Apply flat field correction (if provided)
    4. Deskew volumes
    5. Rotate for visualization (optional)
    6. Save processed data
    
    Parameters
    ----------
    input_paths : str or list of str
        Path(s) to input TIFF files
    output_dir : str
        Directory for output files
    dz : float, optional
        Z step size in microns (default: 0.5)
    angle : float, optional
        Skew angle in degrees (default: 32.45)
    pixel_size : float, optional
        XY pixel size in microns (default: 0.108)
    reverse : bool, optional
        Whether scan direction is reversed (default: False)
    rotate : bool, optional
        Whether to rotate for visualization (default: True)
    flip_mode : str, optional
        Camera flip mode: 'none', 'horizontal', 'vertical', 'both' (default: 'none')
    flat_field_path : str, optional
        Path to flat field correction image (default: None)
    interpolation : str, optional
        Interpolation mode: 'linear' or 'cubic' (default: 'linear')
    save_deskew : bool, optional
        Whether to save deskewed volumes (default: True)
    save_rotate : bool, optional
        Whether to save rotated volumes (default: True)
    bbox : tuple, optional
        Bounding box for cropping (z1, z2, y1, y2, x1, x2) (default: None)
    overwrite : bool, optional
        Whether to overwrite existing files (default: False)
        
    Returns
    -------
    dict
        Dictionary with processing results:
        - 'n_files': number of files processed
        - 'output_dir': output directory path
        - 'deskewed_files': list of deskewed file paths
        - 'rotated_files': list of rotated file paths
        
    Examples
    --------
    >>> # Simple deskewing
    >>> result = deskew_data(
    ...     input_paths='raw_data.tif',
    ...     output_dir='processed/',
    ...     angle=32.45,
    ...     dz=0.5
    ... )
    
    >>> # With rotation and corrections
    >>> result = deskew_data(
    ...     input_paths=['time001.tif', 'time002.tif'],
    ...     output_dir='processed/',
    ...     angle=32.45,
    ...     rotate=True,
    ...     flip_mode='horizontal',
    ...     flat_field_path='flatfield.tif'
    ... )
    """
    # Convert to list
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load flat field if provided
    flat_field = None
    if flat_field_path is not None:
        flat_field = read_tiff(flat_field_path)
    
    # Process each file
    deskewed_files = []
    rotated_files = []
    
    for input_path in input_paths:
        input_file = Path(input_path)
        base_name = input_file.stem
        
        # Define output paths
        deskew_out = output_path / f"{base_name}_deskewed.tif"
        rotate_out = output_path / f"{base_name}_rotated.tif"
        
        # Check if files exist and skip if not overwriting
        if not overwrite:
            if save_deskew and deskew_out.exists():
                print(f"Skipping {deskew_out} (already exists)")
                deskewed_files.append(str(deskew_out))
                if save_rotate and rotate_out.exists():
                    rotated_files.append(str(rotate_out))
                continue
        
        # Load data
        print(f"Processing {input_file.name}...")
        data = read_tiff(str(input_path))
        
        # Apply camera flip
        if flip_mode != 'none':
            data = scmos_camera_flip(data, flip_mode)
        
        # Apply flat field correction
        if flat_field is not None:
            data = process_flatfield_correction_frame(
                data,
                flat_field,
                background_correction=True
            )
        
        # Apply bounding box crop if provided
        if bbox is not None:
            z1, z2, y1, y2, x1, x2 = bbox
            data = data[z1:z2, y1:y2, x1:x2]
        
        # Deskew
        deskewed = deskew_frame_3d(
            data,
            dz=dz,
            angle=angle,
            pixel_size=pixel_size,
            reverse=reverse,
            interpolation=interpolation
        )
        
        # Save deskewed
        if save_deskew:
            write_tiff(str(deskew_out), deskewed)
            deskewed_files.append(str(deskew_out))
            print(f"  Saved deskewed: {deskew_out.name}")
        
        # Rotate if requested
        if rotate:
            rotated = rotate_frame_3d(
                deskewed,
                angle=angle,
                dz=dz,
                pixel_size=pixel_size,
                reverse=reverse,
                crop=True
            )
            
            if save_rotate:
                write_tiff(str(rotate_out), rotated)
                rotated_files.append(str(rotate_out))
                print(f"  Saved rotated: {rotate_out.name}")
    
    return {
        'n_files': len(input_paths),
        'output_dir': str(output_path),
        'deskewed_files': deskewed_files,
        'rotated_files': rotated_files
    }
