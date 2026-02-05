"""
Core deskewing and rotation functions for light sheet microscopy.

Ported from MATLAB deskewFrame3D.m and rotateFrame3D.m
"""

import numpy as np
from scipy.ndimage import affine_transform
from typing import Tuple, Optional


def deskew_frame_3d(
    frame: np.ndarray,
    dz: float,
    angle: float,
    pixel_size: float = 0.108,
    reverse: bool = False,
    interpolation: str = 'linear',
    channel_axis: Optional[int] = None
) -> np.ndarray:
    """
    Deskew a 3D frame using shear transformation.

    Applies shear transformation to correct for oblique imaging plane in
    light sheet microscopy. Converts raw data to real-world coordinates.

    Parameters
    ----------
    frame : np.ndarray
        Input 3D array to deskew (Z, Y, X). If a 2D array is provided,
        it is treated as a single Z slice. For multi-channel data, provide
        a 4D array (C, Z, Y, X) or (Z, Y, X, C) and specify channel_axis.
    dz : float
        Z step size in microns
    angle : float
        Skew angle in degrees (typically 30-45°)
    pixel_size : float, optional
        XY pixel size in microns (default: 0.108)
    reverse : bool, optional
        Whether scan direction is reversed (default: False)
    interpolation : str, optional
        Interpolation mode: 'linear' or 'cubic' (default: 'linear')
    channel_axis : int, optional
        Axis along which channels are stored for 4D input (default: None).
        If None, 4D input is treated as before (squeeze singleton dimensions).
        If specified (typically 0 or -1), each channel is processed independently.

    Returns
    -------
    np.ndarray
        Deskewed 3D array (or 4D if multi-channel input with channel_axis specified)

    Notes
    -----
    The deskewing applies a shear transformation:
    dx = cos(angle) * dz / pixel_size  # pixels shifted per slice

    Output size is automatically calculated to fit the transformed volume.
    
    For multi-channel data, each channel is deskewed independently and the
    channel dimension is preserved in the output.
    """
    frame = np.asarray(frame)
    
    # Handle multi-channel data
    if frame.ndim == 4 and channel_axis is not None:
        # Process each channel independently
        n_channels = frame.shape[channel_axis]
        deskewed_channels = []
        
        for i in range(n_channels):
            # Extract single channel
            if channel_axis == 0:
                channel_data = frame[i]
            elif channel_axis == -1 or channel_axis == 3:
                channel_data = frame[..., i]
            else:
                raise ValueError(f"channel_axis must be 0 or -1 (or 3), got {channel_axis}")
            
            # Deskew single channel (recursive call without channel_axis)
            deskewed_channel = deskew_frame_3d(
                channel_data, dz, angle, pixel_size, reverse, interpolation, channel_axis=None
            )
            deskewed_channels.append(deskewed_channel)
        
        # Stack channels back together
        if channel_axis == 0:
            return np.stack(deskewed_channels, axis=0)
        else:  # channel_axis == -1 or 3
            return np.stack(deskewed_channels, axis=-1)
    
    # Original single-channel logic
    if frame.ndim == 2:
        frame = frame[None, ...]
    elif frame.ndim == 4:
        # Squeeze a singleton dimension (e.g., time or channel)
        if 1 in frame.shape:
            frame = np.squeeze(frame)
        if frame.ndim != 3:
            raise ValueError(
                f"deskew_frame_3d expects a 3D array (Z, Y, X); got shape {frame.shape}. "
                "If this is multi-channel data, specify channel_axis parameter."
            )
    elif frame.ndim != 3:
        raise ValueError(
            f"deskew_frame_3d expects a 3D array (Z, Y, X); got shape {frame.shape}."
        )

    nz, ny, nx = frame.shape
    angle_rad = np.deg2rad(angle)

    # Calculate pixel shift per Z slice
    dx = np.cos(angle_rad) * dz / pixel_size

    if reverse:
        dx = -dx

    # Calculate output size
    output_nx = int(np.ceil(nx + np.abs(dx) * (nz - 1)))
    output_shape = (nz, ny, output_nx)

    # Shear matrix: [1 0 0; 0 1 0; dx 0 1]
    # For scipy's affine_transform, we need the inverse transformation
    matrix = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [-dx, 0, 1]
    ])

    # Offset for centering
    offset = np.array([0, 0, np.abs(dx) * (nz - 1) if dx < 0 else 0])

    # Set interpolation order
    order = 1 if interpolation == 'linear' else 3

    # Apply transformation
    deskewed = affine_transform(
        frame,
        matrix,
        offset=offset,
        output_shape=output_shape,
        order=order,
        mode='constant',
        cval=0
    )

    return deskewed


def rotate_frame_3d(
    frame: np.ndarray,
    angle: float,
    dz: float,
    pixel_size: float = 0.108,
    reverse: bool = False,
    crop: bool = True,
    crop_xy: bool = True,
    channel_axis: Optional[int] = None
) -> np.ndarray:
    """
    Rotate a 3D frame for isotropic visualization.

    Applies rotation around Y axis with Z-anisotropy correction.
    Typically used after deskewing to create isotropic views.

    Parameters
    ----------
    frame : np.ndarray
        Input 3D array to rotate (Z, Y, X). If a 2D array is provided,
        it is treated as a single Z slice. For multi-channel data, provide
        a 4D array (C, Z, Y, X) or (Z, Y, X, C) and specify channel_axis.
    angle : float
        Rotation angle in degrees (typically same as skew angle)
    dz : float
        Z step size in microns
    pixel_size : float, optional
        XY pixel size in microns (default: 0.108)
    reverse : bool, optional
        Whether to reverse rotation direction (default: False)
    crop : bool, optional
        Whether to crop empty slices at Z boundaries (default: True)
    crop_xy : bool, optional
        Whether to crop empty regions in X,Y dimensions after rotation (default: True).
        This removes padding introduced by rotation to minimize output size.
    channel_axis : int, optional
        Axis along which channels are stored for 4D input (default: None).
        If None, 4D input is treated as before (squeeze singleton dimensions).
        If specified (typically 0 or -1), each channel is processed independently.

    Returns
    -------
    np.ndarray
        Rotated 3D array (or 4D if multi-channel input with channel_axis specified).
        If crop_xy=True, the output dimensions in Y and X will be reduced to fit
        the rotated content, removing empty padding.

    Notes
    -----
    The rotation combines:
    1. Z-anisotropy scaling (dz / pixel_size)
    2. Rotation around Y axis by specified angle
    3. Translation to center

    For sample scan: z_aniso = sin(angle) * dz / pixel_size
    For objective scan: z_aniso = dz / pixel_size
    
    For multi-channel data, each channel is rotated independently and the
    channel dimension is preserved in the output.
    
    When crop_xy=True (default), the output size is automatically adjusted to 
    the minimal bounding box containing non-zero data. This removes empty padding
    introduced by rotation, resulting in more compact output.
    
    Examples
    --------
    >>> # Without XY cropping (original behavior)
    >>> data = np.random.rand(10, 20, 30)
    >>> rotated = rotate_frame_3d(data, angle=32.45, dz=0.5, crop_xy=False)
    >>> rotated.shape  # Same Y,X as input
    (10, 20, 30)
    
    >>> # With XY cropping (new default)
    >>> rotated_cropped = rotate_frame_3d(data, angle=32.45, dz=0.5, crop_xy=True)
    >>> rotated_cropped.shape  # Y,X reduced to fit content
    (10, 18, 25)  # Example - actual values depend on rotation
    """
    frame = np.asarray(frame)
    
    # Handle multi-channel data
    if frame.ndim == 4 and channel_axis is not None:
        # Process each channel independently
        n_channels = frame.shape[channel_axis]
        rotated_channels = []
        
        for i in range(n_channels):
            # Extract single channel
            if channel_axis == 0:
                channel_data = frame[i]
            elif channel_axis == -1 or channel_axis == 3:
                channel_data = frame[..., i]
            else:
                raise ValueError(f"channel_axis must be 0 or -1 (or 3), got {channel_axis}")
            
            # Rotate single channel (recursive call without channel_axis)
            rotated_channel = rotate_frame_3d(
                channel_data, angle, dz, pixel_size, reverse, crop, crop_xy, channel_axis=None
            )
            rotated_channels.append(rotated_channel)
        
        # Stack channels back together
        if channel_axis == 0:
            return np.stack(rotated_channels, axis=0)
        else:  # channel_axis == -1 or 3
            return np.stack(rotated_channels, axis=-1)
    
    # Original single-channel logic
    if frame.ndim == 2:
        frame = frame[None, ...]
    elif frame.ndim == 4:
        if 1 in frame.shape:
            frame = np.squeeze(frame)
        if frame.ndim != 3:
            raise ValueError(
                f"rotate_frame_3d expects a 3D array (Z, Y, X); got shape {frame.shape}. "
                "If this is multi-channel data, specify channel_axis parameter."
            )
    elif frame.ndim != 3:
        raise ValueError(
            f"rotate_frame_3d expects a 3D array (Z, Y, X); got shape {frame.shape}."
        )

    nz, ny, nx = frame.shape
    angle_rad = np.deg2rad(angle)

    if reverse:
        angle_rad = -angle_rad

    # Calculate Z-anisotropy (sample scan geometry)
    z_aniso = np.sin(angle_rad) * dz / pixel_size
    zx_ratio = z_aniso if z_aniso > 0 else dz / pixel_size

    # Center coordinates
    cz, cy, cx = (nz - 1) / 2, (ny - 1) / 2, (nx - 1) / 2

    # Build transformation matrix
    # 1. Translate to origin
    T1 = np.array([
        [1, 0, 0, -cz],
        [0, 1, 0, -cy],
        [0, 0, 1, -cx],
        [0, 0, 0, 1]
    ])

    # 2. Scale Z
    S = np.array([
        [zx_ratio, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # 3. Rotate around Y axis
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    R = np.array([
        [cos_a, 0, sin_a, 0],
        [0, 1, 0, 0],
        [-sin_a, 0, cos_a, 0],
        [0, 0, 0, 1]
    ])

    # 4. Translate back
    T2 = np.array([
        [1, 0, 0, cz],
        [0, 1, 0, cy],
        [0, 0, 1, cx],
        [0, 0, 0, 1]
    ])

    # Combine: T2 @ R @ S @ T1
    M = T2 @ R @ S @ T1

    # Extract 3x3 matrix and offset for scipy
    matrix_3x3 = M[:3, :3]
    offset = M[:3, 3]

    # Apply transformation (use inverse for scipy)
    matrix_inv = np.linalg.inv(matrix_3x3)
    offset_inv = -matrix_inv @ offset

    rotated = affine_transform(
        frame,
        matrix_inv,
        offset=offset_inv,
        output_shape=frame.shape,
        order=1,  # Linear interpolation
        mode='constant',
        cval=0
    )

    if crop:
        # Crop empty slices at top and bottom (Z dimension)
        nonzero_slices = np.any(rotated > 0, axis=(1, 2))
        if np.any(nonzero_slices):
            first_slice = np.argmax(nonzero_slices)
            last_slice = len(nonzero_slices) - np.argmax(nonzero_slices[::-1]) - 1
            rotated = rotated[first_slice:last_slice+1]
    
    if crop_xy:
        # Crop empty regions in Y dimension
        nonzero_y = np.any(rotated > 0, axis=(0, 2))
        if np.any(nonzero_y):
            first_y = np.argmax(nonzero_y)
            last_y = len(nonzero_y) - np.argmax(nonzero_y[::-1]) - 1
            rotated = rotated[:, first_y:last_y+1, :]
        
        # Crop empty regions in X dimension
        nonzero_x = np.any(rotated > 0, axis=(0, 1))
        if np.any(nonzero_x):
            first_x = np.argmax(nonzero_x)
            last_x = len(nonzero_x) - np.argmax(nonzero_x[::-1]) - 1
            rotated = rotated[:, :, first_x:last_x+1]

    return rotated
