"""
Additional stitching support utilities for microscopy data processing.

This module provides helper functions for stitching workflows including 
z-stack normalization, distance weighting, and file path processing.
"""

import numpy as np
import os
from typing import Tuple, List, Optional, Literal


def normalize_z_stack(img_in: np.ndarray) -> np.ndarray:
    """
    Normalize intensity across z slices in 3D volume data.
    
    Performs intensity normalization by computing the median intensity
    for each z slice and normalizing based on the median of medians.
    
    Parameters
    ----------
    img_in : np.ndarray
        Input 3D image array (y, x, z).
    
    Returns
    -------
    imn : np.ndarray
        Normalized 3D image with same dtype as input.
    
    Examples
    --------
    >>> img = np.random.rand(100, 100, 50) * 1000
    >>> normalized = normalize_z_stack(img)
    
    Notes
    -----
    - Background threshold is set to 105
    - Median values are clamped within ±10 of global median
    - Original MATLAB function by Xiongtao Ruan (04/09/2020)
    """
    bg = 105
    T = bg
    im = img_in.copy()
    
    # Subtract threshold
    imT = im - T
    
    # Compute median for each z slice
    nz = im.shape[2]
    Med = np.zeros(nz)
    for j in range(nz):
        ims = imT[:, :, j]
        positive_vals = ims[ims > 0]
        if len(positive_vals) > 0:
            Med[j] = np.median(positive_vals)
        else:
            Med[j] = np.nan
    
    # Get global median and clamp values
    m = np.nanmedian(Med)
    spd = 10
    n = Med.copy()
    n[np.isnan(n)] = m
    n[n < m - spd] = m - spd
    n[n > m + spd] = m + spd
    
    # Normalize
    nn = n / np.max(n)
    
    # Apply normalization to each slice
    imn = np.zeros_like(im)
    for j in range(nz):
        if nn[j] != 0:
            imn[:, :, j] = imT[:, :, j] / nn[j]
        else:
            imn[:, :, j] = imT[:, :, j]
    
    return imn


def distance_weight_single_axis(
    sz: int,
    end_points: np.ndarray,
    buffer_size: int = 10,
    dfactor: float = 0.99,
    win_type: Literal['hann'] = 'hann'
) -> np.ndarray:
    """
    Calculate distance weight for a single axis for feather blending.
    
    Computes distance-based weights using a window function at the boundaries
    with exponential decay outside the buffer region.
    
    Parameters
    ----------
    sz : int
        Size of the axis.
    end_points : np.ndarray
        Start and end points [start, end] (1-based indexing).
    buffer_size : int, default=10
        Size of the buffer/transition region.
    dfactor : float, default=0.99
        Decay factor for exponential falloff outside buffer.
        Set to 0 to disable decay.
    win_type : {'hann'}, default='hann'
        Type of window function to use.
    
    Returns
    -------
    dist_weight : np.ndarray
        Distance weights as float32 array of shape (sz,).
    
    Examples
    --------
    >>> weights = distance_weight_single_axis(100, np.array([10, 90]), buffer_size=5)
    >>> weights.shape
    (100,)
    
    Notes
    -----
    - Uses Hann window: cos^2(0.5 * pi * x / buffer_size)
    - Applies exponential decay with dfactor outside buffer regions
    - Original MATLAB function has no specified author
    - Indices are converted from 1-based (MATLAB) to 0-based (Python)
    """
    dist_weight = np.ones(sz, dtype=np.float32)
    
    # Convert from 1-based to 0-based indexing
    s = max(0, end_points[0] - 1)
    t = min(sz - 1, end_points[1] - 1)
    
    # If covers entire range, return all ones
    if s == 0 and t == sz - 1:
        return dist_weight
    
    # Zero out regions outside [s, t]
    dist_weight[:s+1] = 0
    dist_weight[t:] = 0
    
    # Create window weights
    win_dist = np.arange(buffer_size + 1)
    
    if win_type.lower() == 'hann':
        win_func = lambda x, y: np.cos(0.5 * np.pi * (y - x) / y) ** 2
    else:
        raise ValueError(f"Unsupported window type: {win_type}")
    
    win_weight = win_func(win_dist, buffer_size)
    win_weight = np.maximum(win_weight, 1e-3)
    
    # Apply window at start
    s0 = max(0, s - buffer_size)
    start_idx = buffer_size - (s - s0)
    dist_weight[s0:s+1] = win_weight[start_idx:]
    
    # Lower/upper bounds for decay regions
    lb = 1e-4 if dfactor > 0 else 0
    ub = 1e-3 if dfactor > 0 else 1
    
    # Exponential decay before start buffer
    if s0 > 0:
        decay_vals = dfactor ** np.arange(s0 - 1, -1, -1)
        dist_weight[:s0] = decay_vals * np.clip(dist_weight[s0], lb, ub)
    
    # Apply window at end
    win_weight_f = np.flip(win_weight)
    t1 = min(sz - 1, t + buffer_size)
    dist_weight[t:t1+1] = win_weight_f[:t1 - t + 1]
    
    # Exponential decay after end buffer
    if t1 < sz - 1:
        decay_vals = dfactor ** np.arange(1, sz - t1)
        dist_weight[t1+1:] = decay_vals * np.clip(dist_weight[t1], lb, ub)
    
    # Apply lower bound if using decay
    if dfactor > 0:
        dist_weight = np.maximum(dist_weight, 1e-30)
    
    return dist_weight.astype(np.float32)


def stitch_process_filenames(
    tile_fullpaths: List[str],
    processed_dirstr: str = '',
    stitch_mip: np.ndarray = np.array([False, False, False]),
    resample: Optional[np.ndarray] = None,
    zarr_file: bool = False,
    process_tiles: bool = True
) -> Tuple[List[str], List[str], List[str], str]:
    """
    Process file paths for stitching pipeline.
    
    Generates input and output file paths based on processing requirements
    (e.g., deskew/rotate, deconvolution, MIP generation, resampling).
    
    Parameters
    ----------
    tile_fullpaths : List[str]
        List of full paths to tile files.
    processed_dirstr : str, default=''
        Subdirectory name for processed data (e.g., 'DSR', 'DS_Decon').
    stitch_mip : np.ndarray, default=[False, False, False]
        Boolean array indicating MIP generation along [y, x, z] axes.
    resample : np.ndarray, optional
        Resampling factors. If provided, will be included in zarr path.
    zarr_file : bool, default=False
        If True, use .zarr extension; otherwise use .tif.
    process_tiles : bool, default=True
        If True, tiles are being processed (affects path generation).
    
    Returns
    -------
    input_fullpaths : List[str]
        List of input file paths.
    zarr_fullpaths : List[str]
        List of zarr output paths.
    fsnames : List[str]
        List of file stem names (without extension).
    zarr_pathstr : str
        Zarr directory name.
    
    Examples
    --------
    >>> tiles = ['/data/tile_001.tif', '/data/tile_002.tif']
    >>> inputs, zarrs, names, zpath = stitch_process_filenames(
    ...     tiles, processed_dirstr='DSR'
    ... )
    
    Notes
    -----
    - Handles MIP stitching with _MIP_z suffix
    - Supports resampling with zarr_{y}_{x}_{z} directory naming
    - Original MATLAB function by Xiongtao Ruan (07/23/2021)
    """
    nF = len(tile_fullpaths)
    
    input_fullpaths = []
    zarr_fullpaths = []
    fsnames = []
    suffix_str = 'z'
    ext = '.zarr' if zarr_file else '.tif'
    
    for i in range(nF):
        tile_fullpath = tile_fullpaths[i]
        data_path, full_fname = os.path.split(tile_fullpath)
        fsname = os.path.splitext(full_fname)[0]
        data_path = data_path.rstrip('/')
        
        if processed_dirstr:
            tile_path = f"{data_path}/{processed_dirstr}"
        else:
            tile_path = data_path
        
        fsnames.append(fsname)
        
        if process_tiles:
            if resample is not None and np.any(resample != 1):
                # Complete resample to 3D
                rs = resample if len(resample) >= 3 else np.concatenate([
                    np.ones(4 - len(resample)) * resample[0],
                    resample[1:]
                ])[:3]
                rs = rs.astype(int)
                zarr_pathstr = f"zarr_{rs[0]}_{rs[1]}_{rs[2]}"
            else:
                zarr_pathstr = 'zarr'
                if zarr_file:
                    zarr_pathstr = 'zarr_processed'
        else:
            zarr_pathstr = ''
        
        if np.any(stitch_mip):
            input_fullpaths.append(
                f"{tile_path}/MIPs/{fsname}_MIP_{suffix_str}{ext}"
            )
            zarr_fullpaths.append(
                f"{tile_path}/MIPs/{zarr_pathstr}/{fsname}_MIP_{suffix_str}.zarr"
            )
        else:
            input_fullpaths.append(f"{tile_path}/{fsname}{ext}")
            zarr_fullpaths.append(f"{tile_path}/{zarr_pathstr}/{fsname}.zarr")
    
    return input_fullpaths, zarr_fullpaths, fsnames, zarr_pathstr
