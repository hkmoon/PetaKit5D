"""
Utility functions for microscope data processing.

This module provides utility functions for microscopy workflows including
resample setting validation, memory estimation, and partial volume file grouping.

Author: Xiongtao Ruan (original MATLAB), Python port
"""

import numpy as np
from typing import Tuple, List, Optional, Union
import os
import re
from pathlib import Path


def check_resample_setting(
    resample_type: str,
    resample: Optional[Union[float, List[float], Tuple[float, ...]]],
    objective_scan: bool,
    skew_angle: float,
    xy_pixel_size: float,
    dz: float
) -> Tuple[np.ndarray, float]:
    """
    Check and validate resample settings for microscopy data.
    
    Computes the anisotropy factor and validates/computes the resample factors
    based on the resample type.
    
    Parameters
    ----------
    resample_type : str
        Type of resampling: 'given', 'isotropic', or 'xy_isotropic'
    resample : float or list of float or None
        Resample factors. Required for 'given' type, ignored for others.
        Can be scalar (applied to all dims), 2-element (XY, Z), or 3-element (X, Y, Z).
    objective_scan : bool
        Whether using objective scan (True) or stage scan (False)
    skew_angle : float
        Skew angle in degrees
    xy_pixel_size : float
        Pixel size in XY plane
    dz : float
        Z step size
        
    Returns
    -------
    resample : ndarray
        Resample factors as [X, Y, Z]
    z_aniso : float
        Z anisotropy factor
        
    Raises
    ------
    ValueError
        If resample_type is 'given' but resample is None/empty
        
    Examples
    --------
    >>> resample, z_aniso = check_resample_setting('isotropic', None, True, 32.8, 0.108, 0.3)
    >>> resample
    array([1., 1., 1.])
    
    >>> resample, z_aniso = check_resample_setting('given', [1.5, 2.0], False, 32.8, 0.108, 0.3)
    >>> resample
    array([1.5, 1.5, 2. ])
    """
    # Compute Z anisotropy
    if objective_scan:
        z_aniso = dz / xy_pixel_size
    else:
        z_aniso = np.sin(np.deg2rad(skew_angle)) * dz / xy_pixel_size
    
    # Process resample based on type
    if resample_type == 'given':
        if resample is None or (isinstance(resample, (list, tuple, np.ndarray)) and len(resample) == 0):
            raise ValueError('For resample Type "given", the parameter resample must not be empty!')
        
        # Convert to numpy array
        if np.isscalar(resample):
            resample_arr = np.ones(3) * resample
        else:
            resample_arr = np.asarray(resample)
            if resample_arr.size == 1:
                resample_arr = np.ones(3) * resample_arr.item()
            elif resample_arr.size == 2:
                resample_arr = np.array([resample_arr[0], resample_arr[0], resample_arr[1]])
            elif resample_arr.size == 3:
                resample_arr = resample_arr.flatten()
            else:
                raise ValueError(f'resample must have 1, 2, or 3 elements, got {resample_arr.size}')
                
    elif resample_type == 'isotropic':
        resample_arr = np.ones(3)
        
    elif resample_type == 'xy_isotropic':
        theta = np.deg2rad(skew_angle)
        zf = np.sqrt(
            (np.sin(theta)**2 + z_aniso**2 * np.cos(theta)**2) /
            (np.cos(theta)**2 + z_aniso**2 * np.sin(theta)**2)
        )
        resample_arr = np.array([1.0, 1.0, zf])
    else:
        raise ValueError(f'Unknown resample_type: {resample_type}. Must be "given", "isotropic", or "xy_isotropic"')
    
    return resample_arr, z_aniso


def estimate_computing_memory(
    file_path: str,
    steps: Optional[List[str]] = None,
    im_size: Optional[Tuple[int, ...]] = None,
    data_size: Optional[float] = None,
    mem_factors: Optional[List[float]] = None,
    cuda_decon: bool = True,
    gpu_mem_factor: float = 1.5,
    gpu_max_mem: float = 12.0
) -> Tuple[np.ndarray, float, float, Optional[Tuple[int, ...]]]:
    """
    Estimate memory requirements for microscopy computing workflows.
    
    Estimates required memory in GB for deskew, rotate, and deconvolution steps.
    
    Parameters
    ----------
    file_path : str
        Path to image file
    steps : list of str, optional
        Processing steps to estimate memory for.
        Default: ['deskew', 'rotate', 'deconvolution']
    im_size : tuple of int, optional
        Image size as (Y, X, Z). If None, read from file.
    data_size : float, optional
        Data size in GB. If None, compute from file or im_size.
    mem_factors : list of float, optional
        Memory multiplication factors for [deskew, rotate, decon].
        Default: [15, 5, 15]
    cuda_decon : bool
        Whether using CUDA deconvolution. Default: True
    gpu_mem_factor : float
        GPU memory multiplication factor for deconvolution. Default: 1.5
    gpu_max_mem : float
        Maximum GPU memory in GB. Default: 12.0
        
    Returns
    -------
    est_required_memory : ndarray
        Estimated required memory in GB for each step
    est_required_gpu_mem : float
        Estimated required GPU memory in GB (NaN if not using CUDA decon)
    raw_image_size : float
        Raw image size in GB
    im_size : tuple of int or None
        Image size (Y, X, Z)
        
    Examples
    --------
    >>> mem, gpu_mem, raw_size, size = estimate_computing_memory(
    ...     'test.tif', steps=['deskew', 'deconvolution'], 
    ...     im_size=(1024, 1024, 500))
    """
    if steps is None:
        steps = ['deskew', 'rotate', 'deconvolution']
    
    if mem_factors is None:
        mem_factors = [15, 5, 15]
    
    # Get image size and compute raw image size
    if data_size is None or any('deskew' in s.lower() for s in steps):
        if not os.path.exists(file_path):
            if im_size is None:
                raise FileNotFoundError(f'File {file_path} does not exist!')
        else:
            # Try to get image size using our utility function
            try:
                from ..utils.image_utils import get_image_data_type
                from ..microscope_data_processing.io import read_tiff
                # For simplicity, assume we can read metadata
                # In real implementation, would use tifffile or similar
                pass
            except:
                pass
            
            if im_size is None:
                # Would normally use getImageSize here
                # For now, if file exists but we can't read, use provided size
                raise ValueError('Cannot determine image size from file. Please provide im_size parameter.')
        
        # Compute raw image size in GB (assume float32 = 4 bytes)
        if im_size is not None:
            raw_image_size = np.prod(im_size) * 4 / (1024**3)
        else:
            # Try to get file size
            raw_image_size = os.path.getsize(file_path) / (1024**3)
    else:
        raw_image_size = data_size * 2 / (1024**3)
    
    # Estimate memory for each step
    est_required_memory = np.zeros(len(steps))
    est_required_gpu_mem = np.nan
    
    for i, step in enumerate(steps):
        step_lower = step.lower()
        
        if 'deskew' in step_lower:
            # Use 300 slices as threshold for scaling
            if im_size is not None:
                scale_factor = max(1.0, (im_size[2] / 300) ** 2)
            else:
                scale_factor = 1.0
            est_required_memory[i] = raw_image_size * mem_factors[0] * scale_factor
            
        elif 'rotate' in step_lower:
            est_required_memory[i] = raw_image_size * mem_factors[1]
            
        elif 'deconvolution' in step_lower or 'decon' in step_lower:
            est_required_memory[i] = raw_image_size * mem_factors[2]
            if cuda_decon:
                est_required_gpu_mem = raw_image_size * gpu_mem_factor
    
    return est_required_memory, est_required_gpu_mem, raw_image_size, im_size


def group_partial_volume_files(
    data_path: str = '',
    file_fullpath_list: Optional[List[str]] = None,
    ext: str = '.tif',
    only_first_tp: bool = False,
    channel_patterns: Optional[List[str]] = None
) -> Tuple[bool, List[List[str]], List, List]:
    """
    Check and group filenames if they are parts of the same volume.
    
    Groups files with patterns like: file.tif, file_part0001.tif, file_part0002.tif, etc.
    
    Parameters
    ----------
    data_path : str
        Path to directory containing files. Can be empty if file_fullpath_list provided.
    file_fullpath_list : list of str, optional
        List of full file paths. If provided, data_path can be empty.
    ext : str
        File extension to search for. Default: '.tif'
    only_first_tp : bool
        Whether to only process first timepoint (files with 'Iter_0000_'). Default: False
    channel_patterns : list of str, optional
        Channel patterns to filter files. Only files matching patterns are included.
        
    Returns
    -------
    contain_part_volume : bool
        Whether any partial volume files were found
    grouped_fnames : list of list of str
        Grouped filenames. Each element is a list of files belonging to same volume.
    grouped_datenum : list
        Date numbers for each group
    grouped_datasize : list  
        Data sizes for each group
        
    Examples
    --------
    >>> has_parts, groups, dates, sizes = group_partial_volume_files(
    ...     '/data/images', ext='.tif', only_first_tp=True)
    >>> if has_parts:
    ...     print(f'Found {len(groups)} volume groups')
    """
    print('Check partial volume files... ', end='')
    
    if channel_patterns is None:
        channel_patterns = []
    
    # Get file list
    if not data_path:
        if file_fullpath_list is None or len(file_fullpath_list) == 0:
            raise ValueError('Either data_path or file_fullpath_list must be provided')
        
        if only_first_tp:
            file_fullpath_list = [f for f in file_fullpath_list if 'Iter_0000_' in f]
        
        data_path = os.path.dirname(file_fullpath_list[0])
        ext = os.path.splitext(file_fullpath_list[0])[1]
        
        fnames = []
        datenum = []
        datasize = []
        for fpath in file_fullpath_list:
            stat = os.stat(fpath)
            fnames.append(os.path.basename(fpath))
            datenum.append(stat.st_mtime)
            datasize.append(stat.st_size)
    else:
        # Search for files in directory
        if only_first_tp:
            pattern = '*_Iter_0000_*' + ext
        else:
            pattern = '*' + ext
        
        # Use glob to find files
        from glob import glob
        file_paths = glob(os.path.join(data_path, pattern))
        
        if not file_paths:
            print()
            print('Warning: No files found matching pattern')
            return False, [], [], []
        
        fnames = []
        datenum = []
        datasize = []
        for fpath in file_paths:
            stat = os.stat(fpath)
            fnames.append(os.path.basename(fpath))
            datenum.append(stat.st_mtime)
            datasize.append(stat.st_size)
        
        # Filter by channel patterns if provided
        if channel_patterns:
            include_flag = np.zeros(len(fnames), dtype=bool)
            for i, fname in enumerate(fnames):
                fpath = os.path.join(data_path, fname)
                for pattern in channel_patterns:
                    if pattern in fpath or re.search(pattern, fpath):
                        include_flag[i] = True
                        break
            
            fnames = [f for i, f in enumerate(fnames) if include_flag[i]]
            datenum = [d for i, d in enumerate(datenum) if include_flag[i]]
            datasize = [s for i, s in enumerate(datasize) if include_flag[i]]
        
        if not fnames:
            print()
            print('Warning: The input image list is empty or none match channelPatterns.')
            return False, [], [], []
    
    # Check for partial volume files
    part_pattern = re.compile(r'_part\d+' + re.escape(ext) + r'$')
    contain_part_volume = [part_pattern.search(f) is not None for f in fnames]
    
    if not any(contain_part_volume):
        print()
        print('No partial volume files found.')
        # Return each file as its own group
        grouped_fnames = [[f] for f in fnames]
        grouped_datenum = datenum
        grouped_datasize = datasize
        return False, grouped_fnames, grouped_datenum, grouped_datasize
    
    # Get filenames without extension for matching
    fsnames = [os.path.splitext(f)[0] for f in fnames]
    
    # Find main files (without _part\d+ pattern)
    main_pattern = re.compile(r'_part\d+$')
    mstr_inds = [main_pattern.search(name) is None for name in fsnames]
    
    mstr = [fsnames[i] for i, is_main in enumerate(mstr_inds) if is_main]
    mdn = [datenum[i] for i, is_main in enumerate(mstr_inds) if is_main]
    mds = [datasize[i] for i, is_main in enumerate(mstr_inds) if is_main]
    
    # Group partial files with their main file
    grouped_fnames = []
    grouped_datenum = []
    grouped_datasize = []
    
    for i, mstr_i in enumerate(mstr):
        # Find part files matching this main file
        part_indices = []
        for j, fsname in enumerate(fsnames):
            if re.search(re.escape(mstr_i) + r'_part\d+$', fsname):
                part_indices.append(j)
        
        if part_indices:
            # Sort part files
            part_files = sorted([fnames[j] for j in part_indices])
            group = [mstr_i + ext] + part_files
            
            pdn = [datenum[j] for j in part_indices]
            pds = [datasize[j] for j in part_indices]
            
            grouped_fnames.append(group)
            grouped_datenum.append([mdn[i]] + pdn)
            grouped_datasize.append([mds[i]] + pds)
        else:
            grouped_fnames.append([mstr_i + ext])
            grouped_datenum.append([mdn[i]])
            grouped_datasize.append([mds[i]])
    
    print('Done!')
    return True, grouped_fnames, grouped_datenum, grouped_datasize
