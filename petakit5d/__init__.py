"""
PetaKit5D Python Library

Tools for efficient and scalable processing of petabyte-scale 5D live images.
This is a Python port of the MATLAB PetaKit5D library.

Example usage:
    >>> from petakit5d import read_tiff, filter_gauss_3d, decon_psf2otf
    >>> data = read_tiff("myimage.tif")
    >>> filtered = filter_gauss_3d(data, sigma=2.0)
"""

__version__ = "0.1.0"
__author__ = "Xiongtao Ruan"

# Import submodules
from . import utils
from . import image_processing
from . import microscope_data_processing

# Import key functions for convenience
# Users can do: from petakit5d import read_tiff, filter_gauss_3d, etc.

# I/O Functions
from .microscope_data_processing import (
    read_tiff,
    write_tiff,
    read_zarr,
    write_zarr,
    create_zarr,
    write_zarr_block,
)

# Image Filtering
from .image_processing import (
    filter_gauss_1d,
    filter_gauss_2d,
    filter_gauss_3d,
    fast_gauss_3d,
    bilateral_filter,
    filter_log,
    filter_log_nd,
    filter_multiscale_log_nd,
    filter_lobg_nd,
    filter_multiscale_lobg_nd,
)

# Volume Processing
from .microscope_data_processing import (
    crop_3d,
    crop_4d,
    trim_border,
    indexing_4d,
    resample_stack_3d,
    imresize3_average,
)

# Deconvolution
from .microscope_data_processing import (
    decon_otf2psf,
    decon_psf2otf,
    decon_mask_edge_erosion,
    psf_gen,
    rotate_psf,
)

# Stitching
from .microscope_data_processing import (
    normxcorr2_max_shift,
    normxcorr3_fast,
    normxcorr3_max_shift,
    feather_blending_3d,
    feather_distance_map_resize_3d,
    check_major_tile_valid,
    normalize_z_stack,
    distance_weight_single_axis,
    stitch_process_filenames,
)

# Morphology
from .image_processing import (
    bw_largest_obj,
    binary_sphere,
    bw_thin,
    skeleton,
    bwn_hood_3d,
)
from .microscope_data_processing import (
    erode_volume_by_2d_projection,
)

# Detection & Analysis
from .image_processing import (
    non_maximum_suppression,
    non_maximum_suppression_3d,
    local_avg_std_2d,
)

# Transforms
from .image_processing import (
    awt_1d,
    awt,
    awt_denoising,
    b3spline_1d,
    b3spline_2d,
    compute_bspline_coefficients,
    interp_bspline_value,
    calc_interp_maxima,
    binterp,
)

# Utilities
from .microscope_data_processing import (
    check_resample_setting,
    estimate_computing_memory,
    group_partial_volume_files,
    integral_image_3d,
    max_pooling_3d,
    min_bbox_3d,
    project_3d_to_2d,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Submodules
    "utils",
    "image_processing",
    "microscope_data_processing",
    # I/O
    "read_tiff",
    "write_tiff",
    "read_zarr",
    "write_zarr",
    "create_zarr",
    "write_zarr_block",
    # Filtering
    "filter_gauss_1d",
    "filter_gauss_2d",
    "filter_gauss_3d",
    "fast_gauss_3d",
    "bilateral_filter",
    "filter_log",
    "filter_log_nd",
    "filter_multiscale_log_nd",
    "filter_lobg_nd",
    "filter_multiscale_lobg_nd",
    # Volume Processing
    "crop_3d",
    "crop_4d",
    "trim_border",
    "indexing_4d",
    "resample_stack_3d",
    "imresize3_average",
    # Deconvolution
    "decon_otf2psf",
    "decon_psf2otf",
    "decon_mask_edge_erosion",
    "psf_gen",
    "rotate_psf",
    # Stitching
    "normxcorr2_max_shift",
    "normxcorr3_fast",
    "normxcorr3_max_shift",
    "feather_blending_3d",
    "feather_distance_map_resize_3d",
    "check_major_tile_valid",
    "normalize_z_stack",
    "distance_weight_single_axis",
    "stitch_process_filenames",
    # Morphology
    "bw_largest_obj",
    "binary_sphere",
    "bw_thin",
    "skeleton",
    "bwn_hood_3d",
    "erode_volume_by_2d_projection",
    # Detection
    "non_maximum_suppression",
    "non_maximum_suppression_3d",
    "local_avg_std_2d",
    # Transforms
    "awt_1d",
    "awt",
    "awt_denoising",
    "b3spline_1d",
    "b3spline_2d",
    "compute_bspline_coefficients",
    "interp_bspline_value",
    "calc_interp_maxima",
    "binterp",
    # Utilities
    "check_resample_setting",
    "estimate_computing_memory",
    "group_partial_volume_files",
    "integral_image_3d",
    "max_pooling_3d",
    "min_bbox_3d",
    "project_3d_to_2d",
]
