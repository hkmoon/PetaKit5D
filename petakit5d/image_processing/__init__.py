"""
Image processing functions for PetaKit5D.

Ported from MATLAB imageProcessing/ directory.
"""

from .contrast import scale_contrast, invert_contrast
from .color import ch2rgb
from .filters import filter_gauss_2d, filter_gauss_3d, filter_gauss_1d, conv3_fast
from .morphology import bw_largest_obj, binary_sphere
from .mask import mask_vectors, angle_filter
from .nms import non_maximum_suppression
# from .splines import b3spline_1d, b3spline_2d  # TODO: Fix broadcasting issues
from .local_stats import local_avg_std_2d
from .gradient import gradient_filter_gauss_2d, gradient_filter_gauss_3d
from .distance import bw_max_direct_dist, bw_n_neighbors
from .visualization import rgb_overlay, z_proj_image
from .neighbors import bwn_hood_3d
from .log_filter import filter_log
from .bilateral import bilateral_filter
from .convn_fft import convn_fft
from .nms_3d import non_maximum_suppression_3d
from .morphology_thin import bw_thin
from .fast_gauss import fast_gauss_3d
from .surface_filter import surface_filter_gauss_3d
from .bspline_coeffs import b3spline_1d, b3spline_2d
from .awt import awt_1d

__all__ = [
    "scale_contrast",
    "invert_contrast",
    "ch2rgb",
    "filter_gauss_2d",
    "filter_gauss_3d",
    "filter_gauss_1d",
    "conv3_fast",
    "bw_largest_obj",
    "binary_sphere",
    "mask_vectors",
    "angle_filter",
    "non_maximum_suppression",
    # "b3spline_1d",  # TODO: Fix broadcasting issues
    # "b3spline_2d",  # TODO: Fix broadcasting issues
    "local_avg_std_2d",
    "gradient_filter_gauss_2d",
    "gradient_filter_gauss_3d",
    "bw_max_direct_dist",
    "bw_n_neighbors",
    "rgb_overlay",
    "z_proj_image",
    "bwn_hood_3d",
    "filter_log",
    "bilateral_filter",
    "convn_fft",
    "non_maximum_suppression_3d",
    "bw_thin",
    "fast_gauss_3d",
    "surface_filter_gauss_3d",
    "b3spline_1d",
    "b3spline_2d",
    "awt_1d",
]
