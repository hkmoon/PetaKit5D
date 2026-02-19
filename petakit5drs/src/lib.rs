//! Petakit5D Rust implementation
//!
//! This crate provides Rust implementations of the core functionality from
//! the Python Petakit5D library, focusing on image processing and microscopy data handling.
//!
//! # Module Layout
//! - [`utils`] — UUID, path, hostname, FFT-factor utilities
//! - [`io`] — TIFF and ZARR file reading/writing
//! - [`image_processing`] — B-spline, Gaussian, and convolution filters
//! - [`microscope_data_processing`] — Crop, resample, project, and ZARR helpers
//! - [`stitch`] — Cross-correlation and feather blending for tile stitching
//! - [`point_detection`] — Local maxima, Gaussian fitting, spatial queries

pub mod image_processing;
pub mod io;
pub mod microscope_data_processing;
pub mod point_detection;
pub mod stitch;
pub mod utils;

// Re-export public API from each module.
// Note: both `io` and `microscope_data_processing` define a `create_zarr` function
// with different signatures and error types.  We do NOT use glob re-exports here to
// avoid the ambiguous-glob-reexport warning.  Callers can use the fully-qualified
// paths `petakit5drs::io::create_zarr` and
// `petakit5drs::microscope_data_processing::create_zarr` when they need to distinguish
// between the two, or import each module directly.

pub use image_processing::{
    compute_bspline_coefficients, conv3_fast, filter_gauss_1d, filter_gauss_2d, filter_gauss_3d,
    ImageProcessingError,
};
pub use io::{
    read_tiff,
    read_zarr,
    write_tiff,
    write_zarr,
    IoError,
    // io::create_zarr is available as `petakit5drs::io::create_zarr`
};
pub use microscope_data_processing::{
    crop_3d,
    crop_4d,
    group_partial_volume_files,
    project_3d_to_2d,
    resample_stack_3d,
    MicroscopeProcessingError,
    // microscope_data_processing::create_zarr is available as
    // `petakit5drs::microscope_data_processing::create_zarr`
};
pub use point_detection::{
    fit_gaussian_mixtures_3d, fit_gaussians_3d, get_cell_volume, get_intensity_cohorts,
    get_multiplicity, get_short_path, kdtree_ball_query, locmax3d, point_source_detection_3d,
    threshold_otsu, Gaussian3D, PointDetectionError,
};
pub use stitch::{
    check_major_tile_valid, check_slurm_cluster, feather_blending_3d,
    feather_distance_map_resize_3d, normxcorr2_max_shift, normxcorr3_fast,
    stitch_process_filenames, StitchingError,
};
pub use utils::{
    axis_order_mapping, data_type_to_byte_number, fast_power, find_good_factor_number,
    get_hostname, get_image_bounding_box, get_image_data_type, get_image_size, get_uuid,
    ib3spline_1d, ib3spline_2d, mat2str_comma, mkdir_recursive, read_text_file, simplify_path,
    write_json_file, write_text_file,
};
