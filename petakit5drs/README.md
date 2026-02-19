# Petakit5D Rust Implementation

This repository contains a Rust implementation of the Petakit5D Python library, designed to provide improved performance and memory safety while maintaining compatibility with the original functionality.

## Current Implementation Status

The following modules have been implemented:

1. **Image Processing** - Core mathematical functions including:
   - `compute_bspline_coefficients` - B-spline coefficient computation
   - Basic interpolation functions

2. **I/O Operations** - File reading/writing capabilities:
   - TIFF file handling
   - ZARR file handling

3. **Utilities** - Supporting functions:
   - File operations
   - Data type handling
   - Directory operations

## Key Features

- **Memory Safety**: Leverages Rust's ownership model to prevent memory-related issues
- **Performance**: Zero-cost abstractions and optimized mathematical operations
- **Type Safety**: Strong typing prevents runtime errors
- **API Compatibility**: Maintains similar interfaces to Python versions

## Dependencies

The implementation uses the following key Rust crates:

- `ndarray` - Multi-dimensional array operations
- `nalgebra` - Linear algebra operations
- `rustfft` - Fast Fourier Transform
- `tiff` - TIFF file I/O
- `zarr` - ZARR format support
- `num-traits` - Mathematical traits
- `statrs` - Statistical functions
- `thiserror` - Error handling
- `serde` - Serialization/deserialization

## Building and Testing

To build this project, you'll need to have Rust installed:

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build the project
cd petakit5drs
cargo build

# Run tests
cargo test
```

## Implementation Plan

This implementation follows the detailed plan outlined in IMPLEMENTATION_PLAN.md:

1. **Core Mathematical Functions** - Implemented (image processing)
2. **I/O Operations** - Partially implemented (TIFF/ZARR)
3. **Microscope Data Processing** - In progress
4. **Utilities** - Partially implemented
5. **Stitching Operations** - In progress

## Future Work

- Complete implementation of all Python modules
- Performance optimization and benchmarking
- Integration with existing Python ecosystem
- Documentation and examples