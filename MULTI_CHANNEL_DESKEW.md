# Multi-Channel Support for Deskew and Rotate Functions

## Overview

The `deskew_frame_3d` and `rotate_frame_3d` functions now support multi-channel image data while preserving the channel dimension throughout the processing pipeline.

## Features

- **Multi-channel processing**: Process 4D arrays with multiple channels
- **Channel preservation**: Output maintains the same channel structure as input
- **Flexible axis specification**: Support channels on axis 0 or -1
- **Independent processing**: Each channel is deskewed/rotated independently
- **Backward compatible**: Existing 3D (single-channel) code works unchanged

## Usage

### Basic Multi-Channel Deskewing

```python
import numpy as np
from petakit5d import deskew_frame_3d, rotate_frame_3d

# Create multi-channel data (C, Z, Y, X)
# 3 channels, 10 Z-slices, 20 Y pixels, 30 X pixels
data = np.random.rand(3, 10, 20, 30).astype(np.float32)

# Deskew with channels on axis 0
deskewed = deskew_frame_3d(
    data, 
    dz=0.5,           # Z step size in microns
    angle=32.45,      # Skew angle in degrees
    pixel_size=0.108, # XY pixel size in microns
    channel_axis=0    # Channels on first axis
)

print(f"Input shape: {data.shape}")           # (3, 10, 20, 30)
print(f"Deskewed shape: {deskewed.shape}")    # (3, 10, 20, 66) - X enlarged due to shear
```

### Multi-Channel Rotation

```python
# Rotate the deskewed data for isotropic visualization
rotated = rotate_frame_3d(
    deskewed,
    angle=32.45,
    dz=0.5,
    pixel_size=0.108,
    channel_axis=0
)

print(f"Rotated shape: {rotated.shape}")  # (3, Z_new, 20, 66) - Z may be cropped
```

### Alternative Channel Layout (Z, Y, X, C)

```python
# Channels on last axis instead of first
data_alt = np.random.rand(10, 20, 30, 3).astype(np.float32)

# Deskew with channels on axis -1
deskewed_alt = deskew_frame_3d(
    data_alt,
    dz=0.5,
    angle=32.45,
    channel_axis=-1  # Channels on last axis
)

print(f"Input shape: {data_alt.shape}")        # (10, 20, 30, 3)
print(f"Deskewed shape: {deskewed_alt.shape}") # (10, 20, 66, 3)
```

### Complete Pipeline Example

```python
# RGB or multi-fluorophore imaging
n_channels = 3
raw_data = load_lsfm_data()  # Shape: (C, Z, Y, X)

# Step 1: Deskew to correct for oblique imaging plane
deskewed = deskew_frame_3d(
    raw_data,
    dz=0.5,
    angle=32.45,
    pixel_size=0.108,
    interpolation='cubic',  # Higher quality
    channel_axis=0
)

# Step 2: Rotate for isotropic visualization
rotated = rotate_frame_3d(
    deskewed,
    angle=32.45,
    dz=0.5,
    pixel_size=0.108,
    crop=True,           # Remove empty slices
    channel_axis=0
)

# All channels are preserved throughout
print(f"Channels preserved: {raw_data.shape[0]} → {rotated.shape[0]}")
```

## Parameters

### deskew_frame_3d

```python
deskew_frame_3d(
    frame,              # Input array (3D or 4D)
    dz,                 # Z step size (microns)
    angle,              # Skew angle (degrees)
    pixel_size=0.108,   # XY pixel size (microns)
    reverse=False,      # Reverse scan direction
    interpolation='linear',  # 'linear' or 'cubic'
    channel_axis=None   # Channel axis for 4D input (0 or -1)
)
```

### rotate_frame_3d

```python
rotate_frame_3d(
    frame,              # Input array (3D or 4D)
    angle,              # Rotation angle (degrees)
    dz,                 # Z step size (microns)
    pixel_size=0.108,   # XY pixel size (microns)
    reverse=False,      # Reverse rotation
    crop=True,          # Crop empty boundaries
    channel_axis=None   # Channel axis for 4D input (0 or -1)
)
```

## Input/Output Shapes

### Channel Axis = 0 (First Dimension)

| Input Shape | Operation | Output Shape | Notes |
|-------------|-----------|--------------|-------|
| (C, Z, Y, X) | deskew | (C, Z, Y, X') | X' > X due to shear |
| (C, Z, Y, X) | rotate | (C, Z', Y, X) | Z' ≤ Z if cropped |
| (3, 10, 20, 30) | deskew | (3, 10, 20, 66) | Example with angle=32.45° |
| (2, 10, 20, 30) | rotate | (2, 8, 20, 30) | Example with cropping |

### Channel Axis = -1 (Last Dimension)

| Input Shape | Operation | Output Shape | Notes |
|-------------|-----------|--------------|-------|
| (Z, Y, X, C) | deskew | (Z, Y, X', C) | X' > X due to shear |
| (Z, Y, X, C) | rotate | (Z', Y, X, C) | Z' ≤ Z if cropped |
| (10, 20, 30, 3) | deskew | (10, 20, 66, 3) | Example with angle=32.45° |
| (10, 20, 30, 2) | rotate | (8, 20, 30, 2) | Example with cropping |

## Backward Compatibility

### Single-Channel Data (3D)

Existing code works without any changes:

```python
# 3D input (no channel dimension)
data_3d = np.random.rand(10, 20, 30).astype(np.float32)

# Works exactly as before (channel_axis=None by default)
deskewed_3d = deskew_frame_3d(data_3d, dz=0.5, angle=32.45)

print(f"Input: {data_3d.shape}")        # (10, 20, 30)
print(f"Output: {deskewed_3d.shape}")   # (10, 20, 66)
```

### Singleton Dimension Handling

4D arrays with singleton dimensions are still auto-squeezed when `channel_axis=None`:

```python
# 4D with singleton dimension
data_4d = np.random.rand(1, 10, 20, 30).astype(np.float32)

# Auto-squeezed to 3D (backward compatible)
deskewed = deskew_frame_3d(data_4d, dz=0.5, angle=32.45)

print(f"Input: {data_4d.shape}")      # (1, 10, 20, 30)
print(f"Output: {deskewed.shape}")    # (10, 20, 66) - squeezed
```

## Channel Independence

Each channel is processed completely independently:

```python
# Create test data with different values per channel
data = np.zeros((3, 10, 20, 30), dtype=np.float32)
data[0] = 1.0  # Channel 0: all ones
data[1] = 2.0  # Channel 1: all twos  
data[2] = 3.0  # Channel 2: all threes

# Deskew
result = deskew_frame_3d(data, dz=0.5, angle=32.45, channel_axis=0)

# Each channel maintains its distinct values
print(f"Channel 0 mean: {result[0].mean():.2f}")  # ~1.0
print(f"Channel 1 mean: {result[1].mean():.2f}")  # ~2.0
print(f"Channel 2 mean: {result[2].mean():.2f}")  # ~3.0
```

## Common Use Cases

### RGB Light Sheet Microscopy

```python
# RGB data from light sheet microscope
rgb_data = load_rgb_lsfm()  # Shape: (3, Z, Y, X)

# Process all channels together
deskewed_rgb = deskew_frame_3d(
    rgb_data,
    dz=0.5,
    angle=32.45,
    channel_axis=0
)

# Visualize individual channels
red_channel = deskewed_rgb[0]
green_channel = deskewed_rgb[1]
blue_channel = deskewed_rgb[2]
```

### Multi-Fluorophore Imaging

```python
# Multiple fluorescent markers
markers = load_multicolor_data()  # Shape: (4, Z, Y, X)
# Channel 0: DAPI (nuclei)
# Channel 1: GFP (protein 1)
# Channel 2: mCherry (protein 2)
# Channel 3: Cy5 (protein 3)

# Deskew all channels
deskewed_markers = deskew_frame_3d(
    markers,
    dz=0.5,
    angle=32.45,
    channel_axis=0
)

# Rotate for visualization
rotated_markers = rotate_frame_3d(
    deskewed_markers,
    angle=32.45,
    dz=0.5,
    channel_axis=0
)

# All 4 channels preserved
assert rotated_markers.shape[0] == 4
```

### Time-Lapse with Multiple Channels

```python
# For time-lapse data, process each timepoint separately
n_timepoints = 100
n_channels = 3

for t in range(n_timepoints):
    # Load single timepoint with multiple channels
    frame_t = load_timepoint(t)  # Shape: (C, Z, Y, X)
    
    # Deskew
    deskewed_t = deskew_frame_3d(
        frame_t,
        dz=0.5,
        angle=32.45,
        channel_axis=0
    )
    
    # Save
    save_deskewed_timepoint(deskewed_t, t)
```

## Error Handling

### Invalid Channel Axis

```python
# Valid: channel_axis in [None, 0, -1, 3]
deskew_frame_3d(data, dz=0.5, angle=32.45, channel_axis=0)   # OK
deskew_frame_3d(data, dz=0.5, angle=32.45, channel_axis=-1)  # OK
deskew_frame_3d(data, dz=0.5, angle=32.45, channel_axis=3)   # OK (same as -1)

# Invalid: other values raise ValueError
try:
    deskew_frame_3d(data, dz=0.5, angle=32.45, channel_axis=1)
except ValueError as e:
    print(f"Error: {e}")  # channel_axis must be 0 or -1
```

### Wrong Dimensionality

```python
# 4D input without channel_axis specification
data_4d = np.random.rand(3, 10, 20, 30)

try:
    # Without channel_axis, tries to squeeze - may fail
    result = deskew_frame_3d(data_4d, dz=0.5, angle=32.45)
except ValueError as e:
    print(f"Error: {e}")
    # Solution: specify channel_axis
    result = deskew_frame_3d(data_4d, dz=0.5, angle=32.45, channel_axis=0)
```

## Performance Considerations

### Memory Usage

Processing is done channel-by-channel, so memory usage scales linearly with number of channels:

```python
# Memory usage ≈ N_channels × (input_size + output_size)
```

### Processing Time

Each channel is processed independently (sequential, not parallel):

```python
# Processing time ≈ N_channels × single_channel_time
```

For large datasets with many channels, consider:
- Processing channels in parallel using `multiprocessing`
- Using chunked/streaming processing for memory efficiency
- Saving intermediate results to disk

## Migration Guide

### Old Code (Single Channel)

```python
# Old: Process single channel
channel_0 = data[0]
deskewed_0 = deskew_frame_3d(channel_0, dz=0.5, angle=32.45)

channel_1 = data[1]
deskewed_1 = deskew_frame_3d(channel_1, dz=0.5, angle=32.45)

# Manually stack
result = np.stack([deskewed_0, deskewed_1], axis=0)
```

### New Code (Multi-Channel)

```python
# New: Process all channels at once
result = deskew_frame_3d(data, dz=0.5, angle=32.45, channel_axis=0)
```

Much simpler and less error-prone!

## Testing

Comprehensive tests are included in `tests/test_deskew_rotate.py`:

- `test_deskew_frame_3d_multichannel_axis0()` - Test channels on axis 0
- `test_deskew_frame_3d_multichannel_axis_minus1()` - Test channels on axis -1
- `test_deskew_frame_3d_multichannel_different_values()` - Test channel independence
- `test_rotate_frame_3d_multichannel_axis0()` - Test rotation with axis 0
- `test_rotate_frame_3d_multichannel_axis_minus1()` - Test rotation with axis -1
- `test_rotate_frame_3d_multichannel_no_crop()` - Test rotation without cropping
- `test_deskew_and_rotate_multichannel_pipeline()` - Test complete pipeline

Run tests:
```bash
pytest tests/test_deskew_rotate.py -v
```

## References

- Original MATLAB functions: `deskewFrame3D.m`, `rotateFrame3D.m`
- Shear transformation: Used to correct oblique imaging plane
- Light sheet microscopy: Requires deskewing for proper 3D reconstruction

## Support

For issues or questions about multi-channel support:
1. Check this documentation
2. Review test examples in `tests/test_deskew_rotate.py`
3. Open an issue on GitHub with example code and error message
