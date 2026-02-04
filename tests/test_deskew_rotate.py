"""
Tests for deskew_rotate module.
"""

import numpy as np
import pytest
from petakit5d.microscope_data_processing.deskew_rotate import (
    deskew_frame_3d,
    rotate_frame_3d
)


def test_deskew_frame_3d_basic():
    """Test basic deskewing."""
    # Create simple test volume
    frame = np.random.rand(10, 20, 30).astype(np.float32)

    result = deskew_frame_3d(frame, dz=0.5, angle=32.45)

    # Check output shape is larger due to shear
    assert result.shape[0] == 10  # Z same
    assert result.shape[1] == 20  # Y same
    assert result.shape[2] > 30   # X larger

    # Check dtype
    assert result.dtype == np.float32


def test_deskew_frame_3d_reverse():
    """Test reverse scanning direction."""
    frame = np.random.rand(10, 20, 30).astype(np.float32)

    result_forward = deskew_frame_3d(frame, dz=0.5, angle=32.45, reverse=False)
    result_reverse = deskew_frame_3d(frame, dz=0.5, angle=32.45, reverse=True)

    # Results should be different
    assert not np.allclose(result_forward, result_reverse)


def test_deskew_frame_3d_cubic():
    """Test cubic interpolation."""
    frame = np.random.rand(10, 20, 30).astype(np.float32)

    result_linear = deskew_frame_3d(frame, dz=0.5, angle=32.45, interpolation='linear')
    result_cubic = deskew_frame_3d(frame, dz=0.5, angle=32.45, interpolation='cubic')

    # Results should be slightly different
    assert result_linear.shape == result_cubic.shape
    assert not np.allclose(result_linear, result_cubic)


def test_deskew_frame_3d_angles():
    """Test different angles."""
    frame = np.random.rand(10, 20, 30).astype(np.float32)

    result_15 = deskew_frame_3d(frame, dz=0.5, angle=15.0)
    result_30 = deskew_frame_3d(frame, dz=0.5, angle=30.0)
    result_45 = deskew_frame_3d(frame, dz=0.5, angle=45.0)

    # All should produce valid 3D output
    assert result_15.ndim == 3
    assert result_30.ndim == 3
    assert result_45.ndim == 3
    # All should have same Z and Y dimensions as input
    assert result_15.shape[0] == 10
    assert result_30.shape[0] == 10
    assert result_45.shape[0] == 10
    assert result_15.shape[1] == 20
    assert result_30.shape[1] == 20
    assert result_45.shape[1] == 20
    # X dimension should be enlarged
    assert result_15.shape[2] > 30
    assert result_30.shape[2] > 30
    assert result_45.shape[2] > 30


def test_rotate_frame_3d_basic():
    """Test basic rotation."""
    frame = np.random.rand(10, 20, 30).astype(np.float32)

    result = rotate_frame_3d(frame, angle=32.45, dz=0.5, pixel_size=0.108)

    # Check output is 3D
    assert result.ndim == 3

    # With cropping, output Z might be smaller
    assert result.shape[0] <= 10


def test_rotate_frame_3d_angles():
    """Test rotation with different angles."""
    frame = np.ones((10, 20, 30), dtype=np.float32)
    frame[4:6, 8:12, 13:17] = 10  # Add a bright region

    result_15 = rotate_frame_3d(frame, angle=15.0, dz=0.5)
    result_30 = rotate_frame_3d(frame, angle=30.0, dz=0.5)
    result_45 = rotate_frame_3d(frame, angle=45.0, dz=0.5)

    # All should have some non-zero values
    assert np.sum(result_15 > 0) > 0
    assert np.sum(result_30 > 0) > 0
    assert np.sum(result_45 > 0) > 0


def test_deskew_frame_3d_multichannel_axis0():
    """Test multi-channel deskewing with channels on axis 0."""
    # Create multi-channel test volume (C, Z, Y, X)
    n_channels = 3
    frame = np.random.rand(n_channels, 10, 20, 30).astype(np.float32)
    
    result = deskew_frame_3d(frame, dz=0.5, angle=32.45, channel_axis=0)
    
    # Check output shape preserves channel dimension
    assert result.ndim == 4
    assert result.shape[0] == n_channels  # Channels preserved
    assert result.shape[1] == 10  # Z same
    assert result.shape[2] == 20  # Y same
    assert result.shape[3] > 30   # X larger due to shear
    
    # Check dtype
    assert result.dtype == np.float32


def test_deskew_frame_3d_multichannel_axis_minus1():
    """Test multi-channel deskewing with channels on axis -1."""
    # Create multi-channel test volume (Z, Y, X, C)
    n_channels = 2
    frame = np.random.rand(10, 20, 30, n_channels).astype(np.float32)
    
    result = deskew_frame_3d(frame, dz=0.5, angle=32.45, channel_axis=-1)
    
    # Check output shape preserves channel dimension
    assert result.ndim == 4
    assert result.shape[0] == 10  # Z same
    assert result.shape[1] == 20  # Y same
    assert result.shape[2] > 30   # X larger due to shear
    assert result.shape[3] == n_channels  # Channels preserved
    
    # Check dtype
    assert result.dtype == np.float32


def test_deskew_frame_3d_multichannel_different_values():
    """Test that different channels are processed independently."""
    # Create multi-channel volume with different values per channel
    frame = np.zeros((3, 10, 20, 30), dtype=np.float32)
    frame[0] = 1.0  # Channel 0 = 1
    frame[1] = 2.0  # Channel 1 = 2
    frame[2] = 3.0  # Channel 2 = 3
    
    result = deskew_frame_3d(frame, dz=0.5, angle=32.45, channel_axis=0)
    
    # Each channel should maintain different value ranges
    assert np.mean(result[0]) < np.mean(result[1]) < np.mean(result[2])
    # All channels should be deskewed (shape changed)
    for c in range(3):
        assert result[c].shape[2] > 30


def test_rotate_frame_3d_multichannel_axis0():
    """Test multi-channel rotation with channels on axis 0."""
    # Create multi-channel test volume (C, Z, Y, X)
    n_channels = 3
    frame = np.random.rand(n_channels, 10, 20, 30).astype(np.float32)
    
    result = rotate_frame_3d(frame, angle=32.45, dz=0.5, channel_axis=0)
    
    # Check output shape preserves channel dimension
    assert result.ndim == 4
    assert result.shape[0] == n_channels  # Channels preserved
    # Other dimensions may vary due to rotation and cropping
    assert result.shape[1] <= 10  # Z might be cropped
    
    # Check dtype
    assert result.dtype == np.float32


def test_rotate_frame_3d_multichannel_axis_minus1():
    """Test multi-channel rotation with channels on axis -1."""
    # Create multi-channel test volume (Z, Y, X, C)
    n_channels = 2
    frame = np.random.rand(10, 20, 30, n_channels).astype(np.float32)
    
    result = rotate_frame_3d(frame, angle=32.45, dz=0.5, channel_axis=-1)
    
    # Check output shape preserves channel dimension
    assert result.ndim == 4
    assert result.shape[-1] == n_channels  # Channels preserved
    # Other dimensions may vary due to rotation and cropping
    assert result.shape[0] <= 10  # Z might be cropped
    
    # Check dtype
    assert result.dtype == np.float32


def test_rotate_frame_3d_multichannel_no_crop():
    """Test multi-channel rotation without cropping."""
    # Create multi-channel test volume with bright region
    frame = np.ones((2, 10, 20, 30), dtype=np.float32)
    frame[:, 4:6, 8:12, 13:17] = 10  # Bright region in both channels
    
    result = rotate_frame_3d(frame, angle=32.45, dz=0.5, crop=False, channel_axis=0)
    
    # With crop=False, Z dimension should be same
    assert result.shape[0] == 2  # Channels preserved
    assert result.shape[1] == 10  # Z preserved (no crop)
    
    # Both channels should have rotated content
    assert np.sum(result[0] > 1) > 0
    assert np.sum(result[1] > 1) > 0


def test_deskew_and_rotate_multichannel_pipeline():
    """Test complete pipeline: deskew then rotate with multi-channel data."""
    # Create multi-channel test volume (C, Z, Y, X)
    n_channels = 3
    frame = np.random.rand(n_channels, 10, 20, 30).astype(np.float32)
    
    # Deskew
    deskewed = deskew_frame_3d(frame, dz=0.5, angle=32.45, channel_axis=0)
    
    # Rotate
    rotated = rotate_frame_3d(deskewed, angle=32.45, dz=0.5, channel_axis=0)
    
    # Check both operations preserved channel dimension
    assert deskewed.shape[0] == n_channels
    assert rotated.shape[0] == n_channels
    
    # Check all outputs are 4D
    assert deskewed.ndim == 4
    assert rotated.ndim == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
