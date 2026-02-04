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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
