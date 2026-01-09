"""
Tests for 3D non-maximum suppression.
"""

import numpy as np
import pytest
from petakit5d.image_processing import non_maximum_suppression_3d


def test_nms_3d_basic():
    """Test basic 3D NMS functionality."""
    # Create simple 3D vector field
    shape = (20, 20, 10)
    u = np.random.rand(*shape)
    v = np.random.rand(*shape)
    w = np.random.rand(*shape)
    
    result = non_maximum_suppression_3d(u, v, w)
    
    # Check output shape
    assert result.shape == shape
    
    # Check output is non-negative
    assert np.all(result >= 0)


def test_nms_3d_uniform_field():
    """Test NMS on uniform vector field."""
    shape = (15, 15, 8)
    # Uniform field pointing in x-direction
    u = np.ones(shape)
    v = np.zeros(shape)
    w = np.zeros(shape)
    
    result = non_maximum_suppression_3d(u, v, w)
    
    # Most points should be suppressed in a uniform field
    assert np.sum(result > 0) < np.prod(shape) * 0.5


def test_nms_3d_zero_magnitude():
    """Test NMS with zero magnitude vectors."""
    shape = (10, 10, 5)
    u = np.zeros(shape)
    v = np.zeros(shape)
    w = np.zeros(shape)
    
    result = non_maximum_suppression_3d(u, v, w)
    
    # All should be zero
    assert np.allclose(result, 0)


def test_nms_3d_single_peak():
    """Test NMS with single strong peak."""
    shape = (20, 20, 10)
    u = np.random.rand(*shape) * 0.1
    v = np.random.rand(*shape) * 0.1
    w = np.random.rand(*shape) * 0.1
    
    # Add a strong peak in the center
    center = (10, 10, 5)
    u[center] = 10.0
    v[center] = 0.0
    w[center] = 0.0
    
    result = non_maximum_suppression_3d(u, v, w)
    
    # Peak should be retained
    magnitude = np.sqrt(u**2 + v**2 + w**2)
    assert result[center] > 0
    # Peak should have high magnitude
    assert result[center] == pytest.approx(magnitude[center], rel=0.01)


def test_nms_3d_ridge_structure():
    """Test NMS on ridge-like structure."""
    shape = (30, 30, 15)
    Y, X, Z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    
    # Create a ridge along x-axis
    u = np.ones(shape)
    v = -(Y - shape[0]//2) * 0.1  # Gradient pointing away from center
    w = np.zeros(shape)
    
    result = non_maximum_suppression_3d(u, v, w)
    
    # NMS should suppress most points, test basic functionality
    assert result.shape == shape
    # Some suppression should occur
    magnitude = np.sqrt(u**2 + v**2 + w**2)
    assert np.sum(result > 0) <= np.sum(magnitude > 0)


def test_nms_3d_input_validation():
    """Test input validation."""
    # Wrong number of dimensions
    u = np.random.rand(10, 10)
    v = np.random.rand(10, 10)
    w = np.random.rand(10, 10)
    
    with pytest.raises(ValueError, match="must be 3-dimensional"):
        non_maximum_suppression_3d(u, v, w)
    
    # Mismatched shapes
    u = np.random.rand(10, 10, 5)
    v = np.random.rand(10, 10, 5)
    w = np.random.rand(10, 10, 6)  # Different size
    
    with pytest.raises(ValueError, match="must have equal size"):
        non_maximum_suppression_3d(u, v, w)


def test_nms_3d_normalized_output():
    """Test that NMS output matches magnitude at maxima."""
    shape = (15, 15, 8)
    u = np.random.rand(*shape)
    v = np.random.rand(*shape)
    w = np.random.rand(*shape)
    
    result = non_maximum_suppression_3d(u, v, w)
    
    # Where result is non-zero, it should equal magnitude
    magnitude = np.sqrt(u**2 + v**2 + w**2)
    non_zero_mask = result > 0
    
    if np.any(non_zero_mask):
        assert np.allclose(result[non_zero_mask], magnitude[non_zero_mask], rtol=0.01)


def test_nms_3d_gradient_field():
    """Test NMS on gradient field."""
    shape = (25, 25, 12)
    Y, X, Z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    
    # Create radial gradient field
    center = np.array([shape[0]//2, shape[1]//2, shape[2]//2])
    u = X - center[1]
    v = Y - center[0]
    w = Z - center[2]
    
    result = non_maximum_suppression_3d(u, v, w)
    
    # Points near center should be suppressed
    assert result[center[0], center[1], center[2]] == 0


def test_nms_3d_small_volume():
    """Test NMS on small 3D volume."""
    shape = (5, 5, 3)
    u = np.random.rand(*shape)
    v = np.random.rand(*shape)
    w = np.random.rand(*shape)
    
    result = non_maximum_suppression_3d(u, v, w)
    
    assert result.shape == shape


def test_nms_3d_anisotropic():
    """Test NMS on anisotropic volume."""
    shape = (50, 30, 10)
    u = np.random.rand(*shape)
    v = np.random.rand(*shape)
    w = np.random.rand(*shape)
    
    result = non_maximum_suppression_3d(u, v, w)
    
    assert result.shape == shape


def test_nms_3d_edge_behavior():
    """Test NMS behavior at volume edges."""
    shape = (20, 20, 10)
    u = np.random.rand(*shape)
    v = np.random.rand(*shape)
    w = np.random.rand(*shape)
    
    # Add peaks at edges
    u[0, 10, 5] = 10.0
    v[19, 10, 5] = 10.0
    w[10, 10, 0] = 10.0
    
    result = non_maximum_suppression_3d(u, v, w)
    
    # Edge peaks might be retained depending on direction
    assert result.shape == shape
    # At least some suppression should occur
    magnitude = np.sqrt(u**2 + v**2 + w**2)
    assert np.sum(result > 0) < np.sum(magnitude > 0)


def test_nms_3d_nan_handling():
    """Test NMS with NaN values in input."""
    shape = (15, 15, 8)
    u = np.random.rand(*shape)
    v = np.random.rand(*shape)
    w = np.random.rand(*shape)
    
    # Add some NaN values
    u[5, 5, 3] = np.nan
    v[8, 8, 5] = np.nan
    
    result = non_maximum_suppression_3d(u, v, w)
    
    # Should handle NaN gracefully
    assert result.shape == shape
    # NaN locations get normalized to 0, resulting magnitude is also 0
    assert np.isnan(result[5, 5, 3]) or result[5, 5, 3] == 0
