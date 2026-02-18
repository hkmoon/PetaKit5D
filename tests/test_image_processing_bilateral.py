"""
Tests for bilateral filtering.
"""

import numpy as np
import pytest
from petakit5d.image_processing import bilateral_filter


def test_bilateral_filter_basic():
    """Test basic bilateral filtering functionality."""
    # Create a simple test image
    img = np.random.rand(50, 50)
    
    # Apply bilateral filter
    result = bilateral_filter(img, sigma_s=2.0, sigma_r=0.1)
    
    # Check output shape matches input
    assert result.shape == img.shape
    
    # Check output is real-valued
    assert np.isrealobj(result)


def test_bilateral_filter_edge_preservation():
    """Test that bilateral filter preserves edges."""
    # Create an image with a sharp edge
    img = np.zeros((50, 50))
    img[:, 25:] = 1.0
    
    # Apply bilateral filter with appropriate parameters
    result = bilateral_filter(img, sigma_s=2.0, sigma_r=0.1)
    
    # Check that the edge is still relatively sharp
    # The left side should be close to 0, right side close to 1
    assert np.mean(result[:, :20]) < 0.2
    assert np.mean(result[:, 30:]) > 0.8


def test_bilateral_filter_smoothing():
    """Test that bilateral filter smooths uniform regions."""
    # Create noisy uniform region
    img = np.random.randn(50, 50) * 0.1 + 0.5
    
    # Apply bilateral filter
    result = bilateral_filter(img, sigma_s=2.0, sigma_r=0.1)
    
    # Filtered image should have lower variance
    assert np.std(result) < np.std(img)


def test_bilateral_filter_parameters():
    """Test bilateral filter with different parameter values."""
    img = np.random.rand(30, 30)
    
    # Test with different spatial sigmas
    result1 = bilateral_filter(img, sigma_s=1.0, sigma_r=0.1)
    result2 = bilateral_filter(img, sigma_s=3.0, sigma_r=0.1)
    assert result1.shape == result2.shape == img.shape
    
    # Test with different range sigmas
    result3 = bilateral_filter(img, sigma_s=2.0, sigma_r=0.05)
    result4 = bilateral_filter(img, sigma_s=2.0, sigma_r=0.2)
    assert result3.shape == result4.shape == img.shape


def test_bilateral_filter_constant_image():
    """Test bilateral filter on constant image."""
    img = np.ones((30, 30)) * 0.5
    
    result = bilateral_filter(img, sigma_s=2.0, sigma_r=0.1)
    
    # Constant image should remain constant
    assert np.allclose(result, img, atol=1e-10)


def test_bilateral_filter_small_image():
    """Test bilateral filter on small image."""
    img = np.random.rand(10, 10)
    
    result = bilateral_filter(img, sigma_s=1.0, sigma_r=0.1)
    
    assert result.shape == img.shape


def test_bilateral_filter_large_sigma_r():
    """Test bilateral filter with large range sigma."""
    img = np.random.rand(30, 30)
    
    # Large sigma_r should make it behave more like Gaussian blur
    result = bilateral_filter(img, sigma_s=2.0, sigma_r=1.0)
    
    assert result.shape == img.shape


def test_bilateral_filter_invalid_input():
    """Test bilateral filter with invalid input."""
    # 3D input should fail
    img_3d = np.random.rand(10, 10, 10)
    
    with pytest.raises(ValueError, match="only supports 2D images"):
        bilateral_filter(img_3d, sigma_s=2.0, sigma_r=0.1)
    
    # 1D input should fail
    img_1d = np.random.rand(10)
    
    with pytest.raises(ValueError, match="only supports 2D images"):
        bilateral_filter(img_1d, sigma_s=2.0, sigma_r=0.1)


def test_bilateral_filter_normalized_range():
    """Test bilateral filter with normalized intensity range."""
    # Image in [0, 1] range
    img = np.random.rand(40, 40)
    result = bilateral_filter(img, sigma_s=2.0, sigma_r=0.1)
    
    # Output should be in reasonable range
    assert np.min(result) >= -0.1  # Allow small numerical errors
    assert np.max(result) <= 1.1


def test_bilateral_filter_high_contrast():
    """Test bilateral filter on high contrast image."""
    # Create high contrast image
    img = np.random.choice([0.0, 1.0], size=(40, 40))
    
    result = bilateral_filter(img, sigma_s=2.0, sigma_r=0.2)
    
    assert result.shape == img.shape
    # Should preserve the binary nature to some degree
    assert np.min(result) < 0.3
    assert np.max(result) > 0.7
