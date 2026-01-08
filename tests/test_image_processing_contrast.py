"""
Unit tests for image processing contrast utilities.
"""

import pytest
import numpy as np
from petakit5d.image_processing.contrast import scale_contrast, invert_contrast


class TestScaleContrast:
    """Test cases for scale_contrast function."""
    
    def test_default_range(self):
        """Test with default input and output ranges."""
        img = np.array([[0, 50], [100, 200]])
        result = scale_contrast(img)
        
        # Should scale from [0, 200] to [0, 255]
        assert result[0, 0] == 0.0
        assert result[1, 1] == 255.0
    
    def test_custom_output_range(self):
        """Test with custom output range."""
        img = np.array([[0, 50], [100, 200]])
        result = scale_contrast(img, range_out=(0, 1))
        
        assert result[0, 0] == 0.0
        assert result[1, 1] == 1.0
        assert 0.2 <= result[0, 1] <= 0.3  # 50/200 = 0.25
    
    def test_custom_input_range(self):
        """Test with custom input range."""
        img = np.array([[0, 50], [100, 200]])
        result = scale_contrast(img, range_in=(50, 150), range_out=(0, 100))
        
        # Values outside range should still be scaled
        assert result[0, 1] == 0.0  # 50 maps to 0
        assert result[1, 0] == 50.0  # 100 maps to 50
    
    def test_zero_range(self):
        """Test with zero input range."""
        img = np.array([[5, 5], [5, 5]])
        result = scale_contrast(img)
        
        # Should return zeros
        assert np.all(result == 0)
    
    def test_preserves_shape(self):
        """Test that output shape matches input."""
        img = np.random.rand(10, 20, 5)
        result = scale_contrast(img)
        
        assert result.shape == img.shape


class TestInvertContrast:
    """Test cases for invert_contrast function."""
    
    def test_basic_inversion(self):
        """Test basic contrast inversion."""
        img = np.array([[0, 50], [100, 200]])
        result = invert_contrast(img, range_in=(0, 200))
        
        # 0 should map to 200, 200 should map to 0
        assert result[0, 0] == 200.0
        assert result[1, 1] == 0.0
        assert result[0, 1] == 150.0  # 50 -> 150
        assert result[1, 0] == 100.0  # 100 -> 100
    
    def test_default_range(self):
        """Test with automatic range detection."""
        img = np.array([[10, 20], [30, 40]])
        result = invert_contrast(img)
        
        # Range is [10, 40], sum is 50
        assert result[0, 0] == 40.0  # -10 + 50
        assert result[1, 1] == 10.0  # -40 + 50
    
    def test_truncation(self):
        """Test that values are truncated to range."""
        img = np.array([[-10, 50], [150, 250]])
        result = invert_contrast(img, range_in=(0, 200))
        
        # -10 should be clipped to 0, then inverted to 200
        assert result[0, 0] == 200.0
        # 250 should be clipped to 200, then inverted to 0
        assert result[1, 1] == 0.0
    
    def test_preserves_shape(self):
        """Test that output shape matches input."""
        img = np.random.rand(10, 20, 5)
        result = invert_contrast(img)
        
        assert result.shape == img.shape
