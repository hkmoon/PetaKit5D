"""
Unit tests for image processing color utilities.
"""

import pytest
import numpy as np
from petakit5d.image_processing.color import ch2rgb


class TestCh2rgb:
    """Test cases for ch2rgb function."""
    
    def test_all_channels(self):
        """Test with all three channels provided."""
        r = np.random.rand(10, 10) * 100
        g = np.random.rand(10, 10) * 100
        b = np.random.rand(10, 10) * 100
        
        result = ch2rgb(r, g, b)
        
        assert result.shape == (10, 10, 3)
        assert result.dtype == np.uint8
    
    def test_missing_red(self):
        """Test with red channel as None."""
        g = np.random.rand(10, 10) * 255
        b = np.random.rand(10, 10) * 255
        
        result = ch2rgb(None, g, b)
        
        assert result.shape == (10, 10, 3)
        # Red channel should be close to 0 (scaled from array of zeros)
        assert np.all(result[:, :, 0] < 10)  # Allow small rounding
        # Green and blue should have non-zero values since they have variation
        assert np.mean(result[:, :, 1]) > 50  # Green should have some values
        assert np.mean(result[:, :, 2]) > 50  # Blue should have some values
    
    def test_missing_green(self):
        """Test with green channel as None."""
        r = np.ones((10, 10)) * 255
        b = np.ones((10, 10)) * 255
        
        result = ch2rgb(r, None, b)
        
        assert result.shape == (10, 10, 3)
        assert np.all(result[:, :, 1] == 0)  # Green should be 0
    
    def test_missing_blue(self):
        """Test with blue channel as None."""
        r = np.ones((10, 10)) * 255
        g = np.ones((10, 10)) * 255
        
        result = ch2rgb(r, g, None)
        
        assert result.shape == (10, 10, 3)
        assert np.all(result[:, :, 2] == 0)  # Blue should be 0
    
    def test_all_none_raises_error(self):
        """Test that all None channels raises an error."""
        with pytest.raises(ValueError, match="At least one channel"):
            ch2rgb(None, None, None)
    
    def test_scaling(self):
        """Test that channels are scaled to [0, 255]."""
        r = np.array([[0, 0.5], [1, 2]])  # Range [0, 2]
        
        result = ch2rgb(r, None, None)
        
        # Should be scaled from [0, 2] to [0, 255]
        assert result[0, 0, 0] == 0
        assert result[1, 1, 0] == 255
