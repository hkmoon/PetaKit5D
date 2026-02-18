"""
Tests for local statistics functions.
"""

import numpy as np
import pytest
from petakit5d.image_processing import local_avg_std_2d


class TestLocalAvgStd2D:
    """Test suite for local_avg_std_2d function."""
    
    def test_basic_functionality(self):
        """Test basic local statistics computation."""
        img = np.random.rand(50, 50) * 100
        window_size = 5
        
        avg, std = local_avg_std_2d(img, window_size)
        
        assert avg.shape == img.shape
        assert std.shape == img.shape
        assert not np.any(np.isnan(avg))
        assert not np.any(np.isnan(std))
    
    def test_invalid_window_size(self):
        """Test with even window size (should raise error)."""
        img = np.random.rand(30, 30)
        
        with pytest.raises(ValueError, match="must be an odd integer"):
            local_avg_std_2d(img, 4)  # Even number
    
    def test_with_nan_values(self):
        """Test handling of NaN values."""
        img = np.random.rand(40, 40)
        img[10:15, 10:15] = np.nan
        
        avg, std = local_avg_std_2d(img, 5)
        
        assert avg.shape == img.shape
        assert std.shape == img.shape
        # NaN regions should remain NaN
        assert np.all(np.isnan(avg[10:15, 10:15]))
        assert np.all(np.isnan(std[10:15, 10:15]))
    
    def test_constant_image(self):
        """Test with constant image."""
        img = np.ones((30, 30)) * 42
        
        avg, std = local_avg_std_2d(img, 3)
        
        # Average should be close to constant value
        assert np.allclose(avg, 42, rtol=1e-10)
        # Std should be close to zero
        assert np.allclose(std, 0, atol=1e-10)
    
    def test_large_window(self):
        """Test with large window size."""
        img = np.random.rand(100, 100)
        
        avg, std = local_avg_std_2d(img, 21)
        
        assert avg.shape == img.shape
        assert std.shape == img.shape
    
    def test_small_image(self):
        """Test with small image."""
        img = np.random.rand(10, 10)
        
        avg, std = local_avg_std_2d(img, 3)
        
        assert avg.shape == img.shape
        assert std.shape == img.shape
    
    def test_std_non_negative(self):
        """Test that standard deviation is non-negative."""
        img = np.random.rand(50, 50)
        
        _, std = local_avg_std_2d(img, 5)
        
        assert np.all(std >= 0)
