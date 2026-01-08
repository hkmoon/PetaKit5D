"""
Unit tests for microscope data processing crop utilities.
"""

import pytest
import numpy as np
from petakit5d.microscope_data_processing.crop import crop_3d, crop_4d


class TestCrop3D:
    """Test cases for crop_3d function."""
    
    def test_basic_cropping(self):
        """Test basic 3D cropping."""
        data = np.arange(1000).reshape(10, 10, 10)
        
        # MATLAB 1-based: crop from (2,2,2) to (5,5,5)
        bbox = (2, 2, 2, 5, 5, 5)
        result = crop_3d(data, bbox)
        
        # Should be 4x4x4 (indices 1:5 in Python is 4 elements)
        assert result.shape == (4, 4, 4)
        
        # Verify first element matches
        assert result[0, 0, 0] == data[1, 1, 1]
    
    def test_full_range(self):
        """Test cropping full range."""
        data = np.random.rand(20, 30, 15)
        
        # MATLAB 1-based: full range
        bbox = (1, 1, 1, 20, 30, 15)
        result = crop_3d(data, bbox)
        
        assert result.shape == data.shape
        assert np.array_equal(result, data)
    
    def test_single_slice(self):
        """Test cropping to single slice."""
        data = np.random.rand(10, 10, 10)
        
        # MATLAB 1-based: single Z slice
        bbox = (1, 1, 5, 10, 10, 5)
        result = crop_3d(data, bbox)
        
        assert result.shape == (10, 10, 1)
    
    def test_with_array_bbox(self):
        """Test with numpy array bbox."""
        data = np.random.rand(10, 10, 10)
        bbox = np.array([2, 2, 2, 5, 5, 5])
        
        result = crop_3d(data, bbox)
        assert result.shape == (4, 4, 4)


class TestCrop4D:
    """Test cases for crop_4d function."""
    
    def test_basic_cropping_4d(self):
        """Test basic 4D cropping."""
        data = np.random.rand(20, 30, 15, 10)
        
        # MATLAB 1-based: crop spatial and temporal
        bbox = (5, 10, 3, 2, 15, 25, 12, 8)
        result = crop_4d(data, bbox)
        
        # (5:15 = 11, 10:25 = 16, 3:12 = 10, 2:8 = 7)
        assert result.shape == (11, 16, 10, 7)
    
    def test_full_range_4d(self):
        """Test cropping full range in 4D."""
        data = np.random.rand(10, 20, 15, 5)
        
        # MATLAB 1-based: full range
        bbox = (1, 1, 1, 1, 10, 20, 15, 5)
        result = crop_4d(data, bbox)
        
        assert result.shape == data.shape
        assert np.array_equal(result, data)
    
    def test_temporal_subset(self):
        """Test cropping temporal dimension."""
        data = np.random.rand(10, 10, 10, 20)
        
        # MATLAB 1-based: subset of time points
        bbox = (1, 1, 1, 5, 10, 10, 10, 15)
        result = crop_4d(data, bbox)
        
        assert result.shape == (10, 10, 10, 11)
    
    def test_with_array_bbox_4d(self):
        """Test with numpy array bbox for 4D."""
        data = np.random.rand(10, 10, 10, 5)
        bbox = np.array([2, 2, 2, 2, 8, 8, 8, 4])
        
        result = crop_4d(data, bbox)
        assert result.shape == (7, 7, 7, 3)
