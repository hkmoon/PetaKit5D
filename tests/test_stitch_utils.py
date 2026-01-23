"""
Tests for stitching utilities.
"""

import numpy as np
import pytest
from petakit5d.microscope_data_processing.stitch_utils import feather_distance_map_resize_3d


class TestFeatherDistanceMapResize3d:
    """Tests for feather_distance_map_resize_3d function."""
    
    def test_basic_2d_resize(self):
        """Test basic 2D distance map resizing."""
        dmat = np.ones((50, 50))
        bbox = (1, 1, 1, 100, 100, 1)  # 100x100x1 (2D case)
        wd = 2.0
        
        result = feather_distance_map_resize_3d(dmat, bbox, wd)
        
        # Check output shape
        assert result.shape == (100, 100, 1)
        # Check values are powered
        assert np.allclose(result, 1.0**wd)
    
    def test_basic_3d_resize(self):
        """Test basic 3D distance map resizing."""
        dmat = np.ones((50, 50, 25))
        bbox = (1, 1, 1, 100, 100, 50)  # 100x100x50
        wd = 2.0
        
        result = feather_distance_map_resize_3d(dmat, bbox, wd)
        
        # Check output shape
        assert result.shape == (100, 100, 50)
        assert np.allclose(result, 1.0**wd)
    
    def test_power_weighting(self):
        """Test that power weighting is applied correctly."""
        dmat = np.full((50, 50), 0.5)
        bbox = (1, 1, 1, 50, 50, 1)
        wd = 2.0
        
        result = feather_distance_map_resize_3d(dmat, bbox, wd)
        
        # Should be 0.5^2 = 0.25
        assert np.allclose(result, 0.25)
    
    def test_different_power_values(self):
        """Test with different power values."""
        dmat = np.full((50, 50), 0.5)
        bbox = (1, 1, 1, 50, 50, 1)
        
        for wd in [1.0, 2.0, 3.0, 4.0]:
            result = feather_distance_map_resize_3d(dmat, bbox, wd)
            expected = 0.5**wd
            assert np.allclose(result, expected)
    
    def test_upsampling_2d(self):
        """Test upsampling in 2D."""
        dmat = np.ones((25, 25))
        bbox = (1, 1, 1, 50, 50, 1)
        wd = 1.0
        
        result = feather_distance_map_resize_3d(dmat, bbox, wd)
        
        assert result.shape == (50, 50, 1)
    
    def test_downsampling_2d(self):
        """Test downsampling in 2D."""
        dmat = np.ones((100, 100))
        bbox = (1, 1, 1, 50, 50, 1)
        wd = 1.0
        
        result = feather_distance_map_resize_3d(dmat, bbox, wd)
        
        assert result.shape == (50, 50, 1)
    
    def test_upsampling_3d(self):
        """Test upsampling in 3D."""
        dmat = np.ones((25, 25, 10))
        bbox = (1, 1, 1, 50, 50, 20)
        wd = 1.0
        
        result = feather_distance_map_resize_3d(dmat, bbox, wd)
        
        assert result.shape == (50, 50, 20)
    
    def test_downsampling_3d(self):
        """Test downsampling in 3D."""
        dmat = np.ones((100, 100, 50))
        bbox = (1, 1, 1, 50, 50, 25)
        wd = 1.0
        
        result = feather_distance_map_resize_3d(dmat, bbox, wd)
        
        assert result.shape == (50, 50, 25)
    
    def test_2d_input_3d_output(self):
        """Test converting 2D input to 3D output."""
        dmat = np.ones((50, 50))
        bbox = (1, 1, 1, 50, 50, 10)  # Should create 3D output
        wd = 1.0
        
        result = feather_distance_map_resize_3d(dmat, bbox, wd)
        
        assert result.shape == (50, 50, 10)
    
    def test_gradient_pattern(self):
        """Test with gradient pattern."""
        dmat = np.linspace(0, 1, 100).reshape(10, 10)
        bbox = (1, 1, 1, 20, 20, 1)
        wd = 1.0
        
        result = feather_distance_map_resize_3d(dmat, bbox, wd)
        
        assert result.shape == (20, 20, 1)
        # Check values are in reasonable range
        assert result.min() >= 0
        assert result.max() <= 1
    
    def test_zero_values(self):
        """Test with zero values in distance map."""
        dmat = np.zeros((50, 50))
        bbox = (1, 1, 1, 100, 100, 1)
        wd = 2.0
        
        result = feather_distance_map_resize_3d(dmat, bbox, wd)
        
        # 0^2 should still be 0
        assert np.allclose(result, 0.0)
    
    def test_bbox_offset(self):
        """Test with non-origin bounding box."""
        dmat = np.ones((50, 50))
        bbox = (10, 10, 5, 59, 59, 5)  # 50x50x1, but with offset
        wd = 1.0
        
        result = feather_distance_map_resize_3d(dmat, bbox, wd)
        
        # Size should be bbox[3:6] - bbox[0:3] + 1
        assert result.shape == (50, 50, 1)
    
    def test_non_square_2d(self):
        """Test with non-square 2D input."""
        dmat = np.ones((30, 60))
        bbox = (1, 1, 1, 60, 120, 1)
        wd = 1.0
        
        result = feather_distance_map_resize_3d(dmat, bbox, wd)
        
        assert result.shape == (60, 120, 1)
    
    def test_non_cubic_3d(self):
        """Test with non-cubic 3D input."""
        dmat = np.ones((30, 40, 20))
        bbox = (1, 1, 1, 60, 80, 40)
        wd = 1.0
        
        result = feather_distance_map_resize_3d(dmat, bbox, wd)
        
        assert result.shape == (60, 80, 40)
    
    def test_high_power_weighting(self):
        """Test with high power weighting."""
        dmat = np.full((50, 50), 0.9)
        bbox = (1, 1, 1, 50, 50, 1)
        wd = 10.0
        
        result = feather_distance_map_resize_3d(dmat, bbox, wd)
        
        # 0.9^10 should be much smaller
        expected = 0.9**10.0
        assert np.allclose(result, expected, rtol=1e-5)
    
    def test_fractional_power(self):
        """Test with fractional power."""
        dmat = np.full((50, 50), 0.25)
        bbox = (1, 1, 1, 50, 50, 1)
        wd = 0.5  # Square root
        
        result = feather_distance_map_resize_3d(dmat, bbox, wd)
        
        # 0.25^0.5 = 0.5
        assert np.allclose(result, 0.5)
    
    def test_small_bbox(self):
        """Test with small bounding box."""
        dmat = np.ones((10, 10))
        bbox = (1, 1, 1, 5, 5, 1)
        wd = 1.0
        
        result = feather_distance_map_resize_3d(dmat, bbox, wd)
        
        assert result.shape == (5, 5, 1)
    
    def test_large_bbox(self):
        """Test with large bounding box."""
        dmat = np.ones((10, 10))
        bbox = (1, 1, 1, 200, 200, 1)
        wd = 1.0
        
        result = feather_distance_map_resize_3d(dmat, bbox, wd)
        
        assert result.shape == (200, 200, 1)
