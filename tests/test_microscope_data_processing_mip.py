"""
Tests for MIP (Maximum Intensity Projection) and pooling functions.
"""

import numpy as np
import pytest
from petakit5d.microscope_data_processing import (
    max_pooling_3d, min_bbox_3d, project_3d_to_2d
)


class TestMaxPooling3D:
    """Test suite for max_pooling_3d function."""
    
    def test_basic_functionality(self):
        """Test basic max pooling."""
        vol = np.random.rand(20, 30, 10)
        pooled = max_pooling_3d(vol, (2, 2, 2))
        
        assert pooled.shape == (10, 15, 5)
    
    def test_uneven_dimensions(self):
        """Test with dimensions not divisible by pool size."""
        vol = np.random.rand(21, 31, 11)
        pooled = max_pooling_3d(vol, (2, 2, 2))
        
        # Should pad and then pool
        assert pooled.shape == (11, 16, 6)
    
    def test_pool_size_one(self):
        """Test with pool size of 1 (no pooling)."""
        vol = np.random.rand(10, 10, 10)
        pooled = max_pooling_3d(vol, (1, 1, 1))
        
        assert np.array_equal(pooled, vol)
    
    def test_max_values_preserved(self):
        """Test that maximum values are preserved."""
        vol = np.zeros((4, 4, 4))
        vol[0, 0, 0] = 100  # Max in first pool block
        vol[2, 2, 2] = 200  # Max in another pool block
        
        pooled = max_pooling_3d(vol, (2, 2, 2))
        
        # Should have preserved the max value of 200
        assert np.max(pooled) == 200
    
    def test_large_pool_size(self):
        """Test with large pool size."""
        vol = np.random.rand(20, 20, 20)
        pooled = max_pooling_3d(vol, (10, 10, 10))
        
        assert pooled.shape == (2, 2, 2)


class TestMinBbox3D:
    """Test suite for min_bbox_3d function."""
    
    def test_basic_functionality(self):
        """Test basic min bbox computation."""
        vol = np.random.rand(50, 50, 30)
        bbox = (10, 20, 5, 30, 40, 20)  # MATLAB 1-based
        
        min_val = min_bbox_3d(vol, bbox)
        
        assert isinstance(min_val, (float, np.floating))
        assert min_val >= 0
    
    def test_with_known_min(self):
        """Test with known minimum value."""
        vol = np.ones((40, 40, 20)) * 10
        vol[15, 25, 10] = 1.5  # Set a minimum
        
        bbox = (10, 20, 5, 20, 30, 15)  # Includes the minimum
        min_val = min_bbox_3d(vol, bbox)
        
        assert min_val == 1.5
    
    def test_small_bbox(self):
        """Test with small bounding box."""
        vol = np.random.rand(100, 100, 50)
        bbox = (10, 10, 10, 15, 15, 15)
        
        min_val = min_bbox_3d(vol, bbox)
        
        assert isinstance(min_val, (float, np.floating))


class TestProject3DTo2D:
    """Test suite for project_3d_to_2d function."""
    
    def test_central_xy(self):
        """Test central XY slice extraction."""
        vol = np.random.rand(50, 40, 30)
        result = project_3d_to_2d(vol, 'central_xy')
        
        assert result.shape == (50, 40)
        assert result.ndim == 2
    
    def test_central_yz(self):
        """Test central YZ slice extraction."""
        vol = np.random.rand(50, 40, 30)
        result = project_3d_to_2d(vol, 'central_yz')
        
        assert result.shape == (50, 30)
        assert result.ndim == 2
    
    def test_central_xz(self):
        """Test central XZ slice extraction."""
        vol = np.random.rand(50, 40, 30)
        result = project_3d_to_2d(vol, 'central_xz')
        
        assert result.shape == (40, 30)
        assert result.ndim == 2
    
    def test_mip_xy(self):
        """Test MIP along XY (maximum along Z)."""
        vol = np.random.rand(30, 30, 20)
        result = project_3d_to_2d(vol, 'mip_xy')
        
        assert result.shape == (30, 30)
        # Check that it's actually the max
        assert np.all(result >= vol[:, :, 0])
    
    def test_mip_yz(self):
        """Test MIP along YZ."""
        vol = np.random.rand(30, 30, 20)
        result = project_3d_to_2d(vol, 'mip_yz')
        
        assert result.shape == (30, 20)
    
    def test_mip_xz(self):
        """Test MIP along XZ."""
        vol = np.random.rand(30, 30, 20)
        result = project_3d_to_2d(vol, 'mip_xz')
        
        assert result.shape == (30, 20)
    
    def test_mean_xy(self):
        """Test mean projection along XY."""
        vol = np.random.rand(25, 25, 15)
        result = project_3d_to_2d(vol, 'mean_xy')
        
        assert result.shape == (25, 25)
        # Mean should be between min and max
        assert np.all(result >= np.min(vol))
        assert np.all(result <= np.max(vol))
    
    def test_mean_with_nan(self):
        """Test mean projection handles NaN."""
        vol = np.random.rand(20, 20, 10)
        vol[10, 10, 5] = np.nan
        
        result = project_3d_to_2d(vol, 'mean_xy')
        
        # Should handle NaN appropriately
        assert result.shape == (20, 20)
    
    def test_case_insensitive(self):
        """Test that method is case-insensitive."""
        vol = np.random.rand(20, 20, 15)
        
        result1 = project_3d_to_2d(vol, 'MIP_XY')
        result2 = project_3d_to_2d(vol, 'mip_xy')
        
        assert np.array_equal(result1, result2)
    
    def test_unknown_method(self):
        """Test with unknown method returns original."""
        vol = np.random.rand(20, 20, 15)
        result = project_3d_to_2d(vol, 'unknown_method')
        
        # Should return original volume
        assert result.shape == vol.shape
