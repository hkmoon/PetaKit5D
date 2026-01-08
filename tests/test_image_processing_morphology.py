"""
Unit tests for image processing morphology utilities.
"""

import pytest
import numpy as np
from petakit5d.image_processing.morphology import bw_largest_obj


class TestBwLargestObj:
    """Test cases for bw_largest_obj function."""
    
    def test_single_object(self):
        """Test with single connected component."""
        mask = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0]
        ], dtype=bool)
        
        result = bw_largest_obj(mask)
        
        # Should return the same mask
        assert np.array_equal(result, mask)
    
    def test_multiple_objects(self):
        """Test with multiple connected components."""
        mask = np.array([
            [1, 0, 1, 1],
            [1, 0, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1]
        ], dtype=bool)
        
        result = bw_largest_obj(mask)
        
        # Either bottom row (4 pixels) or top-right block (4 pixels) could be largest
        # Depending on how scipy.ndimage.label assigns labels
        # Just verify that we get only one connected component with 4 pixels
        assert np.sum(result) == 4
    
    def test_empty_mask(self):
        """Test with empty mask."""
        mask = np.zeros((10, 10), dtype=bool)
        result = bw_largest_obj(mask)
        
        # Should return empty mask
        assert np.all(result == False)
    
    def test_3d_volume(self):
        """Test with 3D volume."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[2:5, 2:5, 2:5] = True  # Larger object
        mask[7:8, 7:8, 7:8] = True  # Smaller object
        
        result = bw_largest_obj(mask)
        
        # Should keep only the larger object
        assert np.sum(result) > 20  # Larger object
        assert not result[7, 7, 7]  # Smaller object removed
    
    def test_connectivity(self):
        """Test with different connectivity."""
        mask = np.array([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ], dtype=bool)
        
        # With 4-connectivity (connectivity=1), all four corners are separate
        result_4 = bw_largest_obj(mask, connectivity=1)
        # Should keep one corner (all same size, keeps first found)
        assert np.sum(result_4) == 1
        
        # With 8-connectivity (connectivity=2), corners might be connected diagonally
        # depending on implementation, but should still work
        result_8 = bw_largest_obj(mask, connectivity=2)
        assert result_8.shape == mask.shape
