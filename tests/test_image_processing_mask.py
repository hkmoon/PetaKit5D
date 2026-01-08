"""
Unit tests for image processing mask utilities.
"""

import pytest
import numpy as np
from petakit5d.image_processing.mask import mask_vectors


class TestMaskVectors:
    """Test cases for mask_vectors function."""
    
    def test_all_inside(self):
        """Test with all vectors inside the mask."""
        x_coords = np.array([5, 10, 15])
        y_coords = np.array([5, 10, 15])
        mask = np.ones((20, 20), dtype=bool)
        
        result = mask_vectors(x_coords, y_coords, mask)
        
        assert np.all(result == True)
    
    def test_all_outside(self):
        """Test with all vectors outside the mask."""
        x_coords = np.array([5, 10, 15])
        y_coords = np.array([5, 10, 15])
        mask = np.zeros((20, 20), dtype=bool)
        
        result = mask_vectors(x_coords, y_coords, mask)
        
        assert np.all(result == False)
    
    def test_mixed(self):
        """Test with some vectors inside and some outside."""
        x_coords = np.array([5, 10, 25])
        y_coords = np.array([5, 10, 25])
        mask = np.zeros((20, 20), dtype=bool)
        mask[0:15, 0:15] = True
        
        result = mask_vectors(x_coords, y_coords, mask)
        
        # First two should be inside, third outside (25 > 20)
        assert result[0] == True
        assert result[1] == True
        assert result[2] == False
    
    def test_boundary(self):
        """Test vectors at boundaries."""
        x_coords = np.array([1, 20, 0, 21])
        y_coords = np.array([1, 20, 1, 1])
        mask = np.ones((20, 20), dtype=bool)
        
        result = mask_vectors(x_coords, y_coords, mask)
        
        # MATLAB uses 1-based indexing, so 1 and 20 are valid, 0 and 21 are not
        assert result[0] == True   # (1, 1) valid
        assert result[1] == True   # (20, 20) valid
        assert result[2] == False  # (0, 1) invalid
        assert result[3] == False  # (21, 1) invalid
    
    def test_rounding(self):
        """Test that coordinates are properly rounded."""
        x_coords = np.array([5.4, 5.6])
        y_coords = np.array([5.4, 5.6])
        mask = np.zeros((20, 20), dtype=bool)
        mask[4:7, 4:7] = True  # Region around (5, 5)
        
        result = mask_vectors(x_coords, y_coords, mask)
        
        # Both should round to (5, 5) or (6, 6), which are inside
        assert result[0] == True
        assert result[1] == True
