"""
Tests for binary thinning functions.
"""

import numpy as np
import pytest
from petakit5d.image_processing.morphology_thin import bw_thin


class TestBwThin:
    """Tests for bw_thin function."""
    
    def test_2d_simple(self):
        """Test basic 2D thinning."""
        # Create a simple 2D binary image
        img = np.zeros((10, 10), dtype=bool)
        img[3:7, 3:7] = True
        
        result = bw_thin(img)
        
        assert result.shape == img.shape
        assert result.dtype == bool
        # After thinning, should have fewer True pixels
        assert np.sum(result) <= np.sum(img)
    
    def test_2d_line(self):
        """Test 2D thinning on a thick line."""
        img = np.zeros((20, 20), dtype=bool)
        img[8:12, 5:15] = True  # Thick horizontal line
        
        result = bw_thin(img)
        
        assert result.shape == img.shape
        # Thinned line should be thinner
        assert np.sum(result) < np.sum(img)
    
    def test_3d_simple(self):
        """Test basic 3D thinning."""
        img = np.zeros((10, 10, 10), dtype=bool)
        img[3:7, 3:7, 3:7] = True
        
        result = bw_thin(img)
        
        assert result.shape == img.shape
        assert result.dtype == bool
        assert np.sum(result) <= np.sum(img)
    
    def test_empty_image(self):
        """Test with all-zero image."""
        img = np.zeros((10, 10), dtype=bool)
        
        result = bw_thin(img)
        
        assert np.all(result == False)
    
    def test_single_pixel(self):
        """Test with single pixel."""
        img = np.zeros((10, 10), dtype=bool)
        img[5, 5] = True
        
        result = bw_thin(img)
        
        # Single pixel should remain
        assert np.sum(result) == 1
        assert result[5, 5] == True
    
    def test_uint8_input(self):
        """Test with uint8 input."""
        img = np.zeros((10, 10), dtype=np.uint8)
        img[3:7, 3:7] = 1
        
        result = bw_thin(img)
        
        assert result.dtype == bool
        assert result.shape == img.shape
    
    def test_invalid_dimensions(self):
        """Test error handling for invalid dimensions."""
        img_1d = np.zeros(10, dtype=bool)
        img_4d = np.zeros((5, 5, 5, 5), dtype=bool)
        
        with pytest.raises(ValueError, match="must be 2D or 3D"):
            bw_thin(img_1d)
        
        with pytest.raises(ValueError, match="must be 2D or 3D"):
            bw_thin(img_4d)
    
    def test_preserves_structure(self):
        """Test that basic structure is preserved."""
        # Create a square
        img = np.zeros((15, 15), dtype=bool)
        img[5:10, 5:10] = True
        
        result = bw_thin(img)
        
        # Result should still have some pixels in the original region
        assert np.any(result[5:10, 5:10])
    
    def test_3d_cube(self):
        """Test 3D thinning on a cube."""
        img = np.zeros((15, 15, 15), dtype=bool)
        img[5:10, 5:10, 5:10] = True
        
        result = bw_thin(img)
        
        assert result.shape == img.shape
        assert np.sum(result) < np.sum(img)
        # Some structure should remain
        assert np.sum(result) > 0
