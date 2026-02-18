"""
Tests for non-maximum suppression functions.
"""

import numpy as np
import pytest
from petakit5d.image_processing import non_maximum_suppression


class TestNonMaximumSuppression:
    """Test suite for non_maximum_suppression function."""
    
    def test_basic_functionality(self):
        """Test basic NMS operation."""
        # Create a simple response with a peak
        response = np.zeros((10, 10))
        response[5, 5] = 10
        response[5, 4] = 8
        response[5, 6] = 8
        
        # Create orientation pointing horizontally
        orientation = np.zeros((10, 10))
        
        result = non_maximum_suppression(response, orientation)
        
        # Peak should be preserved
        assert result[5, 5] > 0
        assert result.shape == response.shape
    
    def test_with_random_data(self):
        """Test with random data."""
        response = np.random.rand(50, 50) * 100
        orientation = np.random.rand(50, 50) * 2 * np.pi
        
        result = non_maximum_suppression(response, orientation)
        
        # Result should have same shape
        assert result.shape == response.shape
        # Result should have suppressed some values
        assert np.sum(result == 0) >= np.sum(response == 0)
    
    def test_uniform_response(self):
        """Test with uniform response."""
        response = np.ones((20, 20)) * 5
        orientation = np.zeros((20, 20))
        
        result = non_maximum_suppression(response, orientation)
        
        # Uniform values won't have clear maxima, but shouldn't be all zeros
        # The edges will preserve some values due to interpolation
        assert result.shape == response.shape
    
    def test_diagonal_orientation(self):
        """Test with diagonal orientations."""
        response = np.random.rand(30, 30)
        orientation = np.ones((30, 30)) * np.pi / 4  # 45 degrees
        
        result = non_maximum_suppression(response, orientation)
        
        assert result.shape == response.shape
        assert np.all(result >= 0)
    
    def test_edge_handling(self):
        """Test that edges are handled properly."""
        response = np.random.rand(15, 15)
        orientation = np.random.rand(15, 15) * 2 * np.pi
        
        result = non_maximum_suppression(response, orientation)
        
        # Check edges are handled (not NaN or inf)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
