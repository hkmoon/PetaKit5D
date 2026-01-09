"""
Tests for 3D neighborhood connectivity functions.
"""

import pytest
import numpy as np
from petakit5d.image_processing import bwn_hood_3d


class TestBwnHood3D:
    """Test suite for bwn_hood_3d function."""
    
    def test_6_connectivity(self):
        """Test 6-connectivity neighborhood generation."""
        hood = bwn_hood_3d(6)
        
        # Check shape
        assert hood.shape == (3, 3, 3)
        
        # Check type
        assert hood.dtype == bool
        
        # Check number of neighbors
        assert np.sum(hood) == 6
        
        # Center should be False
        assert hood[1, 1, 1] == False
        
        # Face neighbors should be True
        assert hood[1, 1, 0] == True  # Front
        assert hood[1, 1, 2] == True  # Back
        assert hood[1, 0, 1] == True  # Top
        assert hood[1, 2, 1] == True  # Bottom
        assert hood[0, 1, 1] == True  # Left
        assert hood[2, 1, 1] == True  # Right
    
    def test_18_connectivity(self):
        """Test 18-connectivity neighborhood generation."""
        hood = bwn_hood_3d(18)
        
        # Check shape
        assert hood.shape == (3, 3, 3)
        
        # Check number of neighbors
        assert np.sum(hood) == 18
        
        # Center should be False
        assert hood[1, 1, 1] == False
        
        # Face neighbors should be True
        assert hood[1, 1, 0] == True
        assert hood[1, 1, 2] == True
        
        # Edge neighbors should be True
        assert hood[0, 0, 1] == True
        assert hood[2, 2, 1] == True
    
    def test_26_connectivity(self):
        """Test 26-connectivity neighborhood generation."""
        hood = bwn_hood_3d(26)
        
        # Check shape
        assert hood.shape == (3, 3, 3)
        
        # Check number of neighbors
        assert np.sum(hood) == 26
        
        # Center should be False
        assert hood[1, 1, 1] == False
        
        # All other positions should be True
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if (i, j, k) != (1, 1, 1):
                        assert hood[i, j, k] == True
    
    def test_default_connectivity(self):
        """Test default connectivity (should be 26)."""
        hood = bwn_hood_3d()
        
        assert np.sum(hood) == 26
        assert hood[1, 1, 1] == False
    
    def test_invalid_connectivity(self):
        """Test that invalid connectivity raises error."""
        with pytest.raises(ValueError, match="Invalid connectivity"):
            bwn_hood_3d(8)
        
        with pytest.raises(ValueError, match="Invalid connectivity"):
            bwn_hood_3d(10)
        
        with pytest.raises(ValueError, match="Invalid connectivity"):
            bwn_hood_3d(27)
    
    def test_connectivity_hierarchy(self):
        """Test that connectivity forms proper hierarchy (6 ⊂ 18 ⊂ 26)."""
        hood6 = bwn_hood_3d(6)
        hood18 = bwn_hood_3d(18)
        hood26 = bwn_hood_3d(26)
        
        # 6-neighbors should be subset of 18-neighbors
        assert np.all(hood6 <= hood18)
        
        # 18-neighbors should be subset of 26-neighbors
        assert np.all(hood18 <= hood26)
        
        # Check proper subset (not equal)
        assert not np.array_equal(hood6, hood18)
        assert not np.array_equal(hood18, hood26)
    
    def test_symmetry(self):
        """Test that neighborhoods are symmetric."""
        for conn in [6, 18, 26]:
            hood = bwn_hood_3d(conn)
            
            # Check symmetry along all axes
            assert np.array_equal(hood, np.flip(hood, axis=0))
            assert np.array_equal(hood, np.flip(hood, axis=1))
            assert np.array_equal(hood, np.flip(hood, axis=2))
    
    def test_use_in_morphology(self):
        """Test that the neighborhood can be used with scipy morphology."""
        from scipy import ndimage
        
        # Create a simple 3D binary image
        img = np.zeros((5, 5, 5), dtype=bool)
        img[2, 2, 2] = True
        
        # Dilate with 6-connectivity
        hood6 = bwn_hood_3d(6)
        dilated = ndimage.binary_dilation(img, structure=hood6)
        
        # Should have 7 True values (center + 6 neighbors)
        assert np.sum(dilated) == 7
        
        # Check center and face neighbors are True
        assert dilated[2, 2, 2] == True
        assert dilated[2, 2, 1] == True
        assert dilated[2, 2, 3] == True
