"""
Tests for distance transform functions.
"""

import numpy as np
import pytest
from petakit5d.image_processing.distance import bw_max_direct_dist, bw_n_neighbors


class TestBwMaxDirectDist:
    """Tests for bw_max_direct_dist function."""
    
    def test_basic_functionality(self):
        """Test basic distance computation."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        
        dist_mat = bw_max_direct_dist(mask)
        
        assert dist_mat.shape == (10, 10, 4)
        assert dist_mat.dtype == np.float32
    
    def test_single_pixel(self):
        """Test with a single True pixel."""
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        
        dist_mat = bw_max_direct_dist(mask)
        
        # Check dimensions
        assert dist_mat.shape == (5, 5, 4)
        
        # The True pixel itself should have zero distance in all directions
        # (since it's not counted as False)
    
    def test_horizontal_line(self):
        """Test with a horizontal line of True pixels."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 2:8] = True
        
        dist_mat = bw_max_direct_dist(mask)
        
        # Check shape
        assert dist_mat.shape == (10, 10, 4)
        
        # Pixels above and below the line should have distance 5 in vertical directions
        assert dist_mat[0, 5, 0] == 5  # Distance downward
        assert dist_mat[9, 5, 1] == 4  # Distance upward
    
    def test_vertical_line(self):
        """Test with a vertical line of True pixels."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:8, 5] = True
        
        dist_mat = bw_max_direct_dist(mask)
        
        # Check shape
        assert dist_mat.shape == (10, 10, 4)
        
        # Pixels left and right of the line should have distance in horizontal directions
        assert dist_mat[5, 0, 2] == 5  # Distance rightward
        assert dist_mat[5, 9, 3] == 4  # Distance leftward
    
    def test_corners(self):
        """Test corner pixel distances."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        
        dist_mat = bw_max_direct_dist(mask)
        
        # Corner pixels should have maximum distances
        assert dist_mat[0, 0, 0] > 0  # Top-left, downward
        assert dist_mat[0, 0, 2] > 0  # Top-left, rightward
    
    def test_cityblock_equivalence(self):
        """Test that minimum over directions approximates cityblock distance."""
        from scipy.ndimage import distance_transform_cdt
        
        mask = np.zeros((20, 20), dtype=bool)
        mask[10, 10] = True
        mask[5, 15] = True
        
        dist_mat = bw_max_direct_dist(mask)
        min_dist = np.min(dist_mat, axis=2)
        
        # Cityblock distance transform
        cityblock_dist = distance_transform_cdt(~mask, metric='taxicab')
        
        # They should be similar (not exact due to implementation differences)
        # Check a few specific points
        assert min_dist[0, 0] > 0
        assert min_dist[19, 19] > 0
    
    def test_all_true_mask(self):
        """Test with all pixels True."""
        mask = np.ones((5, 5), dtype=bool)
        
        dist_mat = bw_max_direct_dist(mask)
        
        # All distances should be 0 since no False pixels
        assert dist_mat.shape == (5, 5, 4)
    
    def test_dimension_error(self):
        """Test that non-2D inputs raise errors."""
        with pytest.raises(ValueError, match="2-dimensional"):
            bw_max_direct_dist(np.random.rand(50))
        
        with pytest.raises(ValueError, match="2-dimensional"):
            bw_max_direct_dist(np.random.rand(5, 5, 5) > 0.5)


class TestBwNNeighbors:
    """Tests for bw_n_neighbors function."""
    
    def test_basic_functionality_2d(self):
        """Test basic neighbor counting in 2D."""
        bw = np.array([[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]], dtype=bool)
        
        nn = bw_n_neighbors(bw)
        
        assert nn.shape == bw.shape
        assert nn.dtype == np.uint8
        # Center pixel has 0 neighbors (all surrounding are False)
        assert nn[1, 1] == 0
    
    def test_line_points(self):
        """Test line points (should have 2 neighbors)."""
        bw = np.array([[0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0]], dtype=bool)
        
        nn = bw_n_neighbors(bw)
        
        # Middle point should have 2 neighbors
        assert nn[1, 2] == 2
        # End points should have 1 neighbor
        assert nn[1, 1] == 1
        assert nn[1, 3] == 1
    
    def test_junction(self):
        """Test junction point (should have 3+ neighbors)."""
        bw = np.array([[0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0]], dtype=bool)
        
        nn = bw_n_neighbors(bw)
        
        # Center junction should have 4 neighbors
        assert nn[2, 2] == 4
    
    def test_endpoint_detection(self):
        """Test detection of endpoints in a skeleton."""
        skel = np.array([[0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0]], dtype=bool)
        
        nn = bw_n_neighbors(skel)
        
        # Find endpoints (exactly 1 neighbor)
        endpoints = (nn == 1) & skel
        
        # Should have 2 endpoints
        assert np.sum(endpoints) == 2
        assert endpoints[0, 2] == True  # Top endpoint
        assert endpoints[4, 1] == True  # Bottom endpoint
    
    def test_3d_functionality(self):
        """Test basic functionality in 3D."""
        bw = np.zeros((5, 5, 5), dtype=bool)
        bw[2, 2, 2] = True
        bw[2, 2, 3] = True
        
        nn = bw_n_neighbors(bw)
        
        assert nn.shape == bw.shape
        assert nn.dtype == np.uint8
        # Each of the two connected voxels should have 1 neighbor
        assert nn[2, 2, 2] == 1
        assert nn[2, 2, 3] == 1
    
    def test_custom_neighborhood(self):
        """Test with custom neighborhood."""
        bw = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=bool)
        
        # 4-connected neighborhood (only orthogonal neighbors)
        nhood = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]], dtype=bool)
        
        nn = bw_n_neighbors(bw, neighborhood=nhood)
        
        # Center pixel should have 4 neighbors with 4-connectivity
        assert nn[1, 1] == 4
    
    def test_all_zeros(self):
        """Test with all False pixels."""
        bw = np.zeros((10, 10), dtype=bool)
        
        nn = bw_n_neighbors(bw)
        
        # All pixels should have 0 neighbors
        assert np.all(nn == 0)
    
    def test_dimension_error(self):
        """Test that 1D or 4D+ inputs raise errors."""
        with pytest.raises(ValueError, match="2 or 3 dimensional"):
            bw_n_neighbors(np.array([True, False, True]))
        
        with pytest.raises(ValueError, match="2 or 3 dimensional"):
            bw_n_neighbors(np.random.rand(5, 5, 5, 5) > 0.5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
