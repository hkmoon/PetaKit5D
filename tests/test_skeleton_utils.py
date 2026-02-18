"""
Tests for skeleton_utils module.
"""

import numpy as np
import pytest
from petakit5d.image_processing import skeleton


class TestSkeleton:
    """Tests for skeleton function."""
    
    def test_skeleton_2d_thinning(self):
        """Test 2D skeletonization using thinning method."""
        # Create a simple rectangle
        bw = np.zeros((50, 50), dtype=bool)
        bw[10:40, 20:30] = True
        
        skel = skeleton(bw, method='t')
        
        # Check output is boolean
        assert skel.dtype == bool
        
        # Check output shape matches input
        assert skel.shape == bw.shape
        
        # Skeleton should have fewer points than original
        assert np.sum(skel) < np.sum(bw)
        
        # Skeleton should be contained within original
        assert np.all(skel <= bw)
    
    def test_skeleton_3d_thinning(self):
        """Test 3D skeletonization using thinning method."""
        # Create a simple 3D object (sphere)
        z, y, x = np.ogrid[-10:11, -10:11, -10:11]
        bw = (x**2 + y**2 + z**2 <= 64)  # Sphere of radius 8
        
        skel = skeleton(bw, method='t')
        
        # Check output is boolean
        assert skel.dtype == bool
        
        # Check output shape matches input
        assert skel.shape == bw.shape
        
        # Skeleton should have fewer points than original
        assert np.sum(skel) < np.sum(bw)
    
    def test_skeleton_2d_erosion(self):
        """Test 2D skeletonization using erosion & opening method."""
        # Create a simple rectangle
        bw = np.zeros((50, 50), dtype=bool)
        bw[15:35, 20:30] = True
        
        skel = skeleton(bw, method='e')
        
        # Check output is boolean
        assert skel.dtype == bool
        
        # Check output shape matches input
        assert skel.shape == bw.shape
        
        # Skeleton should have fewer points than original
        assert np.sum(skel) < np.sum(bw)
    
    def test_skeleton_3d_erosion(self):
        """Test 3D skeletonization using erosion & opening method."""
        # Create a simple 3D object
        bw = np.zeros((30, 30, 30), dtype=bool)
        bw[10:20, 10:20, 10:20] = True
        
        skel = skeleton(bw, method='e')
        
        # Check output is boolean
        assert skel.dtype == bool
        
        # Check output shape matches input
        assert skel.shape == bw.shape
        
        # Skeleton should have fewer points than original
        assert np.sum(skel) < np.sum(bw)
    
    def test_skeleton_2d_divergence(self):
        """Test 2D skeletonization using divergence method."""
        # Create a simple rectangle
        bw = np.zeros((50, 50), dtype=bool)
        bw[10:40, 20:30] = True
        
        skel = skeleton(bw, method='d', div_threshold=-1.0)
        
        # Check output is boolean
        assert skel.dtype == bool
        
        # Check output shape matches input
        assert skel.shape == bw.shape
        
        # With divergence method, result depends on threshold
        # Just check it runs without error
        assert skel.shape == bw.shape
    
    def test_skeleton_3d_divergence(self):
        """Test 3D skeletonization using divergence method."""
        # Create a simple 3D object
        bw = np.zeros((30, 30, 30), dtype=bool)
        bw[10:20, 10:20, 10:20] = True
        
        skel = skeleton(bw, method='d', div_threshold=-1.5)
        
        # Check output is boolean
        assert skel.dtype == bool
        
        # Check output shape matches input
        assert skel.shape == bw.shape
    
    def test_skeleton_divergence_threshold_warning(self):
        """Test that non-negative threshold raises warning."""
        bw = np.zeros((30, 30), dtype=bool)
        bw[10:20, 10:20] = True
        
        with pytest.warns(UserWarning, match="Non-negative threshold"):
            skel = skeleton(bw, method='d', div_threshold=0.5)
    
    def test_skeleton_invalid_dimensions(self):
        """Test that 1D or 4D+ inputs raise error."""
        # 1D input
        bw_1d = np.ones(50, dtype=bool)
        with pytest.raises(ValueError, match="2D or 3D matrix"):
            skeleton(bw_1d)
        
        # 4D input
        bw_4d = np.ones((10, 10, 10, 10), dtype=bool)
        with pytest.raises(ValueError, match="2D or 3D matrix"):
            skeleton(bw_4d)
    
    def test_skeleton_invalid_method(self):
        """Test that invalid method raises error."""
        bw = np.ones((30, 30), dtype=bool)
        
        with pytest.raises(ValueError, match="not a recognized method"):
            skeleton(bw, method='invalid')
    
    def test_skeleton_empty_image(self):
        """Test skeletonization of empty image."""
        bw = np.zeros((30, 30), dtype=bool)
        
        skel = skeleton(bw, method='t')
        
        # Empty image should produce empty skeleton
        assert not np.any(skel)
    
    def test_skeleton_full_image(self):
        """Test skeletonization of full image."""
        bw = np.ones((20, 20), dtype=bool)
        
        skel = skeleton(bw, method='t')
        
        # Check output is boolean
        assert skel.dtype == bool
        
        # Full image should produce some skeleton
        assert np.any(skel)
    
    def test_skeleton_preserves_single_pixel(self):
        """Test that single pixel is preserved."""
        bw = np.zeros((20, 20), dtype=bool)
        bw[10, 10] = True
        
        skel = skeleton(bw, method='t')
        
        # Single pixel should be preserved
        assert np.sum(skel) == 1
        assert skel[10, 10]
    
    def test_skeleton_line_2d(self):
        """Test skeletonization of 2D line."""
        # Create a thick horizontal line
        bw = np.zeros((30, 50), dtype=bool)
        bw[10:20, 5:45] = True
        
        skel = skeleton(bw, method='t')
        
        # Skeleton of thick line should be thin
        assert np.sum(skel) < np.sum(bw)
        
        # Should roughly follow the centerline
        assert skel.dtype == bool
    
    def test_skeleton_cross_pattern(self):
        """Test skeletonization of cross pattern."""
        bw = np.zeros((50, 50), dtype=bool)
        bw[20:30, 10:40] = True  # Horizontal bar
        bw[10:40, 20:30] = True  # Vertical bar
        
        skel = skeleton(bw, method='t')
        
        # Check output properties
        assert skel.dtype == bool
        assert np.sum(skel) < np.sum(bw)
        
        # Skeleton should preserve the cross structure
        assert np.any(skel)
    
    def test_skeleton_numeric_input_converted_to_bool(self):
        """Test that numeric input is converted to boolean."""
        # Numeric input
        bw = np.zeros((30, 30), dtype=np.uint8)
        bw[10:20, 10:20] = 255
        
        skel = skeleton(bw, method='t')
        
        # Output should be boolean
        assert skel.dtype == bool
    
    def test_skeleton_divergence_different_thresholds(self):
        """Test divergence method with different thresholds."""
        bw = np.zeros((50, 50), dtype=bool)
        bw[15:35, 20:30] = True
        
        # More negative threshold should give fewer points
        skel_strict = skeleton(bw, method='d', div_threshold=-3.0)
        skel_loose = skeleton(bw, method='d', div_threshold=-0.5)
        
        # Check that stricter threshold gives fewer or equal points
        assert np.sum(skel_strict) <= np.sum(skel_loose)
    
    def test_skeleton_circle(self):
        """Test skeletonization of circle."""
        # Create a filled circle
        y, x = np.ogrid[-25:26, -25:26]
        bw = (x**2 + y**2 <= 400)  # Circle of radius 20
        
        skel = skeleton(bw, method='t')
        
        # Check basic properties
        assert skel.dtype == bool
        assert np.sum(skel) < np.sum(bw)
        
        # Skeleton of circle should be close to center
        center_region = skel[20:32, 20:32]
        assert np.any(center_region)
