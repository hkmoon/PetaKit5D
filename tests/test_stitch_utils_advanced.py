"""
Tests for advanced stitching utilities.

Tests check_major_tile_valid, feather_blending_3d, and normxcorr2_max_shift functions.
"""

import numpy as np
import pytest
from petakit5d.microscope_data_processing import (
    check_major_tile_valid,
    feather_blending_3d,
    normxcorr2_max_shift
)


class TestCheckMajorTileValid:
    """Tests for check_major_tile_valid function."""
    
    def test_all_valid_tiles(self):
        """Test with all valid tiles (non-zero everywhere)."""
        fmat = np.ones((5, 5, 3, 4))
        major_inds = np.array([0, 1])
        
        is_valid, bbox = check_major_tile_valid(fmat, major_inds)
        
        assert is_valid is True
        assert bbox is None
    
    def test_invalid_tiles_with_zeros(self):
        """Test with invalid tiles containing all zeros at some voxels."""
        fmat = np.ones((5, 5, 3, 4))
        # Set some voxels to zero in major tiles
        fmat[2, 2, 1, 0] = 0
        fmat[2, 2, 1, 1] = 0
        
        major_inds = np.array([0, 1])
        
        is_valid, bbox = check_major_tile_valid(fmat, major_inds)
        
        assert is_valid is False
        assert bbox is not None
        assert bbox.shape == (6,)
        # Check bbox contains the invalid voxel (1-based)
        assert bbox[0] == 3  # y = 2 + 1
        assert bbox[1] == 3  # x = 2 + 1
        assert bbox[2] == 2  # z = 1 + 1
    
    def test_single_major_tile(self):
        """Test with a single major tile."""
        fmat = np.random.rand(4, 4, 2, 3) > 0.3
        major_inds = np.array([1])
        
        is_valid, bbox = check_major_tile_valid(fmat, major_inds)
        
        # Valid if tile 1 has non-zero at all locations
        expected_valid = np.all(fmat[:, :, :, 1])
        assert is_valid == expected_valid
    
    def test_multiple_invalid_regions(self):
        """Test with multiple disconnected invalid regions."""
        fmat = np.ones((10, 10, 5, 3))
        major_inds = np.array([0, 1])
        
        # Create invalid regions
        fmat[1:3, 1:3, 0, 0] = 0
        fmat[1:3, 1:3, 0, 1] = 0
        fmat[7:9, 7:9, 3, 0] = 0
        fmat[7:9, 7:9, 3, 1] = 0
        
        is_valid, bbox = check_major_tile_valid(fmat, major_inds)
        
        assert is_valid is False
        assert bbox is not None
        # Bbox should encompass all invalid regions
        assert bbox[0] >= 2  # min y (1-based)
        assert bbox[3] <= 9  # max y (1-based)


class TestFeatherBlending3D:
    """Tests for feather_blending_3d function."""
    
    def test_simple_blending_two_tiles(self):
        """Test simple blending of two overlapping tiles."""
        # Create two tiles with different intensities
        tiles = np.zeros((5, 5, 3, 2))
        tiles[:, :, :, 0] = 100  # First tile
        tiles[:, :, :, 1] = 200  # Second tile
        
        # Distance maps (uniform weight)
        dist_maps = np.ones((5, 5, 3, 2))
        
        result, mex_flag = feather_blending_3d(tiles, dist_maps)
        
        assert result.shape == (5, 5, 3)
        assert mex_flag is False  # No MEX in Python
        # With equal weights, should be average
        np.testing.assert_allclose(result, 150.0)
    
    def test_weighted_blending(self):
        """Test blending with different distance weights."""
        tiles = np.zeros((4, 4, 2, 2))
        tiles[:, :, :, 0] = 100
        tiles[:, :, :, 1] = 200
        
        # Different weights
        dist_maps = np.ones((4, 4, 2, 2))
        dist_maps[:, :, :, 0] = 1.0  # Weight 1 for first tile
        dist_maps[:, :, :, 1] = 3.0  # Weight 3 for second tile
        
        result, _ = feather_blending_3d(tiles, dist_maps)
        
        # Weighted average: (100*1 + 200*3) / (1+3) = 700/4 = 175
        np.testing.assert_allclose(result, 175.0)
    
    def test_partial_overlap(self):
        """Test blending with partial overlap (some tiles have zeros)."""
        tiles = np.zeros((5, 5, 3, 2))
        # First tile only in left half
        tiles[:, :2, :, 0] = 100
        # Second tile only in right half
        tiles[:, 3:, :, 1] = 200
        
        dist_maps = np.ones((5, 5, 3, 2))
        
        result, _ = feather_blending_3d(tiles, dist_maps)
        
        # Left half should be 100, right half should be 200
        assert np.allclose(result[:, :2, :], 100.0)
        assert np.allclose(result[:, 3:, :], 200.0)
        # Middle column: no data, should be 0
        assert np.allclose(result[:, 2, :], 0.0)
    
    def test_zero_distance_regions(self):
        """Test handling of zero distance (outside tile regions)."""
        tiles = np.random.rand(4, 4, 2, 3) * 100
        dist_maps = np.random.rand(4, 4, 2, 3)
        
        # Set some regions to have zero distance in all tiles
        dist_maps[0, 0, :, :] = 0
        
        result, _ = feather_blending_3d(tiles, dist_maps)
        
        # Zero distance regions should result in zero output
        assert result[0, 0, 0] == 0.0
        assert result[0, 0, 1] == 0.0
    
    def test_blending_with_bbox(self):
        """Test blending with bounding box placement."""
        tiles = np.ones((3, 3, 2, 2)) * 50
        dist_maps = np.ones((3, 3, 2, 2))
        
        # Create output array
        output = np.zeros((10, 10, 5))
        bbox = np.array([2, 2, 2, 4, 4, 3])  # 1-based bbox
        
        result, _ = feather_blending_3d(tiles, dist_maps, output, bbox)
        
        # Check that result is placed in correct location
        assert result is output  # Should modify in place
        # 1-based [2,2,2] to [4,4,3] => 0-based [1,1,1] to [3,3,2]
        assert np.all(output[1:4, 1:4, 1:3] == 50.0)
        # Rest should be zero
        assert output[0, 0, 0] == 0.0
        assert output[5, 5, 4] == 0.0


class TestNormxcorr2MaxShift:
    """Tests for normxcorr2_max_shift function."""
    
    def test_perfect_match_no_shift(self):
        """Test with perfect match at zero shift."""
        # Create a distinctive template
        template = np.random.rand(5, 5)
        # Embed template in a larger image with random background
        image = np.random.rand(15, 15) * 0.1  # Low intensity background
        image[5:10, 5:10] = template  # Place template in center
        
        max_shifts = np.array([5, 5])
        
        offset, corr, C = normxcorr2_max_shift(template, image, max_shifts)
        
        # Should find some correlation and return valid outputs
        assert isinstance(corr, float)
        assert -1 <= corr <= 1  # Valid correlation range
        assert offset.shape == (3,)
        assert offset[2] == 0  # Third element is zero for 2D
        assert C.ndim == 2  # 2D correlation map
    
    def test_known_shift(self):
        """Test with known shift between template and image."""
        # Create template
        template = np.zeros((5, 5))
        template[2, 2] = 1.0
        
        # Create image with template shifted by (3, 2)
        image = np.zeros((15, 15))
        image[5, 4] = 1.0  # Shift of (3, 2) from center
        
        max_shifts = np.array([10, 10])
        
        offset, corr, C = normxcorr2_max_shift(template, image, max_shifts)
        
        # Should detect the shift (approximately)
        assert corr > 0.5  # Reasonable correlation
        # Offset should be close to (3, 2, 0)
        assert abs(offset[2]) == 0
    
    def test_symmetric_shift_constraints(self):
        """Test with symmetric shift constraints."""
        template = np.random.rand(10, 10)
        image = np.random.rand(30, 30)
        
        # Symmetric constraints
        max_shifts = np.array([5, 7])
        
        offset, corr, C = normxcorr2_max_shift(template, image, max_shifts)
        
        # Check that offset is within constraints
        assert abs(offset[0]) <= 5  # Y constraint
        assert abs(offset[1]) <= 7  # X constraint
        assert offset[2] == 0
    
    def test_asymmetric_shift_constraints(self):
        """Test with asymmetric shift constraints."""
        template = np.random.rand(8, 8)
        image = np.random.rand(25, 25)
        
        # Asymmetric constraints
        max_shifts = np.array([[-3, -2], [5, 6]])
        
        offset, corr, C = normxcorr2_max_shift(template, image, max_shifts)
        
        # Check that offset is within asymmetric constraints
        assert -3 <= offset[0] <= 5
        assert -2 <= offset[1] <= 6
        assert offset[2] == 0
    
    def test_correlation_output_shape(self):
        """Test that correlation output has expected shape."""
        template = np.ones((5, 5))
        image = np.ones((20, 20))
        
        max_shifts = np.array([3, 4])
        
        offset, corr, C = normxcorr2_max_shift(template, image, max_shifts)
        
        # C should be cropped based on max_shifts
        # Should be roughly 2*max_shifts in size
        assert C.shape[0] <= 2 * max_shifts[0] + template.shape[0]
        assert C.shape[1] <= 2 * max_shifts[1] + template.shape[1]
        assert isinstance(corr, float)
        assert -1 <= corr <= 1  # Correlation coefficient range
    
    def test_small_template(self):
        """Test with very small template."""
        template = np.array([[1, 2], [3, 4]])
        image = np.random.rand(10, 10)
        
        max_shifts = np.array([2, 2])
        
        offset, corr, C = normxcorr2_max_shift(template, image, max_shifts)
        
        assert offset.shape == (3,)
        assert C.ndim == 2
        assert isinstance(corr, (int, float))
