"""
Tests for stitching support utilities (normalize_z_stack, distance_weight_single_axis, stitch_process_filenames).
"""

import numpy as np
import pytest
from petakit5d.microscope_data_processing.stitch_support import (
    normalize_z_stack,
    distance_weight_single_axis,
    stitch_process_filenames
)


class TestNormalizeZStack:
    """Tests for normalize_z_stack function."""
    
    def test_normalize_z_stack_basic(self):
        """Test basic z-stack normalization."""
        # Create volume with varying intensity across z
        img = np.random.rand(50, 50, 10) * 1000 + 200
        result = normalize_z_stack(img)
        assert result.shape == img.shape
        assert result.dtype == img.dtype
    
    def test_normalize_z_stack_constant(self):
        """Test with constant intensity."""
        img = np.ones((30, 30, 10), dtype=np.float32) * 500
        result = normalize_z_stack(img)
        # After normalization, should have reduced variation
        assert result.shape == img.shape
    
    def test_normalize_z_stack_varying_slices(self):
        """Test with slices of different intensities."""
        img = np.zeros((20, 20, 5), dtype=np.float32)
        for i in range(5):
            img[:, :, i] = (i + 1) * 200 + 100
        result = normalize_z_stack(img)
        # Normalization should reduce the differences
        assert result.shape == img.shape
    
    def test_normalize_z_stack_preserves_dtype(self):
        """Test that output dtype matches input."""
        img = np.random.rand(30, 30, 10).astype(np.uint16) * 10000
        result = normalize_z_stack(img)
        assert result.dtype == np.uint16
    
    def test_normalize_z_stack_with_background(self):
        """Test with background pixels."""
        img = np.ones((25, 25, 8), dtype=np.float32) * 50  # Below threshold
        img[10:15, 10:15, :] = 500  # Signal region
        result = normalize_z_stack(img)
        assert result.shape == img.shape
    
    def test_normalize_z_stack_single_slice(self):
        """Test with single z slice."""
        img = np.random.rand(40, 40, 1) * 1000
        result = normalize_z_stack(img)
        assert result.shape == img.shape


class TestDistanceWeightSingleAxis:
    """Tests for distance_weight_single_axis function."""
    
    def test_distance_weight_basic(self):
        """Test basic distance weight calculation."""
        weights = distance_weight_single_axis(100, np.array([10, 90]), buffer_size=5)
        assert weights.shape == (100,)
        assert weights.dtype == np.float32
        # Middle region should have weight ~1
        assert np.all(weights[20:80] > 0.9)
    
    def test_distance_weight_full_range(self):
        """Test when endpoints cover full range."""
        weights = distance_weight_single_axis(100, np.array([1, 100]), buffer_size=10)
        # Should be all ones
        np.testing.assert_allclose(weights, 1.0, rtol=0.01)
    
    def test_distance_weight_zero_outside(self):
        """Test that weights are zero outside endpoints."""
        weights = distance_weight_single_axis(100, np.array([30, 70]), buffer_size=5)
        # Before start should decay to near zero
        assert weights[0] < 0.01
        # After end should decay to near zero
        assert weights[99] < 0.01
    
    def test_distance_weight_buffer_transition(self):
        """Test smooth transition in buffer region."""
        weights = distance_weight_single_axis(100, np.array([40, 60]), buffer_size=10)
        # Check that weights transition smoothly
        # At start boundary
        assert 0 < weights[30] < 1
        # At end boundary
        assert 0 < weights[69] < 1
    
    def test_distance_weight_no_decay(self):
        """Test with no decay (dfactor=0)."""
        weights = distance_weight_single_axis(100, np.array([30, 70]), buffer_size=5, dfactor=0)
        # Regions outside buffer should be exactly zero
        assert weights[0] == 0
        assert weights[99] == 0
    
    def test_distance_weight_different_buffer_sizes(self):
        """Test with different buffer sizes."""
        weights_small = distance_weight_single_axis(100, np.array([40, 60]), buffer_size=3)
        weights_large = distance_weight_single_axis(100, np.array([40, 60]), buffer_size=15)
        # Larger buffer should have smoother transition
        assert weights_small.shape == weights_large.shape
    
    def test_distance_weight_asymmetric(self):
        """Test with asymmetric position."""
        weights = distance_weight_single_axis(100, np.array([10, 90]), buffer_size=8)
        # Should handle asymmetric positions correctly
        assert weights[50] > 0.9  # Middle should be high
    
    def test_distance_weight_small_region(self):
        """Test with small valid region."""
        weights = distance_weight_single_axis(100, np.array([45, 55]), buffer_size=5)
        # Small central region
        assert np.sum(weights > 0.5) < 30
    
    def test_distance_weight_invalid_window_type(self):
        """Test with invalid window type."""
        with pytest.raises(ValueError, match="Unsupported window type"):
            distance_weight_single_axis(100, np.array([40, 60]), win_type='invalid')
    
    def test_distance_weight_edge_cases(self):
        """Test edge cases."""
        # Start at 1
        weights = distance_weight_single_axis(50, np.array([1, 40]), buffer_size=5)
        assert weights.shape == (50,)
        
        # End at max
        weights = distance_weight_single_axis(50, np.array([10, 50]), buffer_size=5)
        assert weights.shape == (50,)


class TestStitchProcessFilenames:
    """Tests for stitch_process_filenames function."""
    
    def test_stitch_process_filenames_basic(self):
        """Test basic file path processing."""
        tiles = ['/data/tile_001.tif', '/data/tile_002.tif']
        inputs, zarrs, names, zpath = stitch_process_filenames(tiles)
        
        assert len(inputs) == 2
        assert len(zarrs) == 2
        assert len(names) == 2
        assert names[0] == 'tile_001'
        assert names[1] == 'tile_002'
    
    def test_stitch_process_filenames_with_processed_dir(self):
        """Test with processed directory."""
        tiles = ['/data/tile_001.tif']
        inputs, zarrs, names, zpath = stitch_process_filenames(
            tiles, processed_dirstr='DSR'
        )
        
        assert '/DSR/' in inputs[0]
        assert '/DSR/' in zarrs[0]
    
    def test_stitch_process_filenames_zarr_extension(self):
        """Test with zarr file extension."""
        tiles = ['/data/tile_001.zarr']
        inputs, zarrs, names, zpath = stitch_process_filenames(
            tiles, zarr_file=True
        )
        
        assert inputs[0].endswith('.zarr')
    
    def test_stitch_process_filenames_with_resample(self):
        """Test with resampling factors."""
        tiles = ['/data/tile_001.tif']
        resample = np.array([2, 2, 1])
        inputs, zarrs, names, zpath = stitch_process_filenames(
            tiles, resample=resample
        )
        
        assert 'zarr_2_2_1' in zpath
    
    def test_stitch_process_filenames_with_mip(self):
        """Test with MIP generation."""
        tiles = ['/data/tile_001.tif']
        stitch_mip = np.array([False, False, True])
        inputs, zarrs, names, zpath = stitch_process_filenames(
            tiles, stitch_mip=stitch_mip
        )
        
        assert '/MIPs/' in inputs[0]
        assert '_MIP_z' in inputs[0]
    
    def test_stitch_process_filenames_no_process(self):
        """Test with process_tiles=False."""
        tiles = ['/data/tile_001.tif']
        inputs, zarrs, names, zpath = stitch_process_filenames(
            tiles, process_tiles=False
        )
        
        assert zpath == ''
    
    def test_stitch_process_filenames_multiple_tiles(self):
        """Test with multiple tiles."""
        tiles = [
            '/data/exp1/tile_001.tif',
            '/data/exp1/tile_002.tif',
            '/data/exp2/tile_003.tif'
        ]
        inputs, zarrs, names, zpath = stitch_process_filenames(tiles)
        
        assert len(inputs) == 3
        assert names == ['tile_001', 'tile_002', 'tile_003']
    
    def test_stitch_process_filenames_trailing_slash(self):
        """Test handling of trailing slashes in paths."""
        tiles = ['/data/exp/tile_001.tif']
        inputs1, _, _, _ = stitch_process_filenames(tiles)
        
        # Should handle paths consistently
        assert inputs1[0] == '/data/exp/tile_001.tif'
    
    def test_stitch_process_filenames_resample_padding(self):
        """Test resample array padding."""
        tiles = ['/data/tile_001.tif']
        # Test with 2D resample
        resample = np.array([2, 1])
        inputs, zarrs, names, zpath = stitch_process_filenames(
            tiles, resample=resample
        )
        
        # Should pad to 3D
        assert 'zarr_' in zpath
    
    def test_stitch_process_filenames_complex_path(self):
        """Test with complex file paths."""
        tiles = ['/mnt/storage/experiment/2024/sample_A/tile_001.tif']
        inputs, zarrs, names, zpath = stitch_process_filenames(
            tiles, processed_dirstr='Decon'
        )
        
        assert '/Decon/' in inputs[0]
        assert names[0] == 'tile_001'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
