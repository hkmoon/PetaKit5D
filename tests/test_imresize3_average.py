"""
Tests for 3D resizing with averaging.
"""

import numpy as np
import pytest
from petakit5d.microscope_data_processing.resample import imresize3_average


class TestImresize3Average:
    """Tests for imresize3_average function."""
    
    def test_basic_downsampling(self):
        """Test basic downsampling by factor of 2."""
        vol = np.ones((100, 100, 50), dtype=np.float32)
        result = imresize3_average(vol, 2)
        
        # Check output shape
        assert result.shape == (50, 50, 25)
        # Check values (average of ones should be 1)
        assert np.allclose(result, 1.0)
    
    def test_integer_factor(self):
        """Test with integer resample factor."""
        vol = np.random.rand(64, 64, 64).astype(np.float32)
        result = imresize3_average(vol, 4)
        
        assert result.shape == (16, 16, 16)
        assert result.dtype == np.float32
    
    def test_tuple_factor(self):
        """Test with tuple of resample factors."""
        vol = np.random.rand(100, 100, 50).astype(np.float32)
        result = imresize3_average(vol, (2, 2, 1))
        
        # y and x downsampled by 2, z unchanged
        assert result.shape == (50, 50, 50)
    
    def test_different_factors_per_dimension(self):
        """Test with different factors for each dimension."""
        vol = np.random.rand(120, 80, 60).astype(np.float32)
        result = imresize3_average(vol, (2, 4, 3))
        
        assert result.shape == (60, 20, 20)
    
    def test_non_divisible_size(self):
        """Test when volume size is not evenly divisible by factor."""
        vol = np.ones((101, 99, 51), dtype=np.float32)
        result = imresize3_average(vol, 2)
        
        # Should pad to 102, 100, 52 then downsample to 51, 50, 26
        assert result.shape == (51, 50, 26)
        
        # Values should be close to 1 where original data existed
        # but might be less where padding occurred
        assert result.max() <= 1.0
        assert result.min() >= 0.0
    
    def test_averaging_correctness(self):
        """Test that averaging is computed correctly."""
        # Create a volume with known pattern
        vol = np.zeros((4, 4, 4), dtype=np.float32)
        # Fill first block (indices 0-1 in each dimension) with 1.0
        vol[0:2, 0:2, 0:2] = 1.0  
        
        result = imresize3_average(vol, 2)
        
        # Should be 2x2x2
        assert result.shape == (2, 2, 2)
        # The [0,0,0] block contains all 1's, so average should be 1.0
        assert result[0, 0, 0] == 1.0
        # The [1,1,1] block contains all 0's, so average should be 0.0
        assert result[1, 1, 1] == 0.0
    
    def test_constant_volume(self):
        """Test with constant-valued volume."""
        value = 42.0
        vol = np.full((64, 64, 32), value, dtype=np.float32)
        result = imresize3_average(vol, 2)
        
        # Average of constant should be constant
        assert np.allclose(result, value)
    
    def test_small_volume(self):
        """Test with small volume."""
        vol = np.random.rand(4, 4, 4).astype(np.float32)
        result = imresize3_average(vol, 2)
        
        assert result.shape == (2, 2, 2)
    
    def test_factor_of_one(self):
        """Test with factor of 1 (no change)."""
        vol = np.random.rand(32, 32, 32).astype(np.float32)
        result = imresize3_average(vol, 1)
        
        # Should be identical (just converted to float32)
        assert result.shape == vol.shape
        assert np.allclose(result, vol)
    
    def test_large_factor(self):
        """Test with large downsampling factor."""
        vol = np.random.rand(128, 128, 64).astype(np.float32)
        result = imresize3_average(vol, 8)
        
        assert result.shape == (16, 16, 8)
    
    def test_single_dimension_downsampling(self):
        """Test downsampling only in one dimension."""
        vol = np.random.rand(100, 100, 100).astype(np.float32)
        result = imresize3_average(vol, (1, 1, 5))
        
        # Only z dimension should change
        assert result.shape == (100, 100, 20)
    
    def test_uint8_input(self):
        """Test with uint8 input."""
        vol = np.random.randint(0, 256, (64, 64, 32), dtype=np.uint8)
        result = imresize3_average(vol, 2)
        
        # Output should be float32
        assert result.dtype == np.float32
        assert result.shape == (32, 32, 16)
    
    def test_uint16_input(self):
        """Test with uint16 input."""
        vol = np.random.randint(0, 4096, (64, 64, 32), dtype=np.uint16)
        result = imresize3_average(vol, 2)
        
        assert result.dtype == np.float32
        assert result.shape == (32, 32, 16)
    
    def test_odd_dimensions(self):
        """Test with odd dimensions that need padding."""
        vol = np.ones((33, 37, 41), dtype=np.float32)
        result = imresize3_average(vol, 2)
        
        # Should pad to 34, 38, 42 then downsample
        assert result.shape == (17, 19, 21)
    
    def test_gradient_volume(self):
        """Test with a gradient to verify averaging."""
        # Create a simple gradient along first dimension
        vol = np.zeros((8, 8, 8), dtype=np.float32)
        for i in range(8):
            vol[i, :, :] = float(i)
        
        result = imresize3_average(vol, 2)
        
        # Each downsampled block averages 2x2x2 voxels
        # result[0] should be average of vol[0:2] along first dimension
        # vol[0:2, :, :] has values 0 and 1, so average is 0.5
        assert result.shape == (4, 4, 4)
        assert np.allclose(result[0, 0, 0], 0.5)
        assert np.allclose(result[1, 0, 0], 2.5)
        assert np.allclose(result[2, 0, 0], 4.5)
        assert np.allclose(result[3, 0, 0], 6.5)
    
    def test_3d_checkerboard(self):
        """Test with 3D checkerboard pattern."""
        vol = np.zeros((8, 8, 8), dtype=np.float32)
        # Create checkerboard - 1 where sum of indices is even
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    if (i + j + k) % 2 == 0:
                        vol[i, j, k] = 1.0
        
        result = imresize3_average(vol, 2)
        
        # Each 2x2x2 block in checkerboard has 4 ones and 4 zeros
        # So average should be 0.5
        assert result.shape == (4, 4, 4)
        assert np.allclose(result, 0.5)
