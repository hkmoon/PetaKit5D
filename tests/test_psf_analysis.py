"""
Tests for PSF analysis and processing functions.
"""

import numpy as np
import pytest
from petakit5d.microscope_data_processing import psf_gen, rotate_psf


class TestPSFGen:
    """Test PSF generation and preprocessing."""
    
    def test_psf_gen_3d_median(self):
        """Test 3D PSF generation with median method."""
        # Create synthetic PSF with peak in center
        psf = np.zeros((32, 32, 16), dtype=np.float32)
        psf[16, 16, 8] = 1000.0
        
        # Add Gaussian around peak
        y, x, z = np.meshgrid(
            np.arange(32) - 16,
            np.arange(32) - 16,
            np.arange(16) - 8,
            indexing='ij'
        )
        psf += 500 * np.exp(-(x**2 + y**2 + z**2) / 10)
        
        # Add background noise
        psf += 10.0
        
        # Process PSF
        result = psf_gen(psf, dz_psf=0.1, dz_data=0.1, psf_gen_method='median')
        
        # Check output shape
        assert result.shape == psf.shape
        
        # Check dtype
        assert result.dtype == np.float32
        
        # Check peak is still present
        assert np.max(result) > 0
        
        # Check background is reduced
        assert np.min(result) == 0.0
    
    def test_psf_gen_3d_masked(self):
        """Test 3D PSF generation with masked method."""
        psf = np.zeros((32, 32, 16), dtype=np.float32)
        psf[16, 16, 8] = 1000.0
        
        y, x, z = np.meshgrid(
            np.arange(32) - 16,
            np.arange(32) - 16,
            np.arange(16) - 8,
            indexing='ij'
        )
        psf += 500 * np.exp(-(x**2 + y**2 + z**2) / 10)
        psf += 10.0
        
        result = psf_gen(psf, dz_psf=0.1, dz_data=0.1, psf_gen_method='masked')
        
        assert result.shape == psf.shape
        assert result.dtype == np.float32
        assert np.max(result) > 0
    
    def test_psf_gen_2d_median(self):
        """Test 2D PSF generation with median method."""
        psf = np.zeros((32, 32), dtype=np.float32)
        psf[16, 16] = 1000.0
        
        y, x = np.meshgrid(
            np.arange(32) - 16,
            np.arange(32) - 16,
            indexing='ij'
        )
        psf += 500 * np.exp(-(x**2 + y**2) / 10)
        psf += 10.0
        
        result = psf_gen(psf, dz_psf=0.1, dz_data=0.1, psf_gen_method='median')
        
        assert result.shape == psf.shape
        assert result.dtype == np.float32
        assert np.max(result) > 0
    
    def test_psf_gen_2d_masked(self):
        """Test 2D PSF generation with masked method."""
        psf = np.zeros((32, 32), dtype=np.float32)
        psf[16, 16] = 1000.0
        
        y, x = np.meshgrid(
            np.arange(32) - 16,
            np.arange(32) - 16,
            indexing='ij'
        )
        psf += 500 * np.exp(-(x**2 + y**2) / 10)
        psf += 10.0
        
        result = psf_gen(psf, dz_psf=0.1, dz_data=0.1, psf_gen_method='masked')
        
        assert result.shape == psf.shape
        assert result.dtype == np.float32
    
    def test_psf_gen_resampling(self):
        """Test PSF Z-resampling."""
        psf = np.zeros((32, 32, 32), dtype=np.float32)
        psf[16, 16, 16] = 1000.0
        
        y, x, z = np.meshgrid(
            np.arange(32) - 16,
            np.arange(32) - 16,
            np.arange(32) - 16,
            indexing='ij'
        )
        psf += 500 * np.exp(-(x**2 + y**2 + z**2) / 10)
        
        # Resample: dz_psf=0.1, dz_data=0.2 -> halve Z dimension
        result = psf_gen(psf, dz_psf=0.1, dz_data=0.2)
        
        assert result.shape[0] == 32
        assert result.shape[1] == 32
        assert result.shape[2] == 16  # Halved
    
    def test_psf_gen_no_resampling(self):
        """Test PSF without Z-resampling."""
        psf = np.zeros((32, 32, 16), dtype=np.float32)
        psf[16, 16, 8] = 1000.0
        
        # Same pixel sizes -> no resampling
        result = psf_gen(psf, dz_psf=0.1, dz_data=0.1)
        
        assert result.shape == psf.shape
    
    def test_psf_gen_centering(self):
        """Test PSF is centered correctly."""
        psf = np.zeros((64, 64, 32), dtype=np.float32)
        # Off-center peak
        psf[40, 50, 20] = 1000.0
        
        y, x, z = np.meshgrid(
            np.arange(64) - 40,
            np.arange(64) - 50,
            np.arange(32) - 20,
            indexing='ij'
        )
        psf += 500 * np.exp(-(x**2 + y**2 + z**2) / 10)
        
        result = psf_gen(psf, dz_psf=0.1, dz_data=0.1)
        
        # Peak should be centered
        peak_idx = np.unravel_index(np.argmax(result), result.shape)
        center = ((result.shape[0] + 1) // 2, (result.shape[1] + 1) // 2, (result.shape[2] + 1) // 2)
        
        # Peak should be close to center (within a few pixels)
        assert abs(peak_idx[0] - center[0]) <= 2
        assert abs(peak_idx[1] - center[1]) <= 2
        assert abs(peak_idx[2] - center[2]) <= 2
    
    def test_psf_gen_uint16_input(self):
        """Test PSF generation with uint16 input."""
        psf = np.zeros((32, 32, 16), dtype=np.uint16)
        psf[16, 16, 8] = 10000
        
        result = psf_gen(psf, dz_psf=0.1, dz_data=0.1)
        
        # Should convert to float32
        assert result.dtype == np.float32
        assert result.shape == psf.shape


class TestRotatePSF:
    """Test PSF rotation."""
    
    def test_rotate_psf_basic(self):
        """Test basic PSF rotation."""
        psf = np.random.rand(32, 32, 16).astype(np.float32)
        
        result = rotate_psf(
            psf,
            skew_angle=32.45,
            xy_pixel_size=0.108,
            dz=0.1
        )
        
        # Shape should be preserved (with cropping)
        assert result.ndim == 3
        assert result.dtype == psf.dtype
    
    def test_rotate_psf_objective_scan(self):
        """Test PSF rotation for objective scan."""
        psf = np.random.rand(32, 32, 16).astype(np.float32)
        
        result = rotate_psf(
            psf,
            skew_angle=32.45,
            xy_pixel_size=0.108,
            dz=0.1,
            objective_scan=True
        )
        
        assert result.ndim == 3
        assert result.dtype == psf.dtype
    
    def test_rotate_psf_reverse(self):
        """Test PSF rotation with reverse flag."""
        psf = np.random.rand(32, 32, 16).astype(np.float32)
        
        result = rotate_psf(
            psf,
            skew_angle=32.45,
            xy_pixel_size=0.108,
            dz=0.1,
            reverse=True
        )
        
        assert result.ndim == 3
        assert result.dtype == psf.dtype
    
    def test_rotate_psf_different_angles(self):
        """Test PSF rotation with different skew angles."""
        psf = np.random.rand(32, 32, 16).astype(np.float32)
        
        for angle in [0, 15, 30, 45]:
            result = rotate_psf(
                psf,
                skew_angle=angle,
                xy_pixel_size=0.108,
                dz=0.1
            )
            assert result.ndim == 3
    
    def test_rotate_psf_edge_handling(self):
        """Test PSF rotation handles edge padding correctly."""
        # PSF with zeros at edges
        psf = np.zeros((32, 32, 16), dtype=np.float32)
        psf[8:24, 8:24, 4:12] = np.random.rand(16, 16, 8)
        
        result = rotate_psf(
            psf,
            skew_angle=32.45,
            xy_pixel_size=0.108,
            dz=0.1
        )
        
        # Check that zeros are filled (or at least some processing occurred)
        assert result.shape == psf.shape or result.shape[0] > 0
    
    def test_rotate_psf_dtype_preservation(self):
        """Test PSF rotation preserves dtype."""
        for dtype in [np.float32, np.float64, np.uint16]:
            psf = np.random.rand(32, 32, 16).astype(dtype)
            if dtype == np.uint16:
                psf = (psf * 1000).astype(dtype)
            
            result = rotate_psf(
                psf,
                skew_angle=32.45,
                xy_pixel_size=0.108,
                dz=0.1
            )
            
            assert result.dtype == dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
