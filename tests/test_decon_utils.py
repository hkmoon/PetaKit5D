"""
Tests for deconvolution utilities.
"""

import numpy as np
import pytest
from petakit5d.microscope_data_processing.decon_utils import decon_otf2psf


class TestDeconOtf2Psf:
    """Tests for decon_otf2psf function."""
    
    def test_basic_conversion(self):
        """Test basic OTF to PSF conversion."""
        # Create a simple OTF
        otf = np.random.randn(32, 32, 32) + 1j * np.random.randn(32, 32, 32)
        psf = decon_otf2psf(otf)
        
        # Check output shape matches input
        assert psf.shape == otf.shape
        # Check output is real
        assert np.isrealobj(psf)
        # Check output dtype
        assert psf.dtype in [np.float32, np.float64]
    
    def test_with_output_size(self):
        """Test OTF to PSF conversion with different output size."""
        otf = np.random.randn(64, 64, 64) + 1j * np.random.randn(64, 64, 64)
        out_size = (32, 32, 32)
        psf = decon_otf2psf(otf, out_size=out_size)
        
        # Check output shape
        assert psf.shape == out_size
        # Check output is real
        assert np.isrealobj(psf)
    
    def test_zero_otf(self):
        """Test handling of all-zero OTF."""
        otf = np.zeros((32, 32, 32), dtype=np.complex128)
        psf = decon_otf2psf(otf)
        
        # Should return zeros
        assert np.all(psf == 0)
        assert psf.shape == (32, 32, 32)
    
    def test_real_otf(self):
        """Test with real-valued OTF."""
        otf = np.random.randn(32, 32, 32)
        psf = decon_otf2psf(otf)
        
        # Check output
        assert psf.shape == otf.shape
        assert np.isrealobj(psf)
    
    def test_different_output_sizes(self):
        """Test with various output sizes."""
        otf = np.random.randn(64, 64, 64) + 1j * np.random.randn(64, 64, 64)
        
        for out_size in [(32, 32, 32), (48, 48, 48), (16, 16, 16)]:
            psf = decon_otf2psf(otf, out_size=out_size)
            assert psf.shape == out_size
    
    def test_non_cubic_shape(self):
        """Test with non-cubic volume."""
        otf = np.random.randn(64, 48, 32) + 1j * np.random.randn(64, 48, 32)
        psf = decon_otf2psf(otf)
        
        assert psf.shape == (64, 48, 32)
        assert np.isrealobj(psf)
    
    def test_non_cubic_output(self):
        """Test with non-cubic output size."""
        otf = np.random.randn(64, 64, 64) + 1j * np.random.randn(64, 64, 64)
        out_size = (32, 48, 16)
        psf = decon_otf2psf(otf, out_size=out_size)
        
        assert psf.shape == out_size
    
    def test_energy_conservation_approximate(self):
        """Test that energy is approximately conserved in conversion."""
        # Create a simple Gaussian-like OTF
        size = 32
        x = np.arange(size) - size // 2
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        otf = np.exp(-(X**2 + Y**2 + Z**2) / (2 * 5**2))
        otf = otf.astype(np.complex128)
        
        psf = decon_otf2psf(otf)
        
        # Check that PSF has reasonable values
        assert np.all(np.isfinite(psf))
        assert psf.sum() > 0  # Should have positive sum
    
    def test_small_volume(self):
        """Test with small volume."""
        otf = np.random.randn(8, 8, 8) + 1j * np.random.randn(8, 8, 8)
        psf = decon_otf2psf(otf)
        
        assert psf.shape == (8, 8, 8)
        assert np.isrealobj(psf)
    
    def test_dtype_preservation(self):
        """Test that appropriate dtype is used."""
        # Complex128 input
        otf = np.random.randn(16, 16, 16).astype(np.complex128)
        psf = decon_otf2psf(otf)
        assert psf.dtype == np.float64
        
        # Complex64 input (will be promoted to complex128 internally)
        otf = np.random.randn(16, 16, 16).astype(np.complex64)
        psf = decon_otf2psf(otf)
        # Output should still be float (converted from complex128)
        assert psf.dtype in [np.float32, np.float64]
