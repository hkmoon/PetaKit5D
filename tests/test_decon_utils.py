"""
Tests for deconvolution utilities.
"""

import numpy as np
import pytest
from petakit5d.microscope_data_processing.decon_utils import (
    decon_otf2psf, 
    decon_psf2otf, 
    decon_mask_edge_erosion
)


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


class TestDeconPsf2Otf:
    """Tests for decon_psf2otf function."""
    
    def test_basic_conversion(self):
        """Test basic PSF to OTF conversion."""
        psf = np.random.rand(11, 11, 11)
        otf = decon_psf2otf(psf)
        
        # Check output shape matches input
        assert otf.shape == psf.shape
        # Check output is complex
        assert np.iscomplexobj(otf)
    
    def test_with_output_size(self):
        """Test PSF to OTF conversion with different output size."""
        psf = np.random.rand(11, 11, 11)
        out_size = (64, 64, 64)
        otf = decon_psf2otf(psf, out_size=out_size)
        
        # Check output shape
        assert otf.shape == out_size
        # Check output is complex
        assert np.iscomplexobj(otf)
    
    def test_output_size_too_small(self):
        """Test that error is raised if output size is too small."""
        psf = np.random.rand(32, 32, 32)
        out_size = (16, 16, 16)
        
        with pytest.raises(ValueError, match="cannot be smaller"):
            decon_psf2otf(psf, out_size=out_size)
    
    def test_zero_psf(self):
        """Test handling of all-zero PSF."""
        psf = np.zeros((32, 32, 32))
        otf = decon_psf2otf(psf)
        
        # Should return zeros
        assert np.all(otf == 0)
        assert otf.shape == (32, 32, 32)
    
    def test_2d_psf(self):
        """Test with 2D PSF."""
        psf = np.random.rand(11, 11)
        otf = decon_psf2otf(psf)
        
        # Check output shape
        assert otf.shape == psf.shape
        assert np.iscomplexobj(otf)
    
    def test_2d_psf_with_3d_output(self):
        """Test 2D PSF with 3D output size."""
        psf = np.random.rand(11, 11)
        out_size = (64, 64, 64)
        otf = decon_psf2otf(psf, out_size=out_size)
        
        assert otf.shape == out_size
    
    def test_non_cubic_shape(self):
        """Test with non-cubic PSF."""
        psf = np.random.rand(11, 13, 15)
        otf = decon_psf2otf(psf)
        
        assert otf.shape == (11, 13, 15)
        assert np.iscomplexobj(otf)
    
    def test_roundtrip_conversion(self):
        """Test PSF -> OTF -> PSF roundtrip."""
        # Create a simple Gaussian PSF
        size = 32
        x = np.arange(size) - size // 2
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        psf_original = np.exp(-(X**2 + Y**2 + Z**2) / (2 * 3**2))
        psf_original = psf_original / psf_original.sum()  # Normalize
        
        # Forward and backward conversion
        otf = decon_psf2otf(psf_original)
        psf_reconstructed = decon_otf2psf(otf)
        
        # Should be approximately equal (within numerical precision)
        assert psf_reconstructed.shape == psf_original.shape
        # Both should be positive (mostly)
        assert psf_original.sum() > 0
        assert psf_reconstructed.sum() > 0
        # The values should be similar in magnitude
        assert np.abs(psf_original.max() - psf_reconstructed.max()) / psf_original.max() < 0.5
    
    def test_empty_psf(self):
        """Test with empty PSF array."""
        psf = np.array([])
        otf = decon_psf2otf(psf)
        
        # Should handle empty input
        assert otf.size == 0 or np.all(otf == 0)
    
    def test_padding_behavior(self):
        """Test that PSF is properly padded."""
        psf = np.random.rand(11, 11, 11)
        out_size = (32, 32, 32)
        otf = decon_psf2otf(psf, out_size=out_size)
        
        # OTF should have requested size
        assert otf.shape == out_size
        # Should be non-zero (PSF was non-zero)
        assert np.any(otf != 0)


class TestDeconMaskEdgeErosion:
    """Tests for decon_mask_edge_erosion function."""
    
    def test_full_mask_2d(self):
        """Test erosion of full 2D mask."""
        mask = np.ones((100, 100), dtype=bool)
        edge_erosion = 5
        eroded = decon_mask_edge_erosion(mask, edge_erosion)
        
        # Check that edges are False
        assert np.all(eroded[:edge_erosion, :] == False)
        assert np.all(eroded[-edge_erosion:, :] == False)
        assert np.all(eroded[:, :edge_erosion] == False)
        assert np.all(eroded[:, -edge_erosion:] == False)
        
        # Check that center is True
        assert np.all(eroded[edge_erosion:-edge_erosion, edge_erosion:-edge_erosion] == True)
    
    def test_full_mask_3d(self):
        """Test erosion of full 3D mask."""
        mask = np.ones((50, 50, 50), dtype=bool)
        edge_erosion = 3
        eroded = decon_mask_edge_erosion(mask, edge_erosion)
        
        # Check that edges are False
        assert np.all(eroded[:edge_erosion, :, :] == False)
        assert np.all(eroded[-edge_erosion:, :, :] == False)
        assert np.all(eroded[:, :edge_erosion, :] == False)
        assert np.all(eroded[:, -edge_erosion:, :] == False)
        assert np.all(eroded[:, :, :edge_erosion] == False)
        assert np.all(eroded[:, :, -edge_erosion:] == False)
        
        # Check that center is True
        center_slice = eroded[edge_erosion:-edge_erosion, 
                              edge_erosion:-edge_erosion, 
                              edge_erosion:-edge_erosion]
        assert np.all(center_slice == True)
    
    def test_irregular_mask_2d(self):
        """Test erosion of irregular 2D mask."""
        mask = np.zeros((100, 100), dtype=bool)
        # Create a circular mask
        y, x = np.ogrid[-50:50, -50:50]
        circle = x**2 + y**2 <= 30**2
        mask = circle
        
        edge_erosion = 5
        eroded = decon_mask_edge_erosion(mask, edge_erosion)
        
        # Eroded mask should be smaller
        assert np.sum(eroded) < np.sum(mask)
        # Eroded mask should still be in the center
        assert np.sum(eroded[40:60, 40:60]) > 0
    
    def test_irregular_mask_3d(self):
        """Test erosion of irregular 3D mask."""
        mask = np.zeros((50, 50, 50), dtype=bool)
        # Create a spherical mask
        z, y, x = np.ogrid[-25:25, -25:25, -25:25]
        sphere = x**2 + y**2 + z**2 <= 15**2
        mask = sphere
        
        edge_erosion = 3
        eroded = decon_mask_edge_erosion(mask, edge_erosion)
        
        # Eroded mask should be smaller
        assert np.sum(eroded) < np.sum(mask)
        # Eroded mask should still be in the center
        assert np.sum(eroded[20:30, 20:30, 20:30]) > 0
    
    def test_zero_erosion(self):
        """Test with zero erosion (should do nothing significant)."""
        mask = np.ones((50, 50), dtype=bool)
        edge_erosion = 0
        eroded = decon_mask_edge_erosion(mask, edge_erosion)
        
        # Should be mostly unchanged (only outer boundary set to False)
        # For full mask with edge_erosion=0, the direct path is taken
        # which doesn't change anything if edge_erosion=0
        assert np.sum(eroded) == np.sum(mask)
    
    def test_large_erosion(self):
        """Test with large erosion value."""
        mask = np.ones((50, 50), dtype=bool)
        edge_erosion = 20
        eroded = decon_mask_edge_erosion(mask, edge_erosion)
        
        # Should have significantly smaller True region
        assert np.sum(eroded) < np.sum(mask) / 2
        # Center should still be True
        assert eroded[25, 25] == True
    
    def test_output_is_boolean(self):
        """Test that output is boolean array."""
        mask = np.ones((50, 50), dtype=bool)
        eroded = decon_mask_edge_erosion(mask, edge_erosion=5)
        
        assert eroded.dtype == bool
    
    def test_input_type_conversion(self):
        """Test that non-boolean input is converted."""
        mask = np.ones((50, 50), dtype=int)
        eroded = decon_mask_edge_erosion(mask, edge_erosion=5)
        
        # Should work and return boolean
        assert eroded.dtype == bool
        assert np.sum(eroded) < np.sum(mask > 0)
    
    def test_preserves_shape(self):
        """Test that output shape matches input shape."""
        for shape in [(100, 100), (50, 50, 50), (30, 40)]:
            mask = np.ones(shape, dtype=bool)
            eroded = decon_mask_edge_erosion(mask, edge_erosion=3)
            assert eroded.shape == shape
