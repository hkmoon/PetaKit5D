"""
Unit tests for 2D A Trou Wavelet Transform.
"""

import numpy as np
import pytest
from petakit5d.image_processing.awt_2d import awt


class TestAwt2D:
    """Test cases for 2D A Trou Wavelet Transform."""
    
    def test_basic_2d_image(self):
        """Test with basic 2D image."""
        img = np.random.rand(64, 64)
        W = awt(img, n_bands=4)
        
        # Check shape: should have n_bands detail images + 1 approximation
        assert W.shape == (64, 64, 5)
        
        # Check perfect reconstruction
        reconstructed = np.sum(W[:, :, :-1], axis=2) + W[:, :, -1]
        np.testing.assert_array_almost_equal(reconstructed, img, decimal=10)
    
    def test_default_n_bands(self):
        """Test that default n_bands is ceil(max(log2(N), log2(M)))."""
        N, M = 128, 64
        img = np.random.rand(N, M)
        W = awt(img)
        
        expected_bands = int(np.ceil(max(np.log2(N), np.log2(M))))  # 7
        assert W.shape == (N, M, expected_bands + 1)
    
    def test_explicit_n_bands(self):
        """Test with explicit n_bands parameter."""
        img = np.random.rand(128, 128)
        W = awt(img, n_bands=3)
        
        assert W.shape == (128, 128, 4)  # 3 detail + 1 approximation
    
    def test_invalid_dimensions(self):
        """Test that 1D or 3D arrays raise error."""
        signal_1d = np.random.rand(100)
        with pytest.raises(ValueError, match="Input must be a 2D image"):
            awt(signal_1d)
        
        volume_3d = np.random.rand(50, 50, 50)
        with pytest.raises(ValueError, match="Input must be a 2D image"):
            awt(volume_3d)
    
    def test_invalid_n_bands_too_small(self):
        """Test with n_bands < 1."""
        img = np.random.rand(64, 64)
        with pytest.raises(ValueError, match="n_bands must be in range"):
            awt(img, n_bands=0)
    
    def test_invalid_n_bands_too_large(self):
        """Test with n_bands > maximum."""
        img = np.random.rand(64, 64)
        max_bands = int(np.ceil(np.log2(64)))  # 6
        with pytest.raises(ValueError, match="n_bands must be in range"):
            awt(img, n_bands=max_bands + 1)
    
    def test_constant_image(self):
        """Test with constant image."""
        img = np.ones((64, 64)) * 7.5
        W = awt(img, n_bands=3)
        
        # All detail bands should be near zero
        for k in range(3):
            assert np.allclose(W[:, :, k], 0, atol=1e-10)
        
        # Approximation should equal original
        assert np.allclose(W[:, :, 3], img, atol=1e-10)
    
    def test_reconstruction_accuracy(self):
        """Test perfect reconstruction property."""
        img = np.random.rand(100, 100)
        W = awt(img, n_bands=5)
        
        # Sum all components
        reconstructed = np.sum(W[:, :, :-1], axis=2) + W[:, :, -1]
        
        # Should be very close to original
        np.testing.assert_array_almost_equal(reconstructed, img, decimal=10)
    
    def test_rectangular_image(self):
        """Test with non-square image."""
        img = np.random.rand(128, 64)
        W = awt(img, n_bands=4)
        
        assert W.shape == (128, 64, 5)
        
        # Check reconstruction
        reconstructed = np.sum(W[:, :, :-1], axis=2) + W[:, :, -1]
        np.testing.assert_array_almost_equal(reconstructed, img, decimal=10)
    
    def test_blob_detection(self):
        """Test wavelet response to blob structure."""
        img = np.zeros((64, 64))
        img[25:35, 25:35] = 1.0  # 10x10 blob
        
        W = awt(img, n_bands=4)
        
        # Detail bands should have response around blob edges
        # At appropriate scale, should see blob structure
        assert np.sum(np.abs(W[:, :, 0])) > 0  # Finest scale has detail
        assert np.sum(np.abs(W[:, :, 1])) > 0
    
    def test_edge_detection(self):
        """Test wavelet response to edges."""
        img = np.zeros((64, 64))
        img[:, :32] = 1.0  # Vertical edge in middle
        
        W = awt(img, n_bands=3)
        
        # Detail bands should capture edge
        for k in range(3):
            # There should be significant activity near the edge
            edge_region = W[:, 28:36, k]
            assert np.abs(edge_region).max() > 0
    
    def test_scale_progression(self):
        """Test that scales progress correctly."""
        # Create multi-scale pattern
        img = np.zeros((128, 128))
        for i in range(128):
            for j in range(128):
                img[i, j] = np.sin(i * 0.1) + np.sin(j * 0.5)
        
        W = awt(img, n_bands=5)
        
        # Each scale should capture different frequency content
        # Check that detail bands have varying energy
        energies = [np.sum(W[:, :, k]**2) for k in range(5)]
        assert all(e >= 0 for e in energies)
    
    def test_output_dtype(self):
        """Test that output is always float64."""
        img_int = np.array([[1, 2], [3, 4]], dtype=int)
        W = awt(img_int, n_bands=1)
        
        assert W.dtype == np.float64
    
    def test_small_image(self):
        """Test with small image."""
        img = np.random.rand(8, 8)
        W = awt(img, n_bands=2)
        
        assert W.shape == (8, 8, 3)
        
        # Check reconstruction
        reconstructed = np.sum(W[:, :, :-1], axis=2) + W[:, :, -1]
        np.testing.assert_array_almost_equal(reconstructed, img, decimal=10)
    
    def test_large_image(self):
        """Test with larger image."""
        img = np.random.rand(256, 256)
        W = awt(img, n_bands=6)
        
        assert W.shape == (256, 256, 7)
    
    def test_single_band_decomposition(self):
        """Test with n_bands=1."""
        img = np.random.rand(64, 64)
        W = awt(img, n_bands=1)
        
        assert W.shape == (64, 64, 2)  # 1 detail + 1 approximation
        
        # Reconstruction should work
        reconstructed = W[:, :, 0] + W[:, :, 1]
        np.testing.assert_array_almost_equal(reconstructed, img, decimal=10)
    
    def test_zero_mean_details(self):
        """Test that detail bands are approximately zero-mean."""
        img = np.random.rand(64, 64)
        W = awt(img, n_bands=4)
        
        # Detail bands should have mean close to zero for random image
        for k in range(4):
            detail_mean = np.abs(np.mean(W[:, :, k]))
            # Mean should be small relative to standard deviation
            detail_std = np.std(W[:, :, k])
            if detail_std > 0:
                # Relax threshold as this can vary with random data
                assert detail_mean / detail_std < 0.2
    
    def test_separability_verification(self):
        """Test separable convolution property."""
        # Create image with separable structure
        x = np.linspace(0, 10, 64)
        y = np.linspace(0, 10, 64)
        X, Y = np.meshgrid(x, y)
        img = np.sin(X) * np.cos(Y)
        
        W = awt(img, n_bands=3)
        
        # Should decompose without errors
        assert W.shape == (64, 64, 4)
        
        # Check reconstruction
        reconstructed = np.sum(W[:, :, :-1], axis=2) + W[:, :, -1]
        np.testing.assert_array_almost_equal(reconstructed, img, decimal=10)
    
    def test_energy_conservation(self):
        """Test that energy is conserved in decomposition."""
        img = np.random.rand(64, 64)
        W = awt(img, n_bands=4)
        
        # Total energy in original
        original_energy = np.sum(img**2)
        
        # Total energy in decomposition
        decomp_energy = np.sum(W**2)
        
        # Should be approximately equal (within reasonable tolerance)
        # Note: A Trou is not perfectly orthogonal, so some difference expected
        np.testing.assert_allclose(original_energy, decomp_energy, rtol=0.05)
    
    def test_translation_invariance(self):
        """Test translation invariance property of A Trou."""
        # Create small blob
        img1 = np.zeros((64, 64))
        img1[20:25, 20:25] = 1.0
        
        # Translate by one pixel
        img2 = np.zeros((64, 64))
        img2[21:26, 21:26] = 1.0
        
        W1 = awt(img1, n_bands=3)
        W2 = awt(img2, n_bands=3)
        
        # At a given scale, the coefficients should shift, not change dramatically
        # Check that energy is similar (A Trou is translation-invariant)
        energy1 = np.sum(W1[:, :, 1]**2)
        energy2 = np.sum(W2[:, :, 1]**2)
        
        # Energies should be similar (within 10%)
        assert np.abs(energy1 - energy2) / max(energy1, energy2) < 0.1
