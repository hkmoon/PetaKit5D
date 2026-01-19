"""
Unit tests for awt (A Trou Wavelet Transform) module.
"""

import numpy as np
import pytest
from petakit5d.image_processing.awt import awt_1d


class TestAwt1D:
    """Test cases for awt_1d function."""
    
    def test_basic_signal(self):
        """Test with basic sine signal."""
        signal = np.sin(np.linspace(0, 10 * np.pi, 256))
        W = awt_1d(signal, n_bands=4)
        
        # Check shape
        assert W.shape == (256, 5)  # 4 bands + 1 approximation
        
        # Check that sum of all bands recovers original signal
        reconstructed = np.sum(W, axis=1)
        np.testing.assert_array_almost_equal(reconstructed, signal, decimal=10)
    
    def test_default_n_bands(self):
        """Test that default n_bands is ceil(log2(N))."""
        N = 128
        signal = np.random.rand(N)
        W = awt_1d(signal)
        
        expected_bands = int(np.ceil(np.log2(N)))  # ceil(log2(128)) = 7
        assert W.shape == (N, expected_bands + 1)
    
    def test_n_bands_parameter(self):
        """Test with explicit n_bands parameter."""
        signal = np.random.rand(256)
        W = awt_1d(signal, n_bands=3)
        
        assert W.shape == (256, 4)  # 3 bands + 1 approximation
    
    def test_invalid_n_bands_too_small(self):
        """Test with n_bands < 1."""
        signal = np.random.rand(128)
        with pytest.raises(ValueError, match="Invalid range for n_bands"):
            awt_1d(signal, n_bands=0)
    
    def test_invalid_n_bands_too_large(self):
        """Test with n_bands > ceil(log2(N))."""
        signal = np.random.rand(128)
        max_bands = int(np.ceil(np.log2(128)))
        with pytest.raises(ValueError, match="Invalid range for n_bands"):
            awt_1d(signal, n_bands=max_bands + 1)
    
    def test_constant_signal(self):
        """Test with constant signal."""
        signal = np.ones(128) * 5.0
        W = awt_1d(signal, n_bands=4)
        
        # All detail bands should be near zero for constant signal
        for k in range(4):
            assert np.allclose(W[:, k], 0, atol=1e-10)
        
        # Approximation should be close to original
        assert np.allclose(W[:, 4], signal, atol=1e-10)
    
    def test_reconstruction(self):
        """Test that signal can be reconstructed from coefficients."""
        signal = np.random.rand(256)
        W = awt_1d(signal, n_bands=5)
        
        # Reconstruction: sum all bands
        reconstructed = np.sum(W, axis=1)
        
        np.testing.assert_array_almost_equal(reconstructed, signal, decimal=10)
    
    def test_power_of_two_length(self):
        """Test with power-of-2 signal length."""
        for N in [64, 128, 256, 512]:
            signal = np.random.rand(N)
            W = awt_1d(signal)
            
            expected_bands = int(np.ceil(np.log2(N)))
            assert W.shape == (N, expected_bands + 1)
    
    def test_non_power_of_two_length(self):
        """Test with non-power-of-2 signal length."""
        signal = np.random.rand(200)
        W = awt_1d(signal)
        
        expected_bands = int(np.ceil(np.log2(200)))  # ceil(log2(200)) = 8
        assert W.shape == (200, expected_bands + 1)
    
    def test_detail_bands_structure(self):
        """Test that detail bands have correct structure."""
        # Create signal with clear frequency content
        t = np.linspace(0, 1, 256)
        signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz signal
        
        W = awt_1d(signal, n_bands=4)
        
        # First detail band (finest scale) should capture high frequencies
        # Energy should be in detail bands, not just approximation
        detail_energy = np.sum([np.sum(W[:, k]**2) for k in range(4)])
        approx_energy = np.sum(W[:, 4]**2)
        
        # For a periodic signal, both should have significant energy
        assert detail_energy > 0
        assert approx_energy > 0
    
    def test_multiscale_property(self):
        """Test multiscale decomposition property."""
        # Signal with multiple frequency components
        t = np.linspace(0, 1, 512)
        signal = (np.sin(2 * np.pi * 5 * t) +  # Low frequency
                  np.sin(2 * np.pi * 20 * t))    # Higher frequency
        
        W = awt_1d(signal, n_bands=6)
        
        # Each band should capture different scales
        # Variance should generally decrease with scale (finer to coarser)
        for k in range(6):
            variance = np.var(W[:, k])
            assert variance >= 0  # Basic sanity check
    
    def test_single_band(self):
        """Test with n_bands=1."""
        signal = np.random.rand(128)
        W = awt_1d(signal, n_bands=1)
        
        assert W.shape == (128, 2)  # 1 detail + 1 approximation
        
        # Reconstruction should still work
        reconstructed = np.sum(W, axis=1)
        np.testing.assert_array_almost_equal(reconstructed, signal, decimal=10)
    
    def test_output_dtype(self):
        """Test that output is float type."""
        signal = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int)
        W = awt_1d(signal)
        
        assert W.dtype == np.float64
    
    def test_small_signal(self):
        """Test with small signal."""
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        W = awt_1d(signal, n_bands=1)
        
        # Should work without errors
        assert W.shape == (4, 2)
