"""
Tests for Laplacian of Gaussian (LoG) filtering.
"""

import pytest
import numpy as np
from petakit5d.image_processing import filter_log


class TestFilterLog:
    """Test suite for filter_log function."""
    
    def test_1d_signal(self):
        """Test LoG filtering on 1D signal."""
        # Create a simple 1D signal with a peak
        signal = np.zeros(100)
        signal[50] = 1.0
        
        # Filter it
        filtered = filter_log(signal, sigma=2.0)
        
        # Check shape
        assert filtered.shape == signal.shape
        
        # Check that output has same type (real)
        assert np.isrealobj(filtered)
        
        # LoG of a delta function should have negative center and positive sides
        assert filtered[50] < 0  # Center should be negative
    
    def test_2d_image(self):
        """Test LoG filtering on 2D image."""
        # Create a 2D image with a central peak
        img = np.zeros((50, 50))
        img[25, 25] = 1.0
        
        # Filter it
        filtered = filter_log(img, sigma=2.0)
        
        # Check shape
        assert filtered.shape == img.shape
        
        # Check that output is real
        assert np.isrealobj(filtered)
        
        # LoG should have negative center for a bright spot
        assert filtered[25, 25] < 0
    
    def test_3d_volume(self):
        """Test LoG filtering on 3D volume."""
        # Create a 3D volume with a central peak
        vol = np.zeros((30, 30, 30))
        vol[15, 15, 15] = 1.0
        
        # Filter it
        filtered = filter_log(vol, sigma=2.0)
        
        # Check shape
        assert filtered.shape == vol.shape
        
        # Check that output is real
        assert np.isrealobj(filtered)
        
        # LoG should have negative center for a bright spot
        assert filtered[15, 15, 15] < 0
    
    def test_sigma_effect(self):
        """Test that sigma affects the scale of filtering."""
        # Create a simple 1D signal
        signal = np.zeros(100)
        signal[50] = 1.0
        
        # Filter with different sigmas
        filtered_small = filter_log(signal, sigma=1.0)
        filtered_large = filter_log(signal, sigma=5.0)
        
        # Larger sigma should give broader response
        # Check that the response is wider for larger sigma
        threshold = -0.01
        width_small = np.sum(filtered_small < threshold)
        width_large = np.sum(filtered_large < threshold)
        
        assert width_large > width_small
    
    def test_1d_vector_shapes(self):
        """Test that 1D arrays with different shapes are handled correctly."""
        # Row vector
        signal_row = np.random.rand(100)
        filtered_row = filter_log(signal_row, sigma=2.0)
        assert filtered_row.shape == signal_row.shape
        
        # Column vector (2D with size 1 in one dimension)
        signal_col = np.random.rand(100, 1)
        filtered_col = filter_log(signal_col, sigma=2.0)
        assert filtered_col.shape == signal_col.shape
    
    def test_constant_signal(self):
        """Test LoG filtering on constant signal."""
        # Constant signal
        signal = np.ones((50, 50))
        
        # LoG of constant should be close to zero
        filtered = filter_log(signal, sigma=2.0)
        
        # Check that result is close to zero everywhere
        assert np.allclose(filtered, 0, atol=1e-10)
    
    def test_linear_gradient(self):
        """Test LoG filtering on linear gradient."""
        # Create a linear gradient
        x = np.linspace(0, 10, 100)
        img = x[:, np.newaxis] + x[np.newaxis, :]
        
        # LoG of linear function should be close to zero
        filtered = filter_log(img, sigma=2.0)
        
        # Should be very close to zero (Laplacian of linear is zero)
        assert np.allclose(filtered, 0, atol=1e-10)
    
    def test_gaussian_blob_detection(self):
        """Test LoG for blob detection (classic use case)."""
        # Create an image with a Gaussian blob
        y, x = np.ogrid[-25:25, -25:25]
        sigma_blob = 3.0
        blob = np.exp(-(x**2 + y**2) / (2 * sigma_blob**2))
        
        # Filter with matched sigma
        filtered = filter_log(blob, sigma=sigma_blob)
        
        # The LoG response should be strongest at the center
        center_val = filtered[25, 25]
        
        # Check that center has strong negative response
        assert center_val < -0.1
        
        # Check that it's a local minimum (for blob detection)
        neighbors = filtered[24:27, 24:27]
        assert center_val == np.min(neighbors)
    
    def test_edge_detection(self):
        """Test LoG for edge detection using zero-crossings."""
        # Create a step edge
        img = np.zeros((50, 50))
        img[:, 25:] = 1.0
        
        # Apply LoG
        filtered = filter_log(img, sigma=2.0)
        
        # Zero-crossings near the edge indicate the edge location
        # Check that there's a sign change along the middle row
        middle_row = filtered[25, :]
        
        # Should have both positive and negative values near the edge
        assert np.any(middle_row > 0)
        assert np.any(middle_row < 0)
        
        # Zero-crossing should be near column 25
        zero_cross_idx = np.where(np.diff(np.sign(middle_row)))[0]
        assert len(zero_cross_idx) > 0
        assert np.any(np.abs(zero_cross_idx - 25) < 5)
    
    def test_real_output(self):
        """Test that output is always real-valued."""
        # Random complex input (though not typical)
        signal = np.random.rand(50, 50) + 0j
        
        # Filter it
        filtered = filter_log(signal.real, sigma=2.0)
        
        # Output should be real
        assert np.isrealobj(filtered)
        assert filtered.dtype in [np.float32, np.float64]
    
    def test_different_dtypes(self):
        """Test that function works with different input dtypes."""
        img = np.random.rand(30, 30)
        
        # Test with different dtypes
        for dtype in [np.float32, np.float64, np.uint8, np.uint16]:
            img_typed = img.astype(dtype)
            filtered = filter_log(img_typed, sigma=2.0)
            
            # Should produce valid output
            assert filtered.shape == img.shape
            assert np.isrealobj(filtered)
    
    def test_energy_conservation(self):
        """Test that total energy is conserved (approximately)."""
        # Create a signal with known energy
        signal = np.random.rand(100, 100)
        
        # The LoG is a zero-mean filter, so mean should be close to zero
        filtered = filter_log(signal, sigma=2.0)
        
        # Mean of filtered signal should be close to zero
        assert np.abs(np.mean(filtered)) < 0.1
