"""
Unit tests for N-dimensional Laplacian of Gaussian filter.
"""

import numpy as np
import pytest
from petakit5d.image_processing.filter_log_nd import (
    filter_log_nd, 
    filter_log_nd_with_kernel
)


class TestFilterLogND:
    """Test cases for N-dimensional LoG filter."""
    
    def test_basic_2d_filtering(self):
        """Test basic 2D LoG filtering."""
        img = np.random.rand(50, 50)
        log_response = filter_log_nd(img, sigma=2.0)
        
        assert log_response.shape == img.shape
        assert log_response.dtype == np.float64
    
    def test_basic_3d_filtering(self):
        """Test basic 3D LoG filtering."""
        vol = np.random.rand(30, 30, 20)
        log_response = filter_log_nd(vol, sigma=2.0)
        
        assert log_response.shape == vol.shape
    
    def test_1d_filtering(self):
        """Test 1D LoG filtering."""
        signal = np.random.rand(100)
        log_response = filter_log_nd(signal, sigma=2.0)
        
        assert log_response.shape == signal.shape
    
    def test_with_kernel_return(self):
        """Test returning kernel with response."""
        img = np.random.rand(40, 40)
        log_response, log_kernel = filter_log_nd_with_kernel(img, sigma=2.0)
        
        assert log_response.shape == img.shape
        assert log_kernel is not None
        assert log_kernel.ndim == 2
    
    def test_isotropic_spacing(self):
        """Test with isotropic spacing (scalar)."""
        img = np.random.rand(50, 50)
        log_response = filter_log_nd(img, sigma=2.0, spacing=1.0)
        
        assert log_response.shape == img.shape
    
    def test_anisotropic_spacing_2d(self):
        """Test with anisotropic spacing in 2D."""
        img = np.random.rand(50, 50)
        log_response = filter_log_nd(img, sigma=2.0, spacing=(1.0, 2.0))
        
        assert log_response.shape == img.shape
    
    def test_anisotropic_spacing_3d(self):
        """Test with anisotropic spacing in 3D."""
        vol = np.random.rand(30, 30, 20)
        log_response = filter_log_nd(vol, sigma=2.0, spacing=(1.0, 1.0, 2.0))
        
        assert log_response.shape == vol.shape
    
    def test_border_condition_symmetric(self):
        """Test with symmetric border condition."""
        img = np.random.rand(40, 40)
        log_response = filter_log_nd(img, sigma=2.0, border_condition='symmetric')
        
        assert log_response.shape == img.shape
    
    def test_border_condition_replicate(self):
        """Test with replicate border condition."""
        img = np.random.rand(40, 40)
        log_response = filter_log_nd(img, sigma=2.0, border_condition='replicate')
        
        assert log_response.shape == img.shape
    
    def test_border_condition_constant(self):
        """Test with constant border condition."""
        img = np.random.rand(40, 40)
        log_response = filter_log_nd(img, sigma=2.0, border_condition='constant')
        
        assert log_response.shape == img.shape
    
    def test_border_condition_wrap(self):
        """Test with wrap border condition."""
        img = np.random.rand(40, 40)
        log_response = filter_log_nd(img, sigma=2.0, border_condition='wrap')
        
        assert log_response.shape == img.shape
    
    def test_border_condition_numeric(self):
        """Test with numeric constant border value."""
        img = np.random.rand(40, 40)
        log_response = filter_log_nd(img, sigma=2.0, border_condition=0.5)
        
        assert log_response.shape == img.shape
    
    def test_normalized_derivatives_false(self):
        """Test with unnormalized derivatives."""
        img = np.random.rand(50, 50)
        log_response = filter_log_nd(img, sigma=2.0, 
                                     use_normalized_derivatives=False)
        
        assert log_response.shape == img.shape
    
    def test_normalized_derivatives_true(self):
        """Test with scale-normalized derivatives."""
        img = np.random.rand(50, 50)
        log_response = filter_log_nd(img, sigma=2.0, 
                                     use_normalized_derivatives=True)
        
        assert log_response.shape == img.shape
    
    def test_normalized_gaussian_false(self):
        """Test with unnormalized Gaussian."""
        img = np.random.rand(50, 50)
        log_response = filter_log_nd(img, sigma=2.0, 
                                     use_normalized_gaussian=False)
        
        assert log_response.shape == img.shape
    
    def test_blob_detection_2d(self):
        """Test blob detection in 2D."""
        # Create image with blob
        img = np.zeros((60, 60))
        x = np.linspace(-3, 3, 60)
        y = np.linspace(-3, 3, 60)
        X, Y = np.meshgrid(x, y)
        img = np.exp(-(X**2 + Y**2) / 2)  # Gaussian blob
        
        # LoG should give strong response at blob center
        log_response = filter_log_nd(img, sigma=1.0, 
                                     use_normalized_derivatives=True)
        
        # Center should have strong response
        center_response = np.abs(log_response[30, 30])
        edge_response = np.abs(log_response[10, 10])
        
        # Center response should be stronger than edge
        assert center_response > edge_response
    
    def test_edge_detection(self):
        """Test edge detection with LoG."""
        # Create step edge
        img = np.zeros((50, 50))
        img[:, 25:] = 1.0
        
        log_response = filter_log_nd(img, sigma=2.0)
        
        # Should have response along edge
        edge_region = log_response[:, 23:27]
        assert np.max(np.abs(edge_region)) > 0
    
    def test_kernel_properties(self):
        """Test that kernel has expected properties."""
        img = np.random.rand(40, 40)
        _, kernel = filter_log_nd_with_kernel(img, sigma=2.0)
        
        # Kernel should be zero-mean (approximately)
        assert np.abs(np.mean(kernel)) < 1e-10
        
        # Kernel should be symmetric
        assert np.allclose(kernel, kernel[::-1, ::-1])
    
    def test_kernel_normalization(self):
        """Test kernel normalization."""
        img = np.random.rand(40, 40)
        
        _, kernel_normalized = filter_log_nd_with_kernel(
            img, sigma=2.0, use_normalized_gaussian=True
        )
        
        _, kernel_unnormalized = filter_log_nd_with_kernel(
            img, sigma=2.0, use_normalized_gaussian=False
        )
        
        # Should be different
        assert not np.allclose(kernel_normalized, kernel_unnormalized)
    
    def test_different_sigmas(self):
        """Test with different sigma values."""
        img = np.random.rand(50, 50)
        
        log_small = filter_log_nd(img, sigma=1.0)
        log_large = filter_log_nd(img, sigma=3.0)
        
        # Different scales should give different responses
        assert not np.allclose(log_small, log_large)
    
    def test_scale_invariance_with_normalization(self):
        """Test scale-normalized response."""
        # Create blob at two different scales
        x = np.linspace(-5, 5, 80)
        y = np.linspace(-5, 5, 80)
        X, Y = np.meshgrid(x, y)
        
        # Small blob
        img_small = np.exp(-(X**2 + Y**2) / (2 * 1.0**2))
        
        # Large blob
        img_large = np.exp(-(X**2 + Y**2) / (2 * 2.0**2))
        
        # With scale normalization, responses should be comparable
        log_small = filter_log_nd(img_small, sigma=1.0, 
                                 use_normalized_derivatives=True)
        log_large = filter_log_nd(img_large, sigma=2.0, 
                                 use_normalized_derivatives=True)
        
        # Peak responses should be similar with normalization
        peak_small = np.max(np.abs(log_small))
        peak_large = np.max(np.abs(log_large))
        
        # Should be within a factor of 2
        ratio = peak_small / peak_large if peak_large > 0 else 1.0
        assert 0.5 < ratio < 2.0
    
    def test_constant_image(self):
        """Test with constant image."""
        img = np.ones((40, 40)) * 5.0
        log_response = filter_log_nd(img, sigma=2.0)
        
        # LoG of constant should be near zero
        assert np.allclose(log_response, 0, atol=1e-6)
    
    def test_invalid_sigma(self):
        """Test with invalid sigma values."""
        img = np.random.rand(40, 40)
        
        with pytest.raises(ValueError, match="sigma must be a positive"):
            filter_log_nd(img, sigma=0)
        
        with pytest.raises(ValueError, match="sigma must be a positive"):
            filter_log_nd(img, sigma=-1.0)
    
    def test_invalid_spacing(self):
        """Test with invalid spacing."""
        img = np.random.rand(40, 40)
        
        # Wrong number of elements
        with pytest.raises(ValueError, match="spacing must be scalar"):
            filter_log_nd(img, sigma=2.0, spacing=(1.0, 2.0, 3.0))
    
    def test_invalid_border_condition(self):
        """Test with invalid border condition."""
        img = np.random.rand(40, 40)
        
        with pytest.raises(ValueError, match="Unknown border condition"):
            filter_log_nd(img, sigma=2.0, border_condition='invalid')
    
    def test_invalid_input_type(self):
        """Test with invalid input type."""
        with pytest.raises(TypeError, match="must be a numpy array"):
            filter_log_nd([1, 2, 3], sigma=2.0)
    
    def test_small_sigma(self):
        """Test with small sigma."""
        img = np.random.rand(50, 50)
        log_response = filter_log_nd(img, sigma=0.5)
        
        assert log_response.shape == img.shape
    
    def test_large_sigma(self):
        """Test with large sigma."""
        img = np.random.rand(50, 50)
        log_response = filter_log_nd(img, sigma=5.0)
        
        assert log_response.shape == img.shape
    
    def test_3d_blob_detection(self):
        """Test 3D blob detection."""
        # Create 3D Gaussian blob
        vol = np.zeros((40, 40, 30))
        x = np.linspace(-3, 3, 40)
        y = np.linspace(-3, 3, 40)
        z = np.linspace(-3, 3, 30)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        vol = np.exp(-(X**2 + Y**2 + Z**2) / 2)
        
        log_response = filter_log_nd(vol, sigma=1.0, 
                                     use_normalized_derivatives=True)
        
        # Center should have strong response
        center_response = np.abs(log_response[20, 20, 15])
        assert center_response > 0
    
    def test_spacing_effect_on_sigma(self):
        """Test that spacing correctly adjusts sigma."""
        img = np.random.rand(50, 50)
        
        # Same sigma, different spacing should give different results
        log_1 = filter_log_nd(img, sigma=2.0, spacing=1.0)
        log_2 = filter_log_nd(img, sigma=2.0, spacing=2.0)
        
        # Should be different (sigma is adjusted by spacing)
        assert not np.allclose(log_1, log_2)
    
    def test_reproducibility(self):
        """Test that results are reproducible."""
        img = np.random.rand(40, 40)
        
        log_1 = filter_log_nd(img, sigma=2.0)
        log_2 = filter_log_nd(img, sigma=2.0)
        
        np.testing.assert_array_equal(log_1, log_2)
    
    def test_zero_image(self):
        """Test with zero image."""
        img = np.zeros((40, 40))
        log_response = filter_log_nd(img, sigma=2.0)
        
        # Should remain zero
        assert np.allclose(log_response, 0, atol=1e-10)
    
    def test_kernel_size_scales_with_sigma(self):
        """Test that kernel size increases with sigma."""
        img = np.random.rand(50, 50)
        
        _, kernel_small = filter_log_nd_with_kernel(img, sigma=1.0)
        _, kernel_large = filter_log_nd_with_kernel(img, sigma=3.0)
        
        # Larger sigma should produce larger kernel
        assert kernel_large.size > kernel_small.size
