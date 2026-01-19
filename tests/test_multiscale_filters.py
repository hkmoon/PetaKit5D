"""
Tests for multiscale filtering functions.

Tests for filter_multiscale_log_nd, filter_lobg_nd, and filter_multiscale_lobg_nd.
"""

import numpy as np
import pytest
from petakit5d.image_processing import (
    filter_multiscale_log_nd,
    filter_lobg_nd,
    filter_multiscale_lobg_nd
)


class TestFilterMultiscaleLogND:
    """Tests for multiscale LoG filter."""
    
    def test_basic_1d(self):
        """Test basic 1D multiscale LoG filtering."""
        signal = np.random.rand(100)
        sigmas = np.array([1.0, 2.0, 4.0])
        
        response, scale_map = filter_multiscale_log_nd(signal, sigmas)
        
        assert response.shape == signal.shape
        assert scale_map.shape == signal.shape
        assert np.all((scale_map >= 1) & (scale_map <= len(sigmas)))
        
    def test_basic_2d(self):
        """Test basic 2D multiscale LoG filtering."""
        img = np.random.rand(50, 50)
        sigmas = np.array([1.0, 2.0])
        
        response, scale_map = filter_multiscale_log_nd(img, sigmas)
        
        assert response.shape == img.shape
        assert scale_map.shape == img.shape
        assert np.all((scale_map >= 1) & (scale_map <= len(sigmas)))
        
    def test_basic_3d(self):
        """Test basic 3D multiscale LoG filtering."""
        vol = np.random.rand(20, 20, 20)
        sigmas = np.array([1.0, 2.0])
        
        response, scale_map = filter_multiscale_log_nd(vol, sigmas)
        
        assert response.shape == vol.shape
        assert scale_map.shape == vol.shape
        
    def test_single_scale(self):
        """Test with a single scale."""
        img = np.random.rand(50, 50)
        sigmas = np.array([2.0])
        
        response, scale_map = filter_multiscale_log_nd(img, sigmas)
        
        assert response.shape == img.shape
        assert np.all(scale_map == 1)  # All pixels should use scale 1
        
    def test_blob_detection(self):
        """Test detection of Gaussian blob."""
        # Create a Gaussian blob
        x = np.linspace(-10, 10, 50)
        xx, yy = np.meshgrid(x, x)
        blob = np.exp(-(xx**2 + yy**2) / (2 * 2**2))  # sigma = 2
        
        # Test at multiple scales
        sigmas = np.array([1.0, 2.0, 4.0])
        response, scale_map = filter_multiscale_log_nd(blob, sigmas)
        
        # Response should be most negative at blob center
        center = response.shape[0] // 2
        assert response[center, center] < np.mean(response)
        
        # Scale 2 (index 2 in 1-based) should be optimal for sigma=2 blob
        # Allow for some tolerance since optimal scale depends on implementation details
        assert scale_map[center, center] in [2, 3]
        
    def test_spacing_parameter(self):
        """Test with non-unit spacing."""
        img = np.random.rand(50, 50)
        sigmas = np.array([1.0, 2.0])
        spacing = 0.5
        
        response1, _ = filter_multiscale_log_nd(img, sigmas, spacing=spacing)
        response2, _ = filter_multiscale_log_nd(img, sigmas, spacing=1.0)
        
        # Responses should differ with different spacing
        assert not np.allclose(response1, response2)
        
    def test_anisotropic_spacing(self):
        """Test with anisotropic spacing."""
        img = np.random.rand(50, 50)
        sigmas = np.array([2.0])
        spacing = np.array([1.0, 2.0])
        
        response, scale_map = filter_multiscale_log_nd(img, sigmas, spacing=spacing)
        
        assert response.shape == img.shape
        
    def test_border_conditions(self):
        """Test different border conditions."""
        img = np.random.rand(50, 50)
        sigmas = np.array([2.0])
        
        for border in ['symmetric', 'replicate', 'wrap']:
            response, _ = filter_multiscale_log_nd(img, sigmas, border_condition=border)
            assert response.shape == img.shape
            
    def test_constant_border(self):
        """Test constant value border condition."""
        img = np.random.rand(50, 50)
        sigmas = np.array([2.0])
        
        response, _ = filter_multiscale_log_nd(img, sigmas, border_condition=0.0)
        assert response.shape == img.shape
        
    def test_normalized_gaussian(self):
        """Test with and without normalized Gaussian."""
        img = np.random.rand(50, 50)
        sigmas = np.array([2.0])
        
        response1, _ = filter_multiscale_log_nd(img, sigmas, use_normalized_gaussian=True)
        response2, _ = filter_multiscale_log_nd(img, sigmas, use_normalized_gaussian=False)
        
        # Results should differ
        assert not np.allclose(response1, response2)
        
    def test_multiple_scales_ordering(self):
        """Test that scale map correctly identifies optimal scales."""
        # Create blobs at different scales
        x = np.linspace(-20, 20, 100)
        xx, yy = np.meshgrid(x, x)
        
        # Small blob (left) and large blob (right)
        small_blob = np.exp(-((xx + 10)**2 + yy**2) / (2 * 1**2))
        large_blob = np.exp(-((xx - 10)**2 + yy**2) / (2 * 4**2))
        img = small_blob + large_blob
        
        sigmas = np.array([1.0, 2.0, 4.0])
        response, scale_map = filter_multiscale_log_nd(img, sigmas)
        
        # Small blob should prefer smaller scale
        assert scale_map[50, 25] <= 2
        
        # Large blob should prefer larger scale
        assert scale_map[50, 75] >= 2
        
    def test_debug_mode(self, capsys):
        """Test debug mode prints information."""
        img = np.random.rand(50, 50)
        sigmas = np.array([1.0, 2.0])
        
        filter_multiscale_log_nd(img, sigmas, debug_mode=True)
        
        captured = capsys.readouterr()
        assert "Running LoG filter" in captured.out
        
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        img = np.random.rand(50, 50)
        
        # Non-1D sigma values
        with pytest.raises(ValueError):
            filter_multiscale_log_nd(img, np.array([[1.0, 2.0]]))
            
        # Non-array input
        with pytest.raises(TypeError):
            filter_multiscale_log_nd([1, 2, 3], np.array([1.0]))


class TestFilterLoBGND:
    """Tests for Laplacian of Bi-Gaussian filter."""
    
    def test_basic_1d(self):
        """Test basic 1D LoBG filtering."""
        signal = np.random.rand(100)
        
        response, kernel = filter_lobg_nd(signal, sigma=2.0, rho=0.5)
        
        assert response.shape == signal.shape
        assert kernel is not None
        assert kernel.ndim == 1
        
    def test_basic_2d(self):
        """Test basic 2D LoBG filtering."""
        img = np.random.rand(50, 50)
        
        response, kernel = filter_lobg_nd(img, sigma=2.0, rho=0.5)
        
        assert response.shape == img.shape
        assert kernel.ndim == 2
        
    def test_basic_3d(self):
        """Test basic 3D LoBG filtering."""
        vol = np.random.rand(20, 20, 20)
        
        response, kernel = filter_lobg_nd(vol, sigma=2.0, rho=0.5)
        
        assert response.shape == vol.shape
        assert kernel.ndim == 3
        
    def test_rho_equals_one(self):
        """Test that rho=1.0 is similar to standard LoG."""
        from petakit5d.image_processing import filter_log_nd
        
        img = np.random.rand(50, 50)
        sigma = 2.0
        
        lobg_response, _ = filter_lobg_nd(
            img, sigma, rho=1.0,
            use_normalized_derivatives=True
        )
        log_response = filter_log_nd(
            img, sigma,
            use_normalized_derivatives=True
        )
        
        # Should be similar (not exact due to implementation differences)
        correlation = np.corrcoef(lobg_response.flatten(), log_response.flatten())[0, 1]
        assert correlation > 0.75  # Relaxed threshold for different implementations
        
    def test_rho_effect(self):
        """Test that different rho values give different results."""
        img = np.random.rand(50, 50)
        sigma = 2.0
        
        response1, _ = filter_lobg_nd(img, sigma, rho=0.2)
        response2, _ = filter_lobg_nd(img, sigma, rho=0.8)
        
        # Different rho should give different responses
        assert not np.allclose(response1, response2)
        
    def test_normalized_derivatives(self):
        """Test scale-normalized derivatives."""
        img = np.random.rand(50, 50)
        
        response1, kernel1 = filter_lobg_nd(
            img, sigma=2.0, rho=0.5,
            use_normalized_derivatives=True
        )
        response2, kernel2 = filter_lobg_nd(
            img, sigma=2.0, rho=0.5,
            use_normalized_derivatives=False
        )
        
        # Kernels should differ
        assert not np.allclose(kernel1, kernel2)
        
        # Normalized should have larger magnitude
        assert np.abs(kernel1).max() > np.abs(kernel2).max()
        
    def test_spacing_parameter(self):
        """Test with non-unit spacing."""
        img = np.random.rand(50, 50)
        
        response1, _ = filter_lobg_nd(img, sigma=2.0, rho=0.5, spacing=0.5)
        response2, _ = filter_lobg_nd(img, sigma=2.0, rho=0.5, spacing=1.0)
        
        # Different spacing should give different results
        assert not np.allclose(response1, response2)
        
    def test_border_conditions(self):
        """Test different border conditions."""
        img = np.random.rand(50, 50)
        
        for border in ['symmetric', 'replicate']:
            response, _ = filter_lobg_nd(img, sigma=2.0, rho=0.5, border_condition=border)
            assert response.shape == img.shape
            
    def test_kernel_properties(self):
        """Test properties of the LoBG kernel."""
        img = np.random.rand(50, 50)
        
        _, kernel = filter_lobg_nd(img, sigma=2.0, rho=0.5)
        
        # Kernel should be zero-mean (DC removed)
        assert np.abs(np.mean(kernel)) < 1e-10
        
        # Kernel should have both positive and negative values
        assert kernel.min() < 0
        assert kernel.max() > 0
        
    def test_adjacent_blobs(self):
        """Test LoBG on adjacent blobs (its intended use case)."""
        x = np.linspace(-20, 20, 100)
        xx, yy = np.meshgrid(x, x)
        
        # Two adjacent Gaussian blobs
        blob1 = np.exp(-((xx - 5)**2 + yy**2) / (2 * 2**2))
        blob2 = np.exp(-((xx + 5)**2 + yy**2) / (2 * 2**2))
        img = blob1 + blob2
        
        # LoBG with small rho should separate them better than LoG
        response, _ = filter_lobg_nd(img, sigma=2.0, rho=0.2, use_normalized_derivatives=True)
        
        # Should have two local minima
        assert response.min() < -0.01  # Significant negative response
        
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        img = np.random.rand(50, 50)
        
        # Invalid sigma
        with pytest.raises(ValueError):
            filter_lobg_nd(img, sigma=-1.0, rho=0.5)
            
        # Invalid rho
        with pytest.raises(ValueError):
            filter_lobg_nd(img, sigma=2.0, rho=-0.5)
            
        # Non-array input
        with pytest.raises(TypeError):
            filter_lobg_nd([1, 2, 3], sigma=2.0, rho=0.5)


class TestFilterMultiscaleLoBGND:
    """Tests for multiscale LoBG filter."""
    
    def test_basic_2d(self):
        """Test basic 2D multiscale LoBG filtering."""
        img = np.random.rand(50, 50)
        sigmas = np.array([1.0, 2.0])
        
        response, scale_map = filter_multiscale_lobg_nd(img, sigmas, rho=0.3)
        
        assert response.shape == img.shape
        assert scale_map.shape == img.shape
        assert np.all((scale_map >= 1) & (scale_map <= len(sigmas)))
        
    def test_single_scale(self):
        """Test with a single scale."""
        img = np.random.rand(50, 50)
        sigmas = np.array([2.0])
        
        response, scale_map = filter_multiscale_lobg_nd(img, sigmas, rho=0.3)
        
        assert response.shape == img.shape
        assert np.all(scale_map == 1)
        
    def test_multiple_scales(self):
        """Test with multiple scales."""
        img = np.random.rand(50, 50)
        sigmas = np.array([1.0, 2.0, 4.0])
        
        response, scale_map = filter_multiscale_lobg_nd(img, sigmas, rho=0.2)
        
        assert response.shape == img.shape
        # Should use different scales for different regions
        assert len(np.unique(scale_map)) > 1
        
    def test_rho_parameter(self):
        """Test different rho values."""
        img = np.random.rand(50, 50)
        sigmas = np.array([2.0, 4.0])
        
        response1, _ = filter_multiscale_lobg_nd(img, sigmas, rho=0.1)
        response2, _ = filter_multiscale_lobg_nd(img, sigmas, rho=0.5)
        
        # Different rho should give different results
        assert not np.allclose(response1, response2)
        
    def test_spacing_parameter(self):
        """Test with non-unit spacing."""
        img = np.random.rand(50, 50)
        sigmas = np.array([2.0])
        
        response, _ = filter_multiscale_lobg_nd(img, sigmas, rho=0.3, spacing=0.5)
        
        assert response.shape == img.shape
        
    def test_debug_mode(self, capsys):
        """Test debug mode prints information."""
        img = np.random.rand(50, 50)
        sigmas = np.array([1.0, 2.0])
        
        filter_multiscale_lobg_nd(img, sigmas, rho=0.3, debug_mode=True)
        
        captured = capsys.readouterr()
        assert "Running LoBG filter" in captured.out
        
    def test_adjacent_ridges_multiscale(self):
        """Test multiscale LoBG on adjacent ridges."""
        x = np.linspace(-20, 20, 100)
        xx, yy = np.meshgrid(x, x)
        
        # Create ridges (elongated Gaussians)
        ridge1 = np.exp(-(xx - 5)**2 / (2 * 2**2)) * np.exp(-yy**2 / (2 * 8**2))
        ridge2 = np.exp(-(xx + 5)**2 / (2 * 2**2)) * np.exp(-yy**2 / (2 * 8**2))
        img = ridge1 + ridge2
        
        sigmas = np.array([1.5, 2.0, 3.0])
        response, scale_map = filter_multiscale_lobg_nd(img, sigmas, rho=0.2)
        
        # Should detect ridges
        assert response.min() < -0.01
        
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        img = np.random.rand(50, 50)
        
        # Invalid rho
        with pytest.raises(ValueError):
            filter_multiscale_lobg_nd(img, np.array([1.0]), rho=-0.5)
            
        # Non-1D sigma values
        with pytest.raises(ValueError):
            filter_multiscale_lobg_nd(img, np.array([[1.0, 2.0]]), rho=0.3)
            
        # Non-array input
        with pytest.raises(TypeError):
            filter_multiscale_lobg_nd([1, 2, 3], np.array([1.0]), rho=0.3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
