"""
Tests for gradient filtering functions.
"""

import numpy as np
import pytest
from petakit5d.image_processing.gradient import (
    gradient_filter_gauss_2d,
    gradient_filter_gauss_3d
)


class TestGradientFilterGauss2D:
    """Tests for gradient_filter_gauss_2d function."""
    
    def test_basic_functionality(self):
        """Test basic gradient computation."""
        img = np.random.rand(50, 50) * 255
        dx, dy = gradient_filter_gauss_2d(img, sigma=2.0)
        
        assert dx.shape == img.shape
        assert dy.shape == img.shape
        assert dx.dtype in [np.float32, np.float64]
        assert dy.dtype in [np.float32, np.float64]
    
    def test_gradient_of_ramp(self):
        """Test gradient of a linear ramp."""
        # Create horizontal ramp
        img = np.tile(np.arange(50), (50, 1)).astype(float)
        dx, dy = gradient_filter_gauss_2d(img, sigma=1.0)
        
        # Horizontal gradient should be approximately constant and positive
        assert np.mean(dx) > 0
        # Vertical gradient should be near zero
        assert np.abs(np.mean(dy)) < 0.1
    
    def test_border_conditions(self):
        """Test different border conditions."""
        img = np.random.rand(30, 30)
        
        # Test different border modes
        for mode in ['symmetric', 'replicate', 'constant']:
            dx, dy = gradient_filter_gauss_2d(img, sigma=1.5, border_condition=mode)
            assert dx.shape == img.shape
            assert dy.shape == img.shape
            assert not np.any(np.isnan(dx))
            assert not np.any(np.isnan(dy))
    
    def test_small_sigma(self):
        """Test with small sigma."""
        img = np.random.rand(40, 40)
        dx, dy = gradient_filter_gauss_2d(img, sigma=0.5)
        
        assert dx.shape == img.shape
        assert dy.shape == img.shape
    
    def test_large_sigma(self):
        """Test with large sigma."""
        img = np.random.rand(100, 100)
        dx, dy = gradient_filter_gauss_2d(img, sigma=5.0)
        
        assert dx.shape == img.shape
        assert dy.shape == img.shape
    
    def test_gradient_magnitude(self):
        """Test that gradient magnitude is computed correctly."""
        # Create a simple circular gradient
        y, x = np.ogrid[-25:25, -25:25]
        img = np.sqrt(x**2 + y**2).astype(float)
        
        dx, dy = gradient_filter_gauss_2d(img, sigma=2.0)
        grad_mag = np.sqrt(dx**2 + dy**2)
        
        # Gradient magnitude should be positive everywhere
        assert np.all(grad_mag >= 0)
        # Should have reasonable values
        assert grad_mag.max() > 0
    
    def test_dimension_error(self):
        """Test that 1D or 3D inputs raise errors."""
        with pytest.raises(ValueError, match="2-dimensional"):
            gradient_filter_gauss_2d(np.random.rand(50), sigma=1.0)
        
        with pytest.raises(ValueError, match="2-dimensional"):
            gradient_filter_gauss_2d(np.random.rand(50, 50, 50), sigma=1.0)


class TestGradientFilterGauss3D:
    """Tests for gradient_filter_gauss_3d function."""
    
    def test_basic_functionality(self):
        """Test basic 3D gradient computation."""
        vol = np.random.rand(20, 20, 20) * 255
        dx, dy, dz = gradient_filter_gauss_3d(vol, sigma=1.5)
        
        assert dx.shape == vol.shape
        assert dy.shape == vol.shape
        assert dz.shape == vol.shape
    
    def test_anisotropic_sigma(self):
        """Test with different sigma for each dimension."""
        vol = np.random.rand(30, 30, 30)
        dx, dy, dz = gradient_filter_gauss_3d(vol, sigma=(1.0, 2.0, 1.5))
        
        assert dx.shape == vol.shape
        assert dy.shape == vol.shape
        assert dz.shape == vol.shape
    
    def test_gradient_of_3d_ramp(self):
        """Test gradient of a 3D linear ramp."""
        # Create ramp in X direction
        vol = np.tile(np.arange(20), (20, 20, 1)).transpose(1, 2, 0).astype(float)
        dx, dy, dz = gradient_filter_gauss_3d(vol, sigma=1.0)
        
        # X gradient should be positive
        assert np.mean(dx) > 0
        # Y and Z gradients should be near zero
        assert np.abs(np.mean(dy)) < 0.1
        assert np.abs(np.mean(dz)) < 0.1
    
    def test_dimension_error(self):
        """Test that non-3D inputs raise errors."""
        with pytest.raises(ValueError, match="3-dimensional"):
            gradient_filter_gauss_3d(np.random.rand(50, 50), sigma=1.0)
    
    def test_invalid_sigma_length(self):
        """Test that invalid sigma tuple raises error."""
        vol = np.random.rand(20, 20, 20)
        with pytest.raises(ValueError, match="3 values"):
            gradient_filter_gauss_3d(vol, sigma=(1.0, 2.0))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
