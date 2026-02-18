"""
Tests for surface filtering functions.
"""

import numpy as np
import pytest
from petakit5d.image_processing.surface_filter import surface_filter_gauss_3d


class TestSurfaceFilterGauss3D:
    """Tests for surface_filter_gauss_3d function."""
    
    def test_basic_3d(self):
        """Test basic 3D surface filtering."""
        img = np.random.rand(30, 30, 20)
        sigma = 2.0
        
        d2x, d2y, d2z = surface_filter_gauss_3d(img, sigma)
        
        assert d2x.shape == img.shape
        assert d2y.shape == img.shape
        assert d2z.shape == img.shape
    
    def test_anisotropic_sigma(self):
        """Test with different sigma per dimension."""
        img = np.random.rand(30, 30, 20)
        sigma = [1.0, 1.0, 2.0]
        
        d2x, d2y, d2z = surface_filter_gauss_3d(img, sigma)
        
        assert d2x.shape == img.shape
        assert d2y.shape == img.shape
        assert d2z.shape == img.shape
    
    def test_sphere_detection(self):
        """Test detection of a sphere (bright surface)."""
        img = np.zeros((40, 40, 40))
        # Create a sphere
        center = (20, 20, 20)
        radius = 8
        y, x, z = np.ogrid[:40, :40, :40]
        mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2
        img[mask] = 1.0
        
        sigma = 2.0
        d2x, d2y, d2z = surface_filter_gauss_3d(img, sigma)
        
        # Surface should have strong response
        # Sum of squared responses should be significant
        response = np.sqrt(d2x**2 + d2y**2 + d2z**2)
        assert np.max(response) > 0.01
    
    def test_plane_detection(self):
        """Test detection of a plane."""
        img = np.zeros((30, 30, 20))
        img[:, :, 10:] = 1.0  # Half-space
        
        sigma = 1.5
        d2x, d2y, d2z = surface_filter_gauss_3d(img, sigma)
        
        # Z derivative should have strong response at the boundary
        assert np.max(np.abs(d2z)) > np.max(np.abs(d2x))
        assert np.max(np.abs(d2z)) > np.max(np.abs(d2y))
    
    def test_border_conditions(self):
        """Test different border conditions."""
        img = np.random.rand(20, 20, 15)
        sigma = 1.0
        
        for border in ['reflect', 'constant', 'nearest', 'mirror', 'wrap']:
            d2x, d2y, d2z = surface_filter_gauss_3d(img, sigma, border_condition=border)
            
            assert d2x.shape == img.shape
            assert d2y.shape == img.shape
            assert d2z.shape == img.shape
    
    def test_zero_mean_property(self):
        """Test that second derivatives have zero mean for constant images."""
        img = np.ones((30, 30, 20)) * 5.0
        sigma = 2.0
        
        d2x, d2y, d2z = surface_filter_gauss_3d(img, sigma)
        
        # For constant image, derivatives should be near zero
        assert np.abs(np.mean(d2x)) < 0.01
        assert np.abs(np.mean(d2y)) < 0.01
        assert np.abs(np.mean(d2z)) < 0.01
    
    def test_edge_response(self):
        """Test response to edges."""
        img = np.zeros((30, 30, 20))
        img[10:20, :, :] = 1.0  # Step edge in Y direction
        
        sigma = 1.5
        d2x, d2y, d2z = surface_filter_gauss_3d(img, sigma)
        
        # Y derivative should have strongest response
        assert np.max(np.abs(d2y)) > np.max(np.abs(d2x))
        assert np.max(np.abs(d2y)) > np.max(np.abs(d2z))
    
    def test_not_3d_error(self):
        """Test error handling for non-3D input."""
        img_2d = np.random.rand(50, 50)
        sigma = 2.0
        
        with pytest.raises(ValueError, match="must be 3D"):
            surface_filter_gauss_3d(img_2d, sigma)
    
    def test_sigma_length_error(self):
        """Test error for wrong sigma length."""
        img = np.random.rand(30, 30, 20)
        sigma = [1.0, 2.0]  # Wrong length
        
        with pytest.raises(ValueError, match="must have 3 elements"):
            surface_filter_gauss_3d(img, sigma)
    
    def test_gradient_magnitude(self):
        """Test that gradient magnitude increases with sigma."""
        img = np.random.rand(25, 25, 20)
        
        # Small sigma
        d2x1, d2y1, d2z1 = surface_filter_gauss_3d(img, sigma=0.5)
        mag1 = np.sqrt(d2x1**2 + d2y1**2 + d2z1**2)
        
        # Larger sigma
        d2x2, d2y2, d2z2 = surface_filter_gauss_3d(img, sigma=2.0)
        mag2 = np.sqrt(d2x2**2 + d2y2**2 + d2z2**2)
        
        # Larger sigma should give smoother (different) response
        assert not np.allclose(mag1, mag2)
    
    def test_symmetric_border(self):
        """Test with symmetric border condition (MATLAB compatibility)."""
        img = np.random.rand(20, 20, 15)
        sigma = 1.5
        
        d2x, d2y, d2z = surface_filter_gauss_3d(img, sigma, border_condition='symmetric')
        
        assert d2x.shape == img.shape
        assert not np.any(np.isnan(d2x))
        assert not np.any(np.isnan(d2y))
        assert not np.any(np.isnan(d2z))
