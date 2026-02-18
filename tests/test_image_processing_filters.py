"""
Unit tests for image processing filter utilities.
"""

import pytest
import numpy as np
from petakit5d.image_processing.filters import filter_gauss_2d, filter_gauss_3d


class TestFilterGauss2D:
    """Test cases for filter_gauss_2d function."""
    
    def test_basic_filtering(self):
        """Test basic 2D Gaussian filtering."""
        img = np.random.rand(50, 50)
        filtered, kernel = filter_gauss_2d(img, sigma=2.0)
        
        assert filtered.shape == img.shape
        assert kernel is not None
        assert kernel.ndim == 2
    
    def test_smoothing_effect(self):
        """Test that filtering smooths the image."""
        # Create image with sharp edges
        img = np.zeros((50, 50))
        img[20:30, 20:30] = 1.0
        
        filtered, _ = filter_gauss_2d(img, sigma=2.0)
        
        # Filtered image should have smoother transitions
        # Check that corners are no longer exactly 1 or 0
        assert 0 < filtered[20, 20] < 1
    
    def test_different_border_conditions(self):
        """Test different border handling modes."""
        img = np.random.rand(20, 20)
        
        for mode in ['reflect', 'constant', 'nearest', 'wrap']:
            filtered, _ = filter_gauss_2d(img, sigma=1.0, border_condition=mode)
            assert filtered.shape == img.shape
    
    def test_small_sigma(self):
        """Test with small sigma value."""
        img = np.random.rand(30, 30)
        filtered, _ = filter_gauss_2d(img, sigma=0.5)
        
        # Small sigma should result in some smoothing but retain structure
        # Check that filtered values are similar but not identical
        diff = np.abs(filtered - img).mean()
        assert diff < 0.15  # Mean difference should be reasonably small
        assert filtered.shape == img.shape


class TestFilterGauss3D:
    """Test cases for filter_gauss_3d function."""
    
    def test_isotropic_filtering(self):
        """Test 3D filtering with isotropic sigma."""
        vol = np.random.rand(30, 30, 20)
        filtered = filter_gauss_3d(vol, sigma=2.0)
        
        assert filtered.shape == vol.shape
    
    def test_anisotropic_filtering(self):
        """Test 3D filtering with anisotropic sigma."""
        vol = np.random.rand(30, 30, 20)
        filtered = filter_gauss_3d(vol, sigma=(2.0, 1.0))
        
        assert filtered.shape == vol.shape
    
    def test_smoothing_effect_3d(self):
        """Test that 3D filtering smooths the volume."""
        # Create volume with sharp cube
        vol = np.zeros((40, 40, 30))
        vol[15:25, 15:25, 10:20] = 1.0
        
        filtered = filter_gauss_3d(vol, sigma=2.0)
        
        # Filtered volume should have smoother transitions
        assert 0 < filtered[15, 15, 10] < 1
    
    def test_different_border_conditions_3d(self):
        """Test different border handling modes for 3D."""
        vol = np.random.rand(20, 20, 15)
        
        for mode in ['reflect', 'constant', 'nearest']:
            filtered = filter_gauss_3d(vol, sigma=1.0, border_condition=mode)
            assert filtered.shape == vol.shape
