"""
Tests for fast Gaussian filtering functions.
"""

import numpy as np
import pytest
from petakit5d.image_processing.fast_gauss import fast_gauss_3d


class TestFastGauss3D:
    """Tests for fast_gauss_3d function."""
    
    def test_2d_basic(self):
        """Test basic 2D filtering."""
        img = np.random.rand(50, 50)
        sigma = 2.0
        
        result = fast_gauss_3d(img, sigma)
        
        assert result.shape == img.shape
        # Filtered image should be smoother (lower std)
        assert np.std(result) < np.std(img)
    
    def test_3d_basic(self):
        """Test basic 3D filtering."""
        img = np.random.rand(30, 30, 20)
        sigma = 1.5
        
        result = fast_gauss_3d(img, sigma)
        
        assert result.shape == img.shape
        assert np.std(result) < np.std(img)
    
    def test_anisotropic_sigma(self):
        """Test with different sigma per dimension."""
        img = np.random.rand(40, 40, 20)
        sigma = [1.0, 1.0, 2.0]
        
        result = fast_gauss_3d(img, sigma)
        
        assert result.shape == img.shape
        assert not np.array_equal(result, img)
    
    def test_no_border_correction(self):
        """Test without border correction."""
        img = np.random.rand(50, 50)
        sigma = 2.0
        
        result = fast_gauss_3d(img, sigma, correct_border=0)
        
        assert result.shape == img.shape
    
    def test_old_border_correction(self):
        """Test with old border correction method."""
        img = np.random.rand(50, 50)
        sigma = 2.0
        
        result = fast_gauss_3d(img, sigma, correct_border=2)
        
        assert result.shape == img.shape
    
    def test_with_nan_values(self):
        """Test handling of NaN values."""
        img = np.random.rand(40, 40)
        img[10:15, 10:15] = np.nan
        sigma = 2.0
        
        result = fast_gauss_3d(img, sigma)
        
        assert result.shape == img.shape
        # NaN regions should be handled
        assert not np.all(np.isnan(result))
    
    def test_custom_filter_size(self):
        """Test with custom filter size."""
        img = np.random.rand(50, 50)
        sigma = 2.0
        f_sze = 11
        
        result = fast_gauss_3d(img, sigma, f_sze=f_sze)
        
        assert result.shape == img.shape
    
    def test_empty_image_error(self):
        """Test error handling for empty image."""
        img = np.array([])
        sigma = 2.0
        
        with pytest.raises(ValueError, match="nonempty image"):
            fast_gauss_3d(img, sigma)
    
    def test_sigma_dimension_mismatch(self):
        """Test error for sigma dimension mismatch."""
        img = np.random.rand(30, 30, 20)
        sigma = [1.0, 2.0]  # Wrong length
        
        with pytest.raises(ValueError, match="Sigma must have length"):
            fast_gauss_3d(img, sigma)
    
    def test_preserves_mean(self):
        """Test that mean is approximately preserved."""
        img = np.random.rand(50, 50) + 10.0
        sigma = 1.0
        
        result = fast_gauss_3d(img, sigma)
        
        # Mean should be close (within 5%)
        assert np.abs(np.mean(result) - np.mean(img)) < 0.5
    
    def test_impulse_response(self):
        """Test impulse response (delta function)."""
        img = np.zeros((51, 51))
        img[25, 25] = 1.0
        sigma = 2.0
        
        result = fast_gauss_3d(img, sigma)
        
        # Result should be a Gaussian-like distribution
        # Peak should still be at center
        peak_loc = np.unravel_index(np.argmax(result), result.shape)
        assert peak_loc == (25, 25)
        # Total energy should be conserved approximately
        assert 0.8 < np.sum(result) < 1.2
    
    def test_3d_with_nan_and_correction(self):
        """Test 3D with NaN values and border correction."""
        img = np.random.rand(20, 20, 15)
        img[5:8, 5:8, 5:8] = np.nan
        sigma = 1.5
        
        result = fast_gauss_3d(img, sigma, correct_border=1)
        
        assert result.shape == img.shape
        # Most of the image should have valid values
        assert np.sum(~np.isnan(result)) > 0.8 * result.size
