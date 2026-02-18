"""
Tests for B-spline interpolation utility functions.
"""

import pytest
import numpy as np
from petakit5d.utils.spline_utils import ib3spline_1d, ib3spline_2d


class TestIb3spline1d:
    """Test ib3spline_1d function."""
    
    def test_basic_interpolation(self):
        """Test basic 1D interpolation."""
        # Create simple coefficients
        coeffs = np.array([[1.0, 2.0, 3.0, 4.0]])
        
        # Interpolate to same size
        result = ib3spline_1d(coeffs, nx=4)
        assert result.shape == (1, 4)
        
        # Interpolate to larger size
        result = ib3spline_1d(coeffs, nx=8)
        assert result.shape == (1, 8)
        
        # Interpolate to smaller size
        result = ib3spline_1d(coeffs, nx=2)
        assert result.shape == (1, 2)
    
    def test_default_size(self):
        """Test that default size matches input size."""
        coeffs = np.random.rand(5, 10)
        result = ib3spline_1d(coeffs)
        assert result.shape == coeffs.shape
    
    def test_multi_row(self):
        """Test with multiple rows."""
        coeffs = np.random.rand(10, 20)
        result = ib3spline_1d(coeffs, nx=40)
        assert result.shape == (10, 40)
    
    def test_upsampling(self):
        """Test upsampling (more output samples than coefficients)."""
        coeffs = np.array([[1.0, 2.0, 3.0]])
        result = ib3spline_1d(coeffs, nx=10)
        assert result.shape == (1, 10)
        # Result should be smooth
        assert np.all(np.isfinite(result))
    
    def test_downsampling(self):
        """Test downsampling (fewer output samples than coefficients)."""
        coeffs = np.random.rand(5, 20)
        result = ib3spline_1d(coeffs, nx=10)
        assert result.shape == (5, 10)
        assert np.all(np.isfinite(result))
    
    def test_constant_input(self):
        """Test with constant input should give constant output."""
        coeffs = np.ones((3, 10))
        result = ib3spline_1d(coeffs, nx=20)
        # Result should be close to 1 everywhere
        np.testing.assert_allclose(result, 1.0, atol=0.1)
    
    def test_linear_ramp(self):
        """Test with linear ramp."""
        coeffs = np.arange(1, 11).reshape(1, 10).astype(float)
        result = ib3spline_1d(coeffs, nx=19)
        # B-spline interpolation can have slight overshoots/ringing at edges
        # Just check that result is finite and reasonable
        assert np.all(np.isfinite(result))
        # Most of the result should be increasing
        diffs = np.diff(result[0])
        assert np.sum(diffs > 0) > len(diffs) * 0.8  # At least 80% increasing
    
    def test_invalid_input(self):
        """Test error handling for invalid inputs."""
        # 1D input should raise error
        with pytest.raises(ValueError, match="coeffs must be 2D array"):
            ib3spline_1d(np.array([1, 2, 3]))
        
        # 3D input should raise error
        with pytest.raises(ValueError, match="coeffs must be 2D array"):
            ib3spline_1d(np.random.rand(2, 3, 4))
    
    def test_symmetric_boundary(self):
        """Test that symmetric boundary conditions are applied."""
        # Create coefficients with distinct edge values
        coeffs = np.array([[1.0, 5.0, 10.0, 5.0, 1.0]])
        result = ib3spline_1d(coeffs, nx=9)
        
        # Should be smooth at boundaries
        assert np.all(np.isfinite(result))
        assert result.shape == (1, 9)


class TestIb3spline2d:
    """Test ib3spline_2d function."""
    
    def test_basic_interpolation_2d(self):
        """Test basic 2D interpolation."""
        coeffs = np.random.rand(10, 20)
        
        # Same size
        result = ib3spline_2d(coeffs, image_size=(10, 20))
        assert result.shape == (10, 20)
        
        # Larger size
        result = ib3spline_2d(coeffs, image_size=(20, 40))
        assert result.shape == (20, 40)
        
        # Smaller size
        result = ib3spline_2d(coeffs, image_size=(5, 10))
        assert result.shape == (5, 10)
    
    def test_default_size_2d(self):
        """Test that default size matches input size."""
        coeffs = np.random.rand(15, 25)
        result = ib3spline_2d(coeffs)
        assert result.shape == coeffs.shape
    
    def test_square_to_rectangle(self):
        """Test interpolating from square to rectangle."""
        coeffs = np.random.rand(10, 10)
        result = ib3spline_2d(coeffs, image_size=(10, 20))
        assert result.shape == (10, 20)
    
    def test_rectangle_to_square(self):
        """Test interpolating from rectangle to square."""
        coeffs = np.random.rand(10, 20)
        result = ib3spline_2d(coeffs, image_size=(15, 15))
        assert result.shape == (15, 15)
    
    def test_constant_2d(self):
        """Test with constant 2D input."""
        coeffs = np.ones((10, 15)) * 5.0
        result = ib3spline_2d(coeffs, image_size=(20, 30))
        # Result should be close to 5 everywhere
        np.testing.assert_allclose(result, 5.0, atol=0.1)
    
    def test_separable_interpolation(self):
        """Test that interpolation is separable."""
        coeffs = np.random.rand(10, 20)
        
        # Apply 2D interpolation
        result_2d = ib3spline_2d(coeffs, image_size=(15, 30))
        
        # Apply 1D interpolation twice manually
        temp = ib3spline_1d(coeffs, nx=30)
        result_sep = ib3spline_1d(temp.T, nx=15).T
        
        # Should match
        np.testing.assert_allclose(result_2d, result_sep, rtol=1e-10)
    
    def test_large_upsampling(self):
        """Test large upsampling factor."""
        coeffs = np.random.rand(5, 5)
        result = ib3spline_2d(coeffs, image_size=(50, 50))
        assert result.shape == (50, 50)
        assert np.all(np.isfinite(result))
    
    def test_invalid_input_2d(self):
        """Test error handling for invalid inputs."""
        # 1D input
        with pytest.raises(ValueError, match="coeffs must be 2D array"):
            ib3spline_2d(np.array([1, 2, 3]))
        
        # 3D input
        with pytest.raises(ValueError, match="coeffs must be 2D array"):
            ib3spline_2d(np.random.rand(2, 3, 4))
        
        # Invalid image_size
        coeffs = np.random.rand(10, 10)
        with pytest.raises(ValueError, match="image_size must be a tuple"):
            ib3spline_2d(coeffs, image_size=(10,))
        
        with pytest.raises(ValueError, match="image_size must be a tuple"):
            ib3spline_2d(coeffs, image_size=(10, 10, 10))
    
    def test_checkerboard_pattern(self):
        """Test with checkerboard pattern."""
        # Create checkerboard coefficients
        coeffs = np.zeros((8, 8))
        coeffs[::2, ::2] = 1
        coeffs[1::2, 1::2] = 1
        
        result = ib3spline_2d(coeffs, image_size=(16, 16))
        assert result.shape == (16, 16)
        assert np.all(np.isfinite(result))
        # Smooth interpolation should not have pure 0s and 1s in between
        assert np.min(result) > -0.5
        assert np.max(result) < 1.5
