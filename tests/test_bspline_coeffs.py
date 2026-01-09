"""
Tests for B-spline coefficient computation functions.
"""

import numpy as np
import pytest
from petakit5d.image_processing.bspline_coeffs import b3spline_1d, b3spline_2d


class TestB3spline1D:
    """Tests for b3spline_1d function."""
    
    def test_basic_computation_mirror(self):
        """Test basic B-spline coefficient computation with mirror boundary."""
        img = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        coeffs = b3spline_1d(img, boundary='mirror')
        
        assert coeffs.shape == img.shape
        assert coeffs.dtype == np.float64
        # Coefficients should be similar to input for smooth data
        assert np.allclose(coeffs, img, atol=1.0)
    
    def test_basic_computation_periodic(self):
        """Test basic B-spline coefficient computation with periodic boundary."""
        img = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        coeffs = b3spline_1d(img, boundary='periodic')
        
        assert coeffs.shape == img.shape
        assert coeffs.dtype == np.float64
    
    def test_1d_input(self):
        """Test with 1D input array."""
        img_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        coeffs = b3spline_1d(img_1d, boundary='mirror')
        
        assert coeffs.shape == img_1d.shape
        assert coeffs.ndim == 1
    
    def test_2d_input(self):
        """Test with 2D input array."""
        img_2d = np.random.rand(10, 20)
        coeffs = b3spline_1d(img_2d, boundary='mirror')
        
        assert coeffs.shape == img_2d.shape
        assert coeffs.ndim == 2
    
    def test_constant_image_mirror(self):
        """Test with constant image (mirror boundary)."""
        img = np.ones((5, 10))
        coeffs = b3spline_1d(img, boundary='mirror')
        
        # For constant input, coefficients should also be constant
        assert np.allclose(coeffs, 1.0, rtol=1e-10)
    
    def test_constant_image_periodic(self):
        """Test with constant image (periodic boundary)."""
        img = np.ones((5, 10))
        coeffs = b3spline_1d(img, boundary='periodic')
        
        # For constant input, coefficients should also be constant (allow small numerical error)
        assert np.allclose(coeffs, 1.0, rtol=1e-5, atol=1e-5)
    
    def test_invalid_boundary(self):
        """Test with invalid boundary condition."""
        img = np.random.rand(5, 10)
        with pytest.raises(ValueError, match="Boundary must be"):
            b3spline_1d(img, boundary='invalid')
    
    def test_linear_ramp(self):
        """Test with linear ramp."""
        img = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])
        coeffs = b3spline_1d(img, boundary='mirror')
        
        # B-spline coefficients will differ from input values
        # Just verify shape and type
        assert coeffs.shape == img.shape
        assert coeffs.dtype == np.float64
    
    def test_output_dtype(self):
        """Test that output is always float64."""
        img_int = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
        coeffs = b3spline_1d(img_int, boundary='mirror')
        
        assert coeffs.dtype == np.float64
    
    def test_edge_values_different(self):
        """Test that edge handling differs between boundaries."""
        img = np.random.rand(3, 20)
        coeffs_mirror = b3spline_1d(img, boundary='mirror')
        coeffs_periodic = b3spline_1d(img, boundary='periodic')
        
        # Results should differ at edges
        assert not np.allclose(coeffs_mirror[:, 0], coeffs_periodic[:, 0])
        assert not np.allclose(coeffs_mirror[:, -1], coeffs_periodic[:, -1])


class TestB3spline2D:
    """Tests for b3spline_2d function."""
    
    def test_basic_computation(self):
        """Test basic 2D B-spline coefficient computation."""
        img = np.random.rand(20, 30)
        coeffs = b3spline_2d(img, boundary='mirror')
        
        assert coeffs.shape == img.shape
        assert coeffs.dtype == np.float64
    
    def test_constant_image(self):
        """Test with constant 2D image."""
        img = np.ones((15, 25)) * 5.0
        coeffs = b3spline_2d(img, boundary='mirror')
        
        # For constant input, coefficients should also be constant
        assert np.allclose(coeffs, 5.0, rtol=1e-10)
    
    def test_mirror_vs_periodic(self):
        """Test difference between mirror and periodic boundaries."""
        img = np.random.rand(20, 30)
        coeffs_mirror = b3spline_2d(img, boundary='mirror')
        coeffs_periodic = b3spline_2d(img, boundary='periodic')
        
        # Results should differ
        assert not np.allclose(coeffs_mirror, coeffs_periodic)
    
    def test_separability(self):
        """Test that 2D computation is separable."""
        img = np.random.rand(15, 20)
        
        # Compute using b3spline_2d
        coeffs_2d = b3spline_2d(img, boundary='mirror')
        
        # Compute manually using b3spline_1d twice
        c1 = b3spline_1d(img, boundary='mirror')
        c2 = b3spline_1d(c1.T, boundary='mirror').T
        
        # Should match
        assert np.allclose(coeffs_2d, c2, rtol=1e-12)
    
    def test_invalid_boundary(self):
        """Test with invalid boundary condition."""
        img = np.random.rand(10, 15)
        with pytest.raises(ValueError, match="Boundary must be"):
            b3spline_2d(img, boundary='invalid')
    
    def test_invalid_input_dimension(self):
        """Test with non-2D input."""
        img_1d = np.random.rand(10)
        with pytest.raises(ValueError, match="Input must be a 2D array"):
            b3spline_2d(img_1d, boundary='mirror')
        
        img_3d = np.random.rand(5, 10, 15)
        with pytest.raises(ValueError, match="Input must be a 2D array"):
            b3spline_2d(img_3d, boundary='mirror')
    
    def test_small_image(self):
        """Test with very small image."""
        img = np.array([[1.0, 2.0], [3.0, 4.0]])
        coeffs = b3spline_2d(img, boundary='mirror')
        
        assert coeffs.shape == (2, 2)
        # B-spline coefficients computed correctly (shape and dtype verified)
        assert coeffs.dtype == np.float64
    
    def test_rectangular_image(self):
        """Test with rectangular (non-square) image."""
        img = np.random.rand(10, 50)
        coeffs = b3spline_2d(img, boundary='mirror')
        
        assert coeffs.shape == (10, 50)
    
    def test_output_dtype(self):
        """Test that output is always float64."""
        img_int = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        coeffs = b3spline_2d(img_int, boundary='mirror')
        
        assert coeffs.dtype == np.float64
    
    def test_periodic_boundary(self):
        """Test periodic boundary conditions."""
        img = np.random.rand(20, 20)
        coeffs = b3spline_2d(img, boundary='periodic')
        
        assert coeffs.shape == img.shape
        # For periodic boundary, opposite edges should have related values
        # (not exactly equal due to the nature of B-spline computation)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
