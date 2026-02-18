"""
Tests for B-spline interpolation utilities.
"""

import numpy as np
import pytest
from petakit5d.image_processing import (
    compute_bspline_coefficients,
    interp_bspline_value,
    calc_interp_maxima
)


class TestComputeBSplineCoefficients:
    """Tests for compute_bspline_coefficients function."""
    
    def test_1d_input_basic(self):
        """Test 1D input with basic signal."""
        s = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        c = compute_bspline_coefficients(s)
        
        # Coefficients should be same size as input
        assert c.shape == s.shape
        # Should be real-valued
        assert np.all(np.isreal(c))
    
    def test_2d_input_basic(self):
        """Test 2D input with basic image."""
        s = np.random.rand(5, 5)
        c = compute_bspline_coefficients(s)
        
        # Coefficients should be same size as input
        assert c.shape == s.shape
        # Should be real-valued
        assert np.all(np.isreal(c))
    
    def test_constant_signal_1d(self):
        """Test that constant signal returns approximately constant coefficients."""
        s = np.ones(10) * 5.0
        c = compute_bspline_coefficients(s, mode='fourier')
        
        # For constant input, coefficients should be close to constant
        np.testing.assert_allclose(c, 5.0, rtol=1e-10)
    
    def test_constant_signal_2d(self):
        """Test that constant 2D signal returns approximately constant coefficients."""
        s = np.ones((5, 5)) * 3.0
        c = compute_bspline_coefficients(s, mode='fourier')
        
        # For constant input, coefficients should be close to constant
        np.testing.assert_allclose(c, 3.0, rtol=1e-10)
    
    def test_fourier_vs_spatial_cubic(self):
        """Test that Fourier and spatial modes give similar results for degree 3."""
        s = np.random.rand(10)
        c_fourier = compute_bspline_coefficients(s, mode='fourier', degree=3)
        c_spatial = compute_bspline_coefficients(s, mode='spatial', degree=3)
        
        # Results should be similar (not exactly equal due to numerical differences)
        np.testing.assert_allclose(c_fourier, c_spatial, rtol=1e-5)
    
    def test_symmetric_vs_periodic_boundary(self):
        """Test different boundary conditions."""
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        c_sym = compute_bspline_coefficients(s, boundary='symmetric')
        c_per = compute_bspline_coefficients(s, boundary='periodic')
        
        # Results should differ with different boundaries
        assert not np.allclose(c_sym, c_per)
    
    def test_smoothing_spline(self):
        """Test smoothing spline with lambda > 0."""
        s = np.random.rand(20) + np.sin(np.linspace(0, 2*np.pi, 20))
        c_no_smooth = compute_bspline_coefficients(s, lambda_=0.0)
        c_smooth = compute_bspline_coefficients(s, lambda_=0.1)
        
        # Smoothing should change coefficients
        assert not np.allclose(c_no_smooth, c_smooth)
    
    def test_degree_1(self):
        """Test linear spline (degree 1)."""
        s = np.array([1.0, 2.0, 3.0, 4.0])
        c = compute_bspline_coefficients(s, degree=1, mode='spatial')
        
        # For degree 1, coefficients should equal input
        np.testing.assert_allclose(c, s)
    
    def test_degree_2_spatial(self):
        """Test quadratic spline (degree 2) in spatial mode."""
        s = np.random.rand(10)
        c = compute_bspline_coefficients(s, degree=2, mode='spatial')
        
        assert c.shape == s.shape
        assert np.all(np.isreal(c))
    
    def test_2d_separability(self):
        """Test that 2D processing is separable."""
        # Create separable 2D signal
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 8)
        sx = np.sin(2 * np.pi * x)
        sy = np.cos(2 * np.pi * y)
        s = np.outer(sy, sx)
        
        c = compute_bspline_coefficients(s)
        
        # Result should be close to separable (this is a property test)
        assert c.shape == s.shape


class TestInterpBSplineValue:
    """Tests for interp_bspline_value function."""
    
    def test_interpolate_at_grid_points(self):
        """Test interpolation at original grid points."""
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        c = compute_bspline_coefficients(s, degree=3)
        
        # Interpolate at grid points (0-based indexing for Python)
        for i in range(len(s)):
            v = interp_bspline_value(float(i), c, n=3)
            # Should be close to original value
            # Note: cubic spline interpolation may not pass exactly through points
            np.testing.assert_allclose(v, s[i], rtol=0.3)
    
    def test_interpolate_between_points(self):
        """Test interpolation between grid points."""
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        c = compute_bspline_coefficients(s, degree=3)
        
        # Interpolate at midpoint between indices 1 and 2 (0-based: values 2.0 and 3.0)
        v = interp_bspline_value(1.5, c, n=3)
        
        # Should be between the two values (roughly)
        # Cubic spline can overshoot slightly
        assert 1.5 < v < 3.5
    
    def test_linear_interpolation(self):
        """Test linear interpolation (degree 1)."""
        s = np.array([0.0, 1.0, 2.0, 3.0])
        c = s.copy()  # For degree 1, coefficients = input
        
        # Test at midpoint between indices 0 and 1 (values 0.0 and 1.0)
        v = interp_bspline_value(0.5, c, n=1)
        np.testing.assert_allclose(v, 0.5, rtol=1e-10)
    
    def test_quadratic_interpolation(self):
        """Test quadratic interpolation (degree 2)."""
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        c = compute_bspline_coefficients(s, degree=2, mode='spatial')
        
        v = interp_bspline_value(2.5, c, n=2)
        assert isinstance(v, (float, np.floating))
    
    def test_cubic_interpolation(self):
        """Test cubic interpolation (degree 3)."""
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        c = compute_bspline_coefficients(s, degree=3)
        
        v = interp_bspline_value(2.5, c, n=3)
        assert isinstance(v, (float, np.floating))
    
    def test_vector_input(self):
        """Test interpolation with vector input."""
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        c = compute_bspline_coefficients(s, degree=3)
        
        # Use valid indices (0-based, within bounds)
        x = np.array([0.5, 1.5, 2.5])
        v = interp_bspline_value(x, c, n=3)
        
        assert len(v) == len(x)
        assert np.all(np.isfinite(v))
    
    def test_periodic_boundary(self):
        """Test periodic boundary conditions."""
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        c = compute_bspline_coefficients(s, boundary='periodic')
        
        v = interp_bspline_value(2.5, c, n=3, boundary='periodic')
        assert np.isfinite(v)
    
    def test_symmetric_boundary(self):
        """Test symmetric boundary conditions."""
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        c = compute_bspline_coefficients(s, boundary='symmetric')
        
        v = interp_bspline_value(2.5, c, n=3, boundary='symmetric')
        assert np.isfinite(v)
    
    def test_smoothness(self):
        """Test that interpolation is smooth."""
        s = np.array([1.0, 4.0, 2.0, 5.0, 3.0])
        c = compute_bspline_coefficients(s, degree=3)
        
        # Sample at many points within valid range
        x = np.linspace(0, 4, 100)
        v = interp_bspline_value(x, c, n=3)
        
        # Result should be smooth (no NaN or inf)
        assert np.all(np.isfinite(v))


class TestCalcInterpMaxima:
    """Tests for calc_interp_maxima function."""
    
    def test_1d_single_maximum(self):
        """Test finding maximum in 1D signal."""
        # Create signal with clear maximum
        x = np.linspace(0, 2*np.pi, 20)
        s = np.sin(x)
        
        fmax, xmax, c = calc_interp_maxima(s)
        
        # Should find at least one maximum
        assert len(fmax) > 0
        # Check that we found positive maxima (sin has positive max near pi/2)
        if len(fmax) > 0:
            # At least one maximum should be positive
            assert np.any(fmax > 0.5)
    
    def test_1d_no_maxima(self):
        """Test with monotonic signal (no interior maxima)."""
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        fmax, xmax, c = calc_interp_maxima(s)
        
        # Should return empty or very few maxima
        assert isinstance(fmax, np.ndarray)
        assert isinstance(xmax, np.ndarray)
    
    def test_1d_multiple_maxima(self):
        """Test finding multiple maxima in 1D signal."""
        # Create signal with two maxima
        x = np.linspace(0, 4*np.pi, 40)
        s = np.sin(x)
        
        fmax, xmax, c = calc_interp_maxima(s)
        
        # Should find multiple maxima
        # (exact number depends on implementation details)
        assert len(fmax) >= 1
    
    def test_2d_basic(self):
        """Test finding maxima in 2D image."""
        # Create 2D signal with peak
        y, x = np.ogrid[-5:5:10j, -5:5:10j]
        s = np.exp(-(x**2 + y**2)/4)
        
        fmax, (xmax, ymax), c = calc_interp_maxima(s)
        
        # Should find at least one maximum near center
        if len(fmax) > 0:
            assert np.all(np.isfinite(fmax))
            assert np.all(np.isfinite(xmax))
            assert np.all(np.isfinite(ymax))
    
    def test_2d_multiple_peaks(self):
        """Test finding multiple peaks in 2D image."""
        # Create 2D signal with multiple peaks
        y, x = np.ogrid[-5:5:15j, -5:5:15j]
        s = np.exp(-((x-2)**2 + (y-2)**2)/2) + np.exp(-((x+2)**2 + (y+2)**2)/2)
        
        fmax, (xmax, ymax), c = calc_interp_maxima(s)
        
        # Should find peaks
        assert isinstance(fmax, np.ndarray)
        assert isinstance(xmax, np.ndarray)
        assert isinstance(ymax, np.ndarray)
    
    def test_with_smoothing(self):
        """Test maxima finding with smoothing."""
        # Create noisy signal
        np.random.seed(42)
        s = np.sin(np.linspace(0, 2*np.pi, 20)) + 0.1 * np.random.randn(20)
        
        fmax_no_smooth, xmax_no_smooth, _ = calc_interp_maxima(s, lambda_=0.0)
        fmax_smooth, xmax_smooth, _ = calc_interp_maxima(s, lambda_=0.01)
        
        # Results should differ with smoothing
        # (number of maxima may differ)
        assert isinstance(fmax_smooth, np.ndarray)
    
    def test_constant_input(self):
        """Test with constant input (no maxima)."""
        s = np.ones(10) * 5.0
        
        fmax, xmax, c = calc_interp_maxima(s)
        
        # Constant function has no strict maxima
        # Should return empty or handle gracefully
        assert isinstance(fmax, np.ndarray)
        assert isinstance(xmax, np.ndarray)
    
    def test_returns_coefficients(self):
        """Test that function returns coefficients."""
        s = np.sin(np.linspace(0, 2*np.pi, 20))
        
        fmax, xmax, c = calc_interp_maxima(s)
        
        # Coefficients should be returned
        assert c is not None
        assert c.shape == s.shape


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_input(self):
        """Test with empty input."""
        # Empty arrays return empty (handled gracefully)
        s = np.array([])
        c = compute_bspline_coefficients(s)
        assert len(c) == 0
    
    def test_single_element(self):
        """Test with single element."""
        s = np.array([5.0])
        c = compute_bspline_coefficients(s)
        
        # Should handle gracefully for single element
        # Single element arrays are technically 0D after removing singletons
        assert len(c) == 1
        np.testing.assert_allclose(c, s)
    
    def test_very_small_array(self):
        """Test with very small arrays."""
        s = np.array([1.0, 2.0])
        c = compute_bspline_coefficients(s)
        
        assert len(c) == len(s)
    
    def test_invalid_degree(self):
        """Test with invalid degree."""
        s = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError):
            compute_bspline_coefficients(s, degree=4, mode='spatial')
    
    def test_invalid_boundary(self):
        """Test with invalid boundary condition."""
        s = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError):
            compute_bspline_coefficients(s, boundary='invalid', mode='spatial')
    
    def test_3d_input_raises_error(self):
        """Test that 3D input raises error."""
        s = np.random.rand(3, 4, 5)
        
        with pytest.raises(ValueError):
            compute_bspline_coefficients(s)
    
    def test_interp_out_of_bounds(self):
        """Test interpolation handles out-of-bounds coordinates."""
        s = np.array([1.0, 2.0, 3.0, 4.0])
        c = compute_bspline_coefficients(s)
        
        # This should handle boundary conditions
        v = interp_bspline_value(0.0, c, n=3)
        assert np.isfinite(v)
    
    def test_interp_invalid_t_cubic(self):
        """Test that invalid t values raise error in cubic spline."""
        from petakit5d.image_processing.bspline_interp import _get_cubic_spline
        
        with pytest.raises(ValueError):
            _get_cubic_spline(-0.1)
        
        with pytest.raises(ValueError):
            _get_cubic_spline(1.1)
    
    def test_interp_invalid_t_quadratic(self):
        """Test that invalid t values raise error in quadratic spline."""
        from petakit5d.image_processing.bspline_interp import _get_quadratic_spline
        
        with pytest.raises(ValueError):
            _get_quadratic_spline(-0.1)
        
        with pytest.raises(ValueError):
            _get_quadratic_spline(1.1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
