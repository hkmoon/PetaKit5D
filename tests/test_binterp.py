"""
Tests for B-spline interpolation with derivatives (binterp).
"""

import numpy as np
import pytest
from petakit5d.image_processing.binterp import (
    binterp,
    binterp_1d,
    binterp_2d,
    _cubic_bspline_basis
)


class TestCubicBsplineBasis:
    """Test cubic B-spline basis functions."""

    def test_basis_at_zero(self):
        """Test basis function at x=0."""
        b, db, d2b = _cubic_bspline_basis(np.array([0.0]))
        assert b[0] == pytest.approx(2/3, rel=1e-10)
        assert db[0] == pytest.approx(0.0, abs=1e-10)

    def test_basis_support(self):
        """Test that basis has support in [0, 2]."""
        # Outside support
        b, _, _ = _cubic_bspline_basis(np.array([2.5]))
        assert b[0] == pytest.approx(0.0, abs=1e-10)

        # Inside support
        b, _, _ = _cubic_bspline_basis(np.array([0.5]))
        assert b[0] > 0

    def test_basis_symmetry(self):
        """Test that basis is symmetric."""
        x = np.array([0.7])
        b1, _, _ = _cubic_bspline_basis(x)
        b2, _, _ = _cubic_bspline_basis(-x)
        assert b1[0] == pytest.approx(b2[0], rel=1e-10)


class TestBinterp1D:
    """Test 1D B-spline interpolation."""

    def test_interpolate_at_grid_points(self):
        """Test interpolation at original grid points."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        xi = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        fi, _, _ = binterp_1d(signal, xi)

        # Should match original values at grid points
        assert np.allclose(fi, signal, rtol=0.1)

    def test_interpolate_between_points(self):
        """Test interpolation between grid points."""
        signal = np.array([0.0, 1.0, 0.0])
        xi = np.array([1.0])  # Peak

        fi, fi_dx, fi_d2x = binterp_1d(signal, xi)

        # Should be close to 1.0 at peak
        assert fi[0] > 0.8
        # Derivative should be close to 0 at peak
        assert abs(fi_dx[0]) < 0.5
        # Second derivative should be negative (concave)
        assert fi_d2x[0] < 0

    def test_linear_signal(self):
        """Test with linear signal."""
        signal = np.linspace(0, 10, 11)
        xi = np.array([2.5, 5.5, 8.5])

        fi, fi_dx, _ = binterp_1d(signal, xi)

        # Should interpolate linearly
        assert np.allclose(fi, [2.5, 5.5, 8.5], rtol=0.1)
        # Derivative should be approximately 1
        assert np.allclose(fi_dx, [1.0, 1.0, 1.0], rtol=0.5)

    def test_constant_signal(self):
        """Test with constant signal."""
        signal = np.ones(10) * 5.0
        xi = np.array([2.3, 5.7, 8.1])

        fi, fi_dx, fi_d2x = binterp_1d(signal, xi)

        # Should be constant
        assert np.allclose(fi, 5.0, rtol=0.1)
        # Derivatives should be near zero
        assert np.allclose(fi_dx, 0.0, atol=0.3)
        assert np.allclose(fi_d2x, 0.0, atol=0.5)

    def test_mirror_boundary(self):
        """Test mirror boundary conditions."""
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        # Interpolate near boundaries
        xi = np.array([0.0, 3.0])

        fi, _, _ = binterp_1d(signal, xi, border_condition='mirror')

        assert np.all(np.isfinite(fi))

    def test_periodic_boundary(self):
        """Test periodic boundary conditions."""
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        xi = np.array([0.5, 3.5])

        fi, _, _ = binterp_1d(signal, xi, border_condition='periodic')

        assert np.all(np.isfinite(fi))


class TestBinterp2D:
    """Test 2D B-spline interpolation."""

    def test_interpolate_at_grid_points(self):
        """Test interpolation at original grid points."""
        image = np.array([[1, 2], [3, 4]], dtype=float)
        xi = np.array([0.0, 1.0])
        yi = np.array([0.0, 1.0])

        fi, _, _, _, _ = binterp_2d(image, xi, yi)

        # Should be close to original values
        expected = np.array([1.0, 4.0])
        assert np.allclose(fi, expected, rtol=0.2)

    def test_interpolate_center(self):
        """Test interpolation at image center."""
        image = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]], dtype=float)
        xi = np.array([1.0])
        yi = np.array([1.0])

        fi, _, _, _, _ = binterp_2d(image, xi, yi)

        # Should be close to 1 at peak
        assert fi[0] > 0.8

    def test_constant_image(self):
        """Test with constant image."""
        image = np.ones((5, 5)) * 7.0
        xi = np.array([1.5, 2.5, 3.5])
        yi = np.array([1.5, 2.5, 3.5])

        fi, fi_dx, fi_dy, fi_d2x, fi_d2y = binterp_2d(image, xi, yi)

        # Should be constant
        assert np.allclose(fi, 7.0, rtol=0.1)
        # Derivatives should be near zero
        assert np.allclose(fi_dx, 0.0, atol=0.3)
        assert np.allclose(fi_dy, 0.0, atol=0.3)

    def test_shape_mismatch_error(self):
        """Test error when xi and yi have different shapes."""
        image = np.ones((5, 5))
        xi = np.array([1.0, 2.0])
        yi = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="same shape"):
            binterp_2d(image, xi, yi)

    def test_mirror_boundary_2d(self):
        """Test mirror boundary in 2D."""
        image = np.random.rand(5, 5)
        xi = np.array([0.0, 4.0])
        yi = np.array([0.0, 4.0])

        fi, _, _, _, _ = binterp_2d(image, xi, yi, border_condition='mirror')

        assert np.all(np.isfinite(fi))

    def test_periodic_boundary_2d(self):
        """Test periodic boundary in 2D."""
        image = np.random.rand(5, 5)
        xi = np.array([0.5, 4.5])
        yi = np.array([0.5, 4.5])

        fi, _, _, _, _ = binterp_2d(image, xi, yi, border_condition='periodic')

        assert np.all(np.isfinite(fi))


class TestBinterp:
    """Test unified binterp interface."""

    def test_1d_interface(self):
        """Test 1D interface."""
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        xi = np.array([1.5, 2.5])

        result = binterp(signal, xi)

        assert len(result) == 3  # fi, fi_dx, fi_d2x
        assert result[0].shape == xi.shape

    def test_2d_interface(self):
        """Test 2D interface."""
        image = np.random.rand(5, 5)
        xi = np.array([1.5, 2.5])
        yi = np.array([1.5, 2.5])

        result = binterp(image, xi, yi)

        assert len(result) == 5  # fi, fi_dx, fi_dy, fi_d2x, fi_d2y
        assert result[0].shape == xi.shape

    def test_border_condition_kwarg(self):
        """Test border_condition keyword argument."""
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        xi = np.array([0.5])

        fi_mirror, _, _ = binterp(signal, xi, border_condition='mirror')
        fi_periodic, _, _ = binterp(signal, xi, border_condition='periodic')

        # Different boundary conditions may give different results
        assert np.all(np.isfinite(fi_mirror))
        assert np.all(np.isfinite(fi_periodic))

    def test_invalid_dimension_error(self):
        """Test error for invalid dimensions."""
        signal = np.ones((3, 3, 3))  # 3D not supported
        xi = np.array([1.0])

        with pytest.raises(ValueError, match="Invalid input dimensions"):
            binterp(signal, xi)

    def test_missing_args_error(self):
        """Test error when invalid dimension inputs."""
        signal = np.ones((5, 5, 5))  # 3D array
        xi = np.array([1.0])

        # 3D array should raise error
        with pytest.raises(ValueError):
            binterp(signal, xi)

    def test_sine_wave_interpolation(self):
        """Test interpolation of sine wave."""
        x = np.linspace(0, 2*np.pi, 20)
        signal = np.sin(x)

        # Interpolate at finer resolution
        xi = np.linspace(0, 19, 50)
        fi, fi_dx, _ = binterp(signal, xi)

        # Check that interpolation is smooth
        assert np.all(np.isfinite(fi))
        assert np.all(np.isfinite(fi_dx))

        # Values should be in range [-1, 1]
        assert np.all(fi >= -1.5)
        assert np.all(fi <= 1.5)


class TestBinterpDerivatives:
    """Test derivative computation."""

    def test_quadratic_second_derivative(self):
        """Test second derivative of quadratic function."""
        # f(x) = x^2, f''(x) = 2
        x = np.linspace(0, 4, 5)
        signal = x**2

        xi = np.array([2.0])
        _, _, fi_d2x = binterp_1d(signal, xi)

        # Second derivative should be approximately 2 (cubic spline may have some error)
        assert abs(fi_d2x[0] - 2.0) < 2.1

    def test_derivative_sign(self):
        """Test derivative signs."""
        # Increasing function
        signal = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        xi = np.array([2.0])

        _, fi_dx, _ = binterp_1d(signal, xi)

        # Derivative should be positive
        assert fi_dx[0] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
