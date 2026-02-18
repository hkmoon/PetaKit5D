"""
Unit tests for power_utils module.
"""

import numpy as np
import pytest
from petakit5d.utils.power_utils import fast_power


class TestFastPower:
    """Test cases for fast_power function."""
    
    def test_positive_integer_exponent_scalar(self):
        """Test with positive integer exponent and scalar base."""
        assert fast_power(2, 10) == 1024.0
        assert fast_power(3, 4) == 81.0
        assert fast_power(5, 3) == 125.0
    
    def test_negative_exponent_scalar(self):
        """Test with negative exponent and scalar base."""
        result = fast_power(2, -3)
        assert np.isclose(result, 0.125)
        
        result = fast_power(10, -2)
        assert np.isclose(result, 0.01)
    
    def test_exponent_one(self):
        """Test with exponent equal to 1."""
        assert fast_power(7, 1) == 7.0
        assert fast_power(123.456, 1) == 123.456
    
    def test_exponent_zero(self):
        """Test with exponent equal to 0 (should return 1)."""
        # Note: 0^0 case handled by while loop exiting immediately
        result = fast_power(5, 0)
        assert result == 1.0
    
    def test_array_input(self):
        """Test with array base."""
        base = np.array([2, 3, 4])
        result = fast_power(base, 3)
        expected = np.array([8.0, 27.0, 64.0])
        np.testing.assert_array_equal(result, expected)
    
    def test_array_negative_exponent(self):
        """Test with array base and negative exponent."""
        base = np.array([2, 4, 10])
        result = fast_power(base, -2)
        expected = np.array([0.25, 0.0625, 0.01])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_float_base(self):
        """Test with float base."""
        result = fast_power(1.5, 4)
        expected = 1.5 ** 4
        assert np.isclose(result, expected)
    
    def test_large_exponent(self):
        """Test with large exponent."""
        result = fast_power(2, 20)
        expected = 2.0 ** 20
        assert result == expected
    
    def test_2d_array(self):
        """Test with 2D array base."""
        base = np.array([[2, 3], [4, 5]])
        result = fast_power(base, 2)
        expected = np.array([[4.0, 9.0], [16.0, 25.0]])
        np.testing.assert_array_equal(result, expected)
    
    def test_edge_case_one(self):
        """Test with base = 1."""
        assert fast_power(1, 100) == 1.0
        assert fast_power(1, -100) == 1.0
    
    def test_efficiency(self):
        """Test that function works efficiently for large exponents."""
        # Should complete quickly even for large exponents
        result = fast_power(2, 1000)
        expected = 2.0 ** 1000
        assert result == expected
