"""
Tests for math utility functions.
"""

import pytest
import numpy as np
from petakit5d.utils.math_utils import find_good_factor_number, _prime_factors


class TestPrimeFactors:
    """Test the prime factorization helper function."""
    
    def test_small_primes(self):
        """Test factorization of small prime numbers."""
        assert _prime_factors(2) == [2]
        assert _prime_factors(3) == [3]
        assert _prime_factors(5) == [5]
        assert _prime_factors(7) == [7]
    
    def test_composite_numbers(self):
        """Test factorization of composite numbers."""
        assert _prime_factors(4) == [2, 2]
        assert _prime_factors(6) == [2, 3]
        assert _prime_factors(12) == [2, 2, 3]
        assert _prime_factors(30) == [2, 3, 5]
        assert _prime_factors(210) == [2, 3, 5, 7]
    
    def test_edge_cases(self):
        """Test edge cases."""
        assert _prime_factors(1) == []
        assert _prime_factors(0) == []


class TestFindGoodFactorNumber:
    """Test find_good_factor_number function."""
    
    def test_already_good_number(self):
        """Test when input is already a good number."""
        # Numbers with only factors 2, 3, 5
        assert find_good_factor_number(1) == 1
        assert find_good_factor_number(2) == 2
        assert find_good_factor_number(6) == 6
        assert find_good_factor_number(30) == 30
        assert find_good_factor_number(100) == 100  # 2^2 * 5^2
    
    def test_search_upward(self):
        """Test searching for next higher good number."""
        # 105 = 3 * 5 * 7 is odd with factor 7, so by default skipped
        # Next good even number is 108 = 2^2 * 3^3
        assert find_good_factor_number(101, direction=1) == 108
        assert find_good_factor_number(99, direction=1) == 100   # Next is 100 = 2^2 * 5^2
        assert find_good_factor_number(11, direction=1) == 12    # Next is 12 = 2^2 * 3
    
    def test_search_downward(self):
        """Test searching for next lower good number."""
        assert find_good_factor_number(101, direction=-1) == 100
        assert find_good_factor_number(99, direction=-1) == 98  # 98 = 2 * 7^2
        assert find_good_factor_number(13, direction=-1) == 12
    
    def test_factor_seven_even_constraint(self):
        """Test that factor 7 requires even number when allow_odd_seven=False."""
        # 105 = 3 * 5 * 7 is odd with factor 7
        # With allow_odd_seven=False, should skip it
        result = find_good_factor_number(105, direction=1, allow_odd_seven=False)
        # Should find next even number with good factors
        assert result > 105
        # Check it's even
        assert result % 2 == 0
    
    def test_allow_odd_seven(self):
        """Test that odd numbers with factor 7 are allowed when flag is True."""
        result = find_good_factor_number(105, direction=1, allow_odd_seven=True)
        # 105 = 3 * 5 * 7 should be acceptable
        assert result == 105
    
    def test_large_numbers(self):
        """Test with larger numbers."""
        # 1024 = 2^10 is good
        assert find_good_factor_number(1024) == 1024
        
        # 1023 = 3 * 11 * 31 has factor 11 and 31
        result = find_good_factor_number(1023, direction=1)
        assert result >= 1024
        factors = _prime_factors(result)
        assert all(f <= 7 for f in factors)
    
    def test_invalid_direction(self):
        """Test that invalid direction raises error."""
        with pytest.raises(ValueError, match="direction must be 1 or -1"):
            find_good_factor_number(100, direction=0)
        
        with pytest.raises(ValueError, match="direction must be 1 or -1"):
            find_good_factor_number(100, direction=2)
    
    def test_non_integer_input(self):
        """Test that non-integer input raises error."""
        with pytest.raises(TypeError, match="given_num must be an integer"):
            find_good_factor_number(100.5)
    
    def test_numpy_integer(self):
        """Test that numpy integers are accepted."""
        result = find_good_factor_number(np.int32(100))
        assert result == 100
        
        result = find_good_factor_number(np.int64(101), direction=1)
        assert result == 108  # 108 = 2^2 * 3^3 (105 is odd with factor 7)
