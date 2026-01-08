"""
Unit tests for string utilities.
"""

import pytest
import numpy as np
from petakit5d.utils.string_utils import mat2str_comma


class TestMat2strComma:
    """Test cases for mat2str_comma function."""
    
    def test_integer_list(self):
        """Test conversion of integer list."""
        result = mat2str_comma([1, 2, 3])
        assert result == '[1,2,3]'
    
    def test_integer_array(self):
        """Test conversion of integer numpy array."""
        result = mat2str_comma(np.array([4, 5, 6]))
        assert result == '[4,5,6]'
    
    def test_single_value(self):
        """Test conversion of single value."""
        result = mat2str_comma([42])
        assert result == '[42]'
    
    def test_float_with_sn(self):
        """Test conversion of floats with significant numbers."""
        result = mat2str_comma([1.5, 2.7, 3.9], sn=2)
        assert result == '[1.50,2.70,3.90]'
    
    def test_float_with_sn_1(self):
        """Test conversion of floats with 1 decimal place."""
        result = mat2str_comma([1.456, 2.789], sn=1)
        assert result == '[1.5,2.8]'
    
    def test_float_with_sn_3(self):
        """Test conversion of floats with 3 decimal places."""
        result = mat2str_comma([3.14159, 2.71828], sn=3)
        assert result == '[3.142,2.718]'
    
    def test_negative_numbers(self):
        """Test conversion of negative numbers."""
        result = mat2str_comma([-1, -2, -3])
        assert result == '[-1,-2,-3]'
    
    def test_mixed_positive_negative(self):
        """Test conversion of mixed positive and negative numbers."""
        result = mat2str_comma([1, -2, 3])
        assert result == '[1,-2,3]'
    
    def test_zero(self):
        """Test conversion including zero."""
        result = mat2str_comma([0, 1, 2])
        assert result == '[0,1,2]'
    
    def test_2d_array_flattened(self):
        """Test that 2D arrays are flattened."""
        arr = np.array([[1, 2], [3, 4]])
        result = mat2str_comma(arr)
        assert result == '[1,2,3,4]'
