"""
Unit tests for axis utilities.
"""

import pytest
from petakit5d.utils.axis_utils import axis_order_mapping


class TestAxisOrderMapping:
    """Test cases for axis_order_mapping function."""
    
    def test_xyz_to_yxz(self):
        """Test mapping from xyz to yxz."""
        result = axis_order_mapping('xyz', 'yxz')
        assert result == (2, 1, 3)
    
    def test_yxz_to_xyz(self):
        """Test mapping from yxz to xyz."""
        result = axis_order_mapping('yxz', 'xyz')
        assert result == (2, 1, 3)
    
    def test_zyx_to_xyz(self):
        """Test mapping from zyx to xyz."""
        result = axis_order_mapping('zyx', 'xyz')
        assert result == (3, 2, 1)
    
    def test_xyz_to_xyz(self):
        """Test identity mapping (xyz to xyz)."""
        result = axis_order_mapping('xyz', 'xyz')
        assert result == (1, 2, 3)
    
    def test_default_output_order(self):
        """Test default output order (yxz)."""
        result = axis_order_mapping('xyz')
        assert result == (2, 1, 3)
    
    def test_case_insensitive(self):
        """Test that function is case-insensitive."""
        result1 = axis_order_mapping('XYZ', 'YXZ')
        result2 = axis_order_mapping('xyz', 'yxz')
        assert result1 == result2
    
    def test_all_valid_permutations(self):
        """Test some valid axis order permutations."""
        # Test a few different valid orderings
        result1 = axis_order_mapping('zxy', 'xyz')
        assert result1 == (3, 1, 2)  # z->3, x->1, y->2 in xyz
        
        result2 = axis_order_mapping('yzx', 'xyz')
        assert result2 == (2, 3, 1)  # y->2, z->3, x->1 in xyz
        
        result3 = axis_order_mapping('xzy', 'xyz')
        assert result3 == (1, 3, 2)  # x->1, z->3, y->2 in xyz
    
    def test_invalid_input_length(self):
        """Test that invalid input length raises ValueError."""
        with pytest.raises(ValueError, match="must contain exactly x, y, and z"):
            axis_order_mapping('xy', 'xyz')
    
    def test_invalid_input_characters(self):
        """Test that invalid characters raise ValueError."""
        with pytest.raises(ValueError, match="must contain exactly x, y, and z"):
            axis_order_mapping('abc', 'xyz')
    
    def test_invalid_input_order(self):
        """Test that invalid order raises ValueError."""
        with pytest.raises(ValueError, match="must be one of"):
            axis_order_mapping('xyx', 'xyz')  # duplicate 'x'
    
    def test_invalid_output_order(self):
        """Test that invalid output order raises ValueError."""
        with pytest.raises(ValueError, match="must be one of"):
            axis_order_mapping('xyz', 'xyx')  # duplicate 'x'
