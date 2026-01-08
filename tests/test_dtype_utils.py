"""
Unit tests for data type utilities.
"""

import pytest
from petakit5d.utils.dtype_utils import data_type_to_byte_number


class TestDataTypeToByteNumber:
    """Test cases for data_type_to_byte_number function."""
    
    def test_uint8(self):
        """Test uint8 returns 1 byte."""
        assert data_type_to_byte_number('uint8') == 1
    
    def test_uint16(self):
        """Test uint16 returns 2 bytes."""
        assert data_type_to_byte_number('uint16') == 2
    
    def test_single(self):
        """Test single returns 4 bytes."""
        assert data_type_to_byte_number('single') == 4
    
    def test_float32(self):
        """Test float32 returns 4 bytes."""
        assert data_type_to_byte_number('float32') == 4
    
    def test_double(self):
        """Test double returns 8 bytes."""
        assert data_type_to_byte_number('double') == 8
    
    def test_float64(self):
        """Test float64 returns 8 bytes."""
        assert data_type_to_byte_number('float64') == 8
    
    def test_unsupported_type(self):
        """Test that unsupported type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported data type"):
            data_type_to_byte_number('int32')
    
    def test_invalid_type(self):
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported data type"):
            data_type_to_byte_number('invalid_type')
    
    def test_empty_string(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported data type"):
            data_type_to_byte_number('')
