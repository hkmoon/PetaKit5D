"""
Unit tests for system utilities.
"""

import pytest
from petakit5d.utils.system_utils import get_hostname


class TestGetHostname:
    """Test cases for get_hostname function."""
    
    def test_get_hostname_returns_string(self):
        """Test that get_hostname returns a string."""
        result = get_hostname()
        assert isinstance(result, str)
    
    def test_get_hostname_not_empty(self):
        """Test that get_hostname returns a non-empty string."""
        result = get_hostname()
        assert len(result) > 0
    
    def test_get_hostname_consistent(self):
        """Test that get_hostname returns consistent value."""
        result1 = get_hostname()
        result2 = get_hostname()
        assert result1 == result2
