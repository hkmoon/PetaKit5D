"""
Unit tests for UUID utilities.
"""

import pytest
import uuid
import platform
from petakit5d.utils.uuid_utils import get_uuid


class TestGetUuid:
    """Test cases for get_uuid function."""
    
    def test_get_uuid_returns_string(self):
        """Test that get_uuid returns a string."""
        result = get_uuid()
        assert isinstance(result, str)
    
    def test_get_uuid_not_empty(self):
        """Test that get_uuid returns a non-empty string."""
        result = get_uuid()
        assert len(result) > 0
    
    def test_get_uuid_unique(self):
        """Test that get_uuid generates unique values."""
        uuid1 = get_uuid()
        uuid2 = get_uuid()
        # While theoretically possible to be equal, practically should be different
        assert uuid1 != uuid2
    
    @pytest.mark.skipif(
        platform.system() != "Windows",
        reason="Windows-specific behavior"
    )
    def test_get_uuid_windows_truncation(self):
        """Test that UUID is truncated on Windows."""
        result = get_uuid()
        # On Windows, should be truncated to 4 characters or less
        assert len(result) <= 4
    
    @pytest.mark.skipif(
        platform.system() == "Windows",
        reason="Non-Windows behavior"
    )
    def test_get_uuid_non_windows_full_length(self):
        """Test that UUID is full length on non-Windows systems."""
        result = get_uuid()
        # Standard UUID string is 36 characters (with dashes)
        # Or it could be a random number string
        assert len(result) > 4
