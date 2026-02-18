"""
Tests for path utility functions.
"""

import pytest
import os
import tempfile
from pathlib import Path
from petakit5d.utils.path_utils import simplify_path


class TestSimplifyPath:
    """Test simplify_path function."""
    
    def test_current_directory(self):
        """Test simplifying current directory."""
        result = simplify_path('.')
        expected = os.getcwd()
        assert result == expected
        assert not result.endswith(os.sep)
    
    def test_absolute_path_file(self, tmp_path):
        """Test with absolute path to a file."""
        # Create a temporary file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        
        result = simplify_path(str(test_file))
        assert result == str(test_file)
        assert os.path.isabs(result)
    
    def test_absolute_path_directory(self, tmp_path):
        """Test with absolute path to a directory."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        
        result = simplify_path(str(test_dir))
        assert result == str(test_dir)
        assert not result.endswith(os.sep)
        assert os.path.isabs(result)
    
    def test_relative_path_file(self, tmp_path):
        """Test with relative path to a file."""
        # Create a file in tmp directory
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        
        # Change to tmp directory and use relative path
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = simplify_path("test.txt")
            assert result == str(test_file)
            assert os.path.isabs(result)
        finally:
            os.chdir(original_dir)
    
    def test_relative_path_directory(self, tmp_path):
        """Test with relative path to a directory."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = simplify_path("test_dir")
            assert result == str(test_dir)
            assert not result.endswith(os.sep)
            assert os.path.isabs(result)
        finally:
            os.chdir(original_dir)
    
    def test_directory_with_trailing_slash(self, tmp_path):
        """Test directory path with trailing slash is normalized."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        
        # Add trailing slash
        path_with_slash = str(test_dir) + os.sep
        result = simplify_path(path_with_slash)
        
        # Result should not have trailing slash
        assert not result.endswith(os.sep)
        assert result == str(test_dir)
    
    def test_nested_directories(self, tmp_path):
        """Test with nested directory structure."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        nested_dir.mkdir(parents=True)
        
        result = simplify_path(str(nested_dir))
        assert result == str(nested_dir)
        assert not result.endswith(os.sep)
    
    def test_parent_directory_reference(self, tmp_path):
        """Test with parent directory references (..)."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        nested_dir = test_dir / "nested"
        nested_dir.mkdir()
        
        # Use .. to reference parent
        original_dir = os.getcwd()
        try:
            os.chdir(nested_dir)
            result = simplify_path("..")
            assert result == str(test_dir)
            assert not result.endswith(os.sep)
        finally:
            os.chdir(original_dir)
    
    def test_nonexistent_path(self):
        """Test that nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Path does not exist"):
            simplify_path("/nonexistent/path/that/does/not/exist")
    
    def test_symlink(self, tmp_path):
        """Test with symbolic link."""
        # Create a real directory
        real_dir = tmp_path / "real_dir"
        real_dir.mkdir()
        
        # Create a symbolic link
        link_path = tmp_path / "link_dir"
        try:
            link_path.symlink_to(real_dir)
            
            # simplify_path should resolve the symlink
            result = simplify_path(str(link_path))
            # The result should be the resolved absolute path
            assert os.path.isabs(result)
            assert not result.endswith(os.sep)
        except OSError:
            # Skip test if symlinks are not supported (e.g., Windows without admin rights)
            pytest.skip("Symlinks not supported on this system")
