"""
Unit tests for file utilities.
"""

import pytest
import json
from pathlib import Path
import tempfile
from petakit5d.utils.file_utils import (
    read_text_file,
    write_text_file,
    write_json_file
)


class TestReadTextFile:
    """Test cases for read_text_file function."""
    
    def test_read_simple_file(self, tmp_path):
        """Test reading a simple text file."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        content = "line1\nline2\nline3"
        test_file.write_text(content)
        
        result = read_text_file(str(test_file))
        assert result == ["line1", "line2", "line3"]
    
    def test_read_empty_file(self, tmp_path):
        """Test reading an empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")
        
        result = read_text_file(str(test_file))
        # Empty file returns empty list (no lines to read)
        assert result == []
    
    def test_read_file_with_empty_lines(self, tmp_path):
        """Test reading a file with empty lines."""
        test_file = tmp_path / "test_empty_lines.txt"
        content = "line1\n\nline3"
        test_file.write_text(content)
        
        result = read_text_file(str(test_file))
        assert result == ["line1", "", "line3"]
    
    def test_read_nonexistent_file(self):
        """Test that reading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_text_file("/nonexistent/path/file.txt")


class TestWriteTextFile:
    """Test cases for write_text_file function."""
    
    def test_write_string(self, tmp_path):
        """Test writing a single string."""
        test_file = tmp_path / "output.txt"
        content = "Hello, World!"
        
        write_text_file(content, str(test_file))
        
        assert test_file.read_text() == content
    
    def test_write_list_small(self, tmp_path):
        """Test writing a small list of strings."""
        test_file = tmp_path / "output_list.txt"
        content = ["line1", "line2", "line3"]
        
        write_text_file(content, str(test_file))
        
        assert test_file.read_text() == "line1\nline2\nline3"
    
    def test_write_list_large(self, tmp_path):
        """Test writing a large list in batches."""
        test_file = tmp_path / "output_large.txt"
        # Create a list larger than default batch size
        content = [f"line{i}" for i in range(15000)]
        
        write_text_file(content, str(test_file), batch_size=5000)
        
        result = read_text_file(str(test_file))
        assert len(result) == 15000
        assert result[0] == "line0"
        assert result[-1] == "line14999"
    
    def test_write_overwrites_existing(self, tmp_path):
        """Test that writing overwrites existing file."""
        test_file = tmp_path / "overwrite.txt"
        test_file.write_text("old content")
        
        write_text_file("new content", str(test_file))
        
        assert test_file.read_text() == "new content"


class TestWriteJsonFile:
    """Test cases for write_json_file function."""
    
    def test_write_simple_dict(self, tmp_path):
        """Test writing a simple dictionary."""
        test_file = tmp_path / "output.json"
        data = {"key1": "value1", "key2": 42}
        
        write_json_file(data, str(test_file))
        
        with open(test_file, 'r') as f:
            loaded = json.load(f)
        
        assert loaded == data
    
    def test_write_nested_dict(self, tmp_path):
        """Test writing a nested dictionary."""
        test_file = tmp_path / "nested.json"
        data = {
            "level1": {
                "level2": {
                    "key": "value"
                }
            }
        }
        
        write_json_file(data, str(test_file))
        
        with open(test_file, 'r') as f:
            loaded = json.load(f)
        
        assert loaded == data
    
    def test_write_with_list(self, tmp_path):
        """Test writing a dict containing a list."""
        test_file = tmp_path / "with_list.json"
        data = {"items": [1, 2, 3, 4, 5]}
        
        write_json_file(data, str(test_file))
        
        with open(test_file, 'r') as f:
            loaded = json.load(f)
        
        assert loaded == data
    
    def test_json_is_formatted(self, tmp_path):
        """Test that JSON output is pretty-printed."""
        test_file = tmp_path / "formatted.json"
        data = {"a": 1, "b": 2}
        
        write_json_file(data, str(test_file))
        
        content = test_file.read_text()
        # Pretty-printed JSON should have newlines
        assert '\n' in content
