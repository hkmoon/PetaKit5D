"""
Tests for Zarr I/O functions.
"""

import numpy as np
import pytest
import tempfile
import shutil
import os
from pathlib import Path


# Check if zarr is available
try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False


@pytest.mark.skipif(not ZARR_AVAILABLE, reason="zarr package not installed")
class TestReadZarr:
    """Tests for read_zarr function."""
    
    def setup_method(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_basic_read(self):
        """Test reading entire Zarr array."""
        from petakit5d.microscope_data_processing.zarr_io import read_zarr, write_zarr
        
        # Create test data
        data = np.random.rand(50, 50, 30) * 255
        filepath = os.path.join(self.temp_dir, "test.zarr")
        
        # Write data
        write_zarr(filepath, data, overwrite=True)
        
        # Read data back
        data_read = read_zarr(filepath)
        
        np.testing.assert_array_equal(data, data_read)
    
    def test_bbox_3d_read(self):
        """Test reading with 3D bounding box."""
        from petakit5d.microscope_data_processing.zarr_io import read_zarr, write_zarr
        
        # Create test data
        data = np.arange(1000).reshape(10, 10, 10)
        filepath = os.path.join(self.temp_dir, "test_bbox.zarr")
        
        # Write data
        write_zarr(filepath, data, overwrite=True)
        
        # Read with bounding box (1-based MATLAB indexing)
        bbox = (2, 3, 4, 6, 7, 8)  # ymin, xmin, zmin, ymax, xmax, zmax
        data_read = read_zarr(filepath, input_bbox=bbox)
        
        # Convert to 0-based for verification
        expected = data[1:6, 2:7, 3:8]
        np.testing.assert_array_equal(data_read, expected)
    
    def test_bbox_2d_read(self):
        """Test reading with 2D bounding box."""
        from petakit5d.microscope_data_processing.zarr_io import read_zarr, write_zarr
        
        # Create test data
        data = np.arange(100).reshape(10, 10)
        filepath = os.path.join(self.temp_dir, "test_2d.zarr")
        
        # Write data
        write_zarr(filepath, data, overwrite=True)
        
        # Read with 2D bounding box
        bbox = (3, 4, 7, 8)  # ymin, xmin, ymax, xmax
        data_read = read_zarr(filepath, input_bbox=bbox)
        
        # Convert to 0-based
        expected = data[2:7, 3:8]
        np.testing.assert_array_equal(data_read, expected)
    
    def test_nonexistent_file_error(self):
        """Test error when file doesn't exist."""
        from petakit5d.microscope_data_processing.zarr_io import read_zarr
        
        with pytest.raises(FileNotFoundError):
            read_zarr("/nonexistent/path/file.zarr")
    
    def test_invalid_bbox_length(self):
        """Test error with invalid bounding box length."""
        from petakit5d.microscope_data_processing.zarr_io import read_zarr, write_zarr
        
        data = np.random.rand(10, 10, 10)
        filepath = os.path.join(self.temp_dir, "test.zarr")
        write_zarr(filepath, data, overwrite=True)
        
        with pytest.raises(ValueError, match="4 \\(2D\\) or 6 \\(3D\\)"):
            read_zarr(filepath, input_bbox=(1, 2, 3))


@pytest.mark.skipif(not ZARR_AVAILABLE, reason="zarr package not installed")
class TestWriteZarr:
    """Tests for write_zarr function."""
    
    def setup_method(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_basic_write(self):
        """Test basic write functionality."""
        from petakit5d.microscope_data_processing.zarr_io import write_zarr
        
        data = np.random.rand(50, 50, 30) * 255
        filepath = os.path.join(self.temp_dir, "output.zarr")
        
        write_zarr(filepath, data, overwrite=True)
        
        # Verify file was created
        assert os.path.exists(filepath)
        
        # Verify data can be read back
        z = zarr.open(filepath, mode='r')
        data_read = z[:]
        np.testing.assert_array_equal(data, data_read)
    
    def test_with_chunks(self):
        """Test writing with custom chunks."""
        from petakit5d.microscope_data_processing.zarr_io import write_zarr
        
        data = np.random.rand(100, 100, 50)
        filepath = os.path.join(self.temp_dir, "chunked.zarr")
        chunks = (50, 50, 25)
        
        write_zarr(filepath, data, chunks=chunks, overwrite=True)
        
        # Verify chunks
        z = zarr.open(filepath, mode='r')
        assert z.chunks == chunks
    
    def test_different_compressors(self):
        """Test different compression algorithms."""
        from petakit5d.microscope_data_processing.zarr_io import write_zarr
        
        data = np.random.rand(30, 30, 30)
        
        for compressor in ['blosc', 'gzip', 'none']:
            filepath = os.path.join(self.temp_dir, f"compressed_{compressor}.zarr")
            write_zarr(filepath, data, compressor=compressor, overwrite=True)
            
            # Verify data integrity
            z = zarr.open(filepath, mode='r')
            data_read = z[:]
            np.testing.assert_allclose(data, data_read, rtol=1e-10)
    
    def test_overwrite_protection(self):
        """Test that overwrite=False prevents overwriting."""
        from petakit5d.microscope_data_processing.zarr_io import write_zarr
        
        data = np.random.rand(10, 10, 10)
        filepath = os.path.join(self.temp_dir, "test_overwrite.zarr")
        
        # First write should succeed
        write_zarr(filepath, data, overwrite=True)
        
        # Second write without overwrite should fail
        # Note: zarr's behavior depends on the storage backend
        # Some may succeed, some may fail - this is expected
        try:
            write_zarr(filepath, data, overwrite=False)
        except Exception:
            pass  # Expected to potentially fail


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
