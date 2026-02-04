"""
Tests for TIFF I/O functions.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from petakit5d.microscope_data_processing import write_tiff, read_tiff


class TestWriteTiff:
    """Test suite for write_tiff function."""

    def test_write_2d_image(self):
        """Test writing a 2D image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_2d.tif')

            # Create test image
            img = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)

            # Write it
            write_tiff(img, filepath)

            # Check file exists
            assert os.path.exists(filepath)

            # Read it back and verify
            img_read = read_tiff(filepath)
            np.testing.assert_array_equal(img, img_read)

    def test_write_3d_stack(self):
        """Test writing a 3D stack."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_3d.tif')

            # Create test stack
            stack = np.random.randint(0, 255, (50, 100, 100), dtype=np.uint8)

            # Write it
            write_tiff(stack, filepath)

            # Check file exists
            assert os.path.exists(filepath)

            # Read it back and verify
            stack_read = read_tiff(filepath)
            np.testing.assert_array_equal(stack, stack_read)

    def test_write_with_no_compression(self):
        """Test writing without compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_no_compress.tif')

            # Create test image
            img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)

            # Write without compression
            write_tiff(img, filepath, compression='none')

            # Check file exists
            assert os.path.exists(filepath)

            # Read it back and verify
            img_read = read_tiff(filepath)
            np.testing.assert_array_equal(img, img_read)

    def test_write_with_lzw_compression(self):
        """Test writing with LZW compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_lzw.tif')

            # Create test image
            img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)

            # Write with LZW compression
            write_tiff(img, filepath, compression='lzw')

            # Check file exists
            assert os.path.exists(filepath)

            # Read it back and verify
            img_read = read_tiff(filepath)
            np.testing.assert_array_equal(img, img_read)

    def test_write_creates_parent_directories(self):
        """Test that parent directories are created if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'subdir1', 'subdir2', 'test.tif')

            # Create test image
            img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)

            # Write (should create parent directories)
            write_tiff(img, filepath)

            # Check file exists
            assert os.path.exists(filepath)

            # Check parent directories were created
            assert os.path.exists(os.path.join(tmpdir, 'subdir1'))
            assert os.path.exists(os.path.join(tmpdir, 'subdir1', 'subdir2'))

    def test_write_different_dtypes(self):
        """Test writing images with different data types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for dtype in [np.uint8, np.uint16, np.float32, np.float64]:
                filepath = os.path.join(tmpdir, f'test_{dtype.__name__}.tif')

                # Create test image
                if dtype in [np.uint8, np.uint16]:
                    img = np.random.randint(0, 100, (50, 50), dtype=dtype)
                else:
                    img = np.random.rand(50, 50).astype(dtype)

                # Write it
                write_tiff(img, filepath)

                # Check file exists
                assert os.path.exists(filepath)

                # Read it back and verify
                img_read = read_tiff(filepath)
                np.testing.assert_array_almost_equal(img, img_read)

    def test_write_invalid_compression(self):
        """Test that invalid compression raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test.tif')
            img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)

            with pytest.raises(ValueError, match="Invalid compression"):
                write_tiff(img, filepath, compression='invalid')

    def test_write_invalid_mode(self):
        """Test that invalid mode raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test.tif')
            img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)

            with pytest.raises(OSError, match="Invalid mode"):
                write_tiff(img, filepath, mode='invalid')

    def test_write_overwrite(self):
        """Test that writing overwrites existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test.tif')

            # Create and write first image
            img1 = np.ones((50, 50), dtype=np.uint8)
            write_tiff(img1, filepath)

            # Create and write second image (should overwrite)
            img2 = np.zeros((50, 50), dtype=np.uint8)
            write_tiff(img2, filepath)

            # Read back and verify it's the second image
            img_read = read_tiff(filepath)
            np.testing.assert_array_equal(img2, img_read)

    def test_write_4d_data(self):
        """Test writing 4D data (e.g., time series of 3D stacks)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_4d.tif')

            # Create 4D data
            data = np.random.randint(0, 255, (10, 20, 30, 30), dtype=np.uint8)

            # Write it
            write_tiff(data, filepath)

            # Check file exists
            assert os.path.exists(filepath)

            # Read it back and verify
            data_read = read_tiff(filepath)
            np.testing.assert_array_equal(data, data_read)

    @pytest.mark.skipif(
        not hasattr(os, 'chmod'),
        reason="File permissions not supported on this platform"
    )
    def test_write_file_permissions(self):
        """Test that file permissions are set correctly on Unix systems."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test.tif')

            # Create test image
            img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)

            # Write it
            write_tiff(img, filepath)

            # Check file permissions (should be readable/writable by user and group)
            if os.name != 'nt':  # Skip on Windows
                stat_info = os.stat(filepath)
                # Check that file has read permissions
                assert stat_info.st_mode & 0o400  # User read

    def test_roundtrip_consistency(self):
        """Test that write -> read produces identical data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_roundtrip.tif')

            # Test with various image types
            test_images = [
                np.random.randint(0, 255, (100, 100), dtype=np.uint8),
                np.random.randint(0, 65535, (100, 100), dtype=np.uint16),
                np.random.rand(100, 100).astype(np.float32),
                np.random.randint(0, 255, (10, 50, 50), dtype=np.uint8),
            ]

            for i, img in enumerate(test_images):
                filepath_i = filepath.replace('.tif', f'_{i}.tif')
                write_tiff(img, filepath_i)
                img_read = read_tiff(filepath_i)
                np.testing.assert_array_almost_equal(img, img_read)


class TestReadTiffRange:
    """Test suite for read_tiff with range parameter."""

    def test_read_specific_range(self):
        """Test reading specific range from a stack."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_stack.tif')

            # Create and write a stack
            stack = np.random.randint(0, 255, (100, 50, 50), dtype=np.uint8)
            write_tiff(stack, filepath)

            # Read a specific range (MATLAB 1-based indexing)
            range_data = read_tiff(filepath, range_indices=(10, 20))

            # Should get slices 9:20 (0-based)
            expected = stack[9:20]
            np.testing.assert_array_equal(range_data, expected)

    def test_read_range_boundary(self):
        """Test reading range at boundaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_stack.tif')

            # Create and write a stack
            stack = np.random.randint(0, 255, (50, 30, 30), dtype=np.uint8)
            write_tiff(stack, filepath)

            # Read from beginning
            range_data = read_tiff(filepath, range_indices=(1, 10))
            expected = stack[0:10]
            np.testing.assert_array_equal(range_data, expected)

            # Read to end
            range_data = read_tiff(filepath, range_indices=(41, 50))
            expected = stack[40:50]
            np.testing.assert_array_equal(range_data, expected)
