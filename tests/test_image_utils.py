"""
Unit tests for image_utils module.
"""

import numpy as np
import pytest
import tempfile
import os
from petakit5d.utils.image_utils import get_image_bounding_box, get_image_data_type


class TestGetImageBoundingBox:
    """Test cases for get_image_bounding_box function."""
    
    def test_2d_simple(self):
        """Test 2D image with simple non-zero region."""
        img = np.zeros((10, 10))
        img[3:7, 2:8] = 1
        bbox = get_image_bounding_box(img)
        # Expected: (y1, x1, y2, x2) in 1-based indexing
        assert bbox == (4, 3, 7, 8)
    
    def test_3d_simple(self):
        """Test 3D image with simple non-zero region."""
        img = np.zeros((10, 10, 10))
        img[2:8, 3:7, 4:6] = 1
        bbox = get_image_bounding_box(img)
        # Expected: (y1, x1, z1, y2, x2, z2) in 1-based indexing
        assert bbox == (3, 4, 5, 8, 7, 6)
    
    def test_2d_full_image(self):
        """Test 2D image that is fully non-zero."""
        img = np.ones((5, 7))
        bbox = get_image_bounding_box(img)
        assert bbox == (1, 1, 5, 7)
    
    def test_3d_full_image(self):
        """Test 3D image that is fully non-zero."""
        img = np.ones((4, 6, 8))
        bbox = get_image_bounding_box(img)
        assert bbox == (1, 1, 1, 4, 6, 8)
    
    def test_2d_empty_image(self):
        """Test 2D image with all zeros."""
        img = np.zeros((10, 10))
        bbox = get_image_bounding_box(img)
        assert bbox == (0, 0, 0, 0)
    
    def test_3d_empty_image(self):
        """Test 3D image with all zeros."""
        img = np.zeros((10, 10, 10))
        bbox = get_image_bounding_box(img)
        assert bbox == (0, 0, 0, 0, 0, 0)
    
    def test_2d_single_pixel(self):
        """Test 2D image with single non-zero pixel."""
        img = np.zeros((10, 10))
        img[5, 7] = 1
        bbox = get_image_bounding_box(img)
        assert bbox == (6, 8, 6, 8)
    
    def test_3d_single_voxel(self):
        """Test 3D image with single non-zero voxel."""
        img = np.zeros((10, 10, 10))
        img[3, 5, 7] = 1
        bbox = get_image_bounding_box(img)
        assert bbox == (4, 6, 8, 4, 6, 8)
    
    def test_2d_corner_values(self):
        """Test 2D image with values in corners."""
        img = np.zeros((10, 10))
        img[0, 0] = 1
        img[9, 9] = 1
        bbox = get_image_bounding_box(img)
        assert bbox == (1, 1, 10, 10)
    
    def test_3d_corner_values(self):
        """Test 3D image with values in corners."""
        img = np.zeros((10, 10, 10))
        img[0, 0, 0] = 1
        img[9, 9, 9] = 1
        bbox = get_image_bounding_box(img)
        assert bbox == (1, 1, 1, 10, 10, 10)
    
    def test_invalid_dimensions(self):
        """Test with invalid image dimensions."""
        img = np.zeros((10, 10, 10, 10))  # 4D
        with pytest.raises(ValueError, match="Image must be 2D or 3D"):
            get_image_bounding_box(img)
        
        img = np.zeros(10)  # 1D
        with pytest.raises(ValueError, match="Image must be 2D or 3D"):
            get_image_bounding_box(img)
    
    def test_float_values(self):
        """Test with float image values."""
        img = np.zeros((10, 10))
        img[2:5, 3:7] = 0.5
        bbox = get_image_bounding_box(img)
        assert bbox == (3, 4, 5, 7)


class TestGetImageDataType:
    """Test cases for get_image_data_type function."""
    
    def test_nonexistent_file(self):
        """Test with file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            get_image_data_type('/nonexistent/path/file.tif')
    
    def test_unsupported_format(self):
        """Test with unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            filepath = f.name
        
        try:
            with pytest.raises(ValueError, match="Unknown file format"):
                get_image_data_type(filepath)
        finally:
            os.unlink(filepath)
    
    @pytest.mark.skipif(True, reason="Requires tifffile library and test file")
    def test_tiff_file(self):
        """Test with TIFF file (requires tifffile library)."""
        # This test would require creating an actual TIFF file
        # Skipped in standard test suite
        pass
    
    @pytest.mark.skipif(True, reason="Requires zarr library and test file")
    def test_zarr_file(self):
        """Test with Zarr file (requires zarr library)."""
        # This test would require creating an actual Zarr file
        # Skipped in standard test suite
        pass
    
    def test_tiff_extension_variations(self):
        """Test that both .tif and .tiff extensions are recognized."""
        # Create temporary files with different extensions
        for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                filepath = f.name
            
            try:
                # Should recognize as TIFF format (will fail due to invalid content)
                try:
                    get_image_data_type(filepath)
                except (ImportError, Exception) as e:
                    # Expected to fail - we're just checking format recognition
                    assert 'tifffile' in str(e).lower() or 'tiff' in str(e).lower()
            finally:
                os.unlink(filepath)
