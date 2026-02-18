"""
Tests for volume processing utilities.
"""

import numpy as np
import pytest
from petakit5d.microscope_data_processing.volume_utils import (
    erode_volume_by_2d_projection,
    process_flatfield_correction_frame,
)


class TestErodeVolumeBy2DProjection:
    """Tests for erode_volume_by_2d_projection function."""
    
    def test_basic_erosion(self):
        """Test basic 3D volume erosion."""
        # Create a simple 3D volume
        vol = np.ones((20, 30, 25), dtype=np.float32)
        esize = 3
        
        vol_eroded = erode_volume_by_2d_projection(vol, esize)
        
        # Check that edges are zeroed
        assert np.all(vol_eroded[:esize, :, :] == 0)
        assert np.all(vol_eroded[-esize:, :, :] == 0)
    
    def test_no_erosion(self):
        """Test with esize=0 (no erosion)."""
        vol = np.random.rand(10, 15, 12)
        vol_original = vol.copy()
        
        vol_eroded = erode_volume_by_2d_projection(vol, esize=0)
        
        # Should be unchanged
        np.testing.assert_array_equal(vol_eroded, vol_original)
    
    def test_2d_input(self):
        """Test with 2D input."""
        vol = np.random.rand(20, 30)
        esize = 2
        
        vol_eroded = erode_volume_by_2d_projection(vol, esize)
        
        # Should handle 2D case
        assert vol_eroded.shape == vol.shape
    
    def test_full_volume(self):
        """Test with fully filled volume."""
        # All positive volume
        vol = np.ones((15, 20, 18), dtype=np.float32)
        esize = 2
        
        vol_eroded = erode_volume_by_2d_projection(vol, esize)
        
        # Edges should be zeroed
        assert np.all(vol_eroded[:esize, :, :] == 0)
        assert np.all(vol_eroded[-esize:, :, :] == 0)
    
    def test_sparse_volume(self):
        """Test with sparse volume."""
        # Sparse volume with some zero regions
        vol = np.zeros((20, 25, 22), dtype=np.float32)
        vol[5:15, 5:20, 5:17] = 1.0
        esize = 2
        
        vol_eroded = erode_volume_by_2d_projection(vol, esize)
        
        # Should erode the edges
        assert vol_eroded.shape == vol.shape
        # Center should still have some values
        assert np.any(vol_eroded[7:13, 7:18, 7:15] > 0)
    
    def test_dtype_preservation(self):
        """Test that dtype is preserved."""
        for dtype in [np.uint8, np.uint16, np.float32, np.float64]:
            vol = np.ones((10, 12, 11), dtype=dtype)
            vol_eroded = erode_volume_by_2d_projection(vol, esize=1)
            assert vol_eroded.dtype == dtype
    
    def test_large_erosion(self):
        """Test with large erosion size."""
        vol = np.ones((20, 25, 22), dtype=np.float32)
        esize = 8
        
        vol_eroded = erode_volume_by_2d_projection(vol, esize)
        
        # Large portions should be zeroed
        assert np.all(vol_eroded[:esize, :, :] == 0)
        assert np.all(vol_eroded[-esize:, :, :] == 0)


class TestProcessFlatfieldCorrectionFrame:
    """Tests for process_flatfield_correction_frame function."""
    
    def test_basic_correction(self):
        """Test basic flat field correction."""
        # Create synthetic data
        frame = np.random.rand(100, 100, 10).astype(np.float32) * 1000 + 100
        ls_image = np.random.rand(100, 100).astype(np.float32) * 0.5 + 0.7
        background = np.ones((100, 100), dtype=np.float32) * 50
        
        corrected = process_flatfield_correction_frame(
            frame, ls_image, background
        )
        
        # Should return same shape
        assert corrected.shape == frame.shape
    
    def test_2d_frame(self):
        """Test with 2D frame."""
        frame = np.random.rand(128, 128).astype(np.float32) * 1000
        ls_image = np.random.rand(128, 128).astype(np.float32) * 0.8 + 0.5
        background = np.ones((128, 128), dtype=np.float32) * 30
        
        corrected = process_flatfield_correction_frame(
            frame, ls_image, background
        )
        
        assert corrected.shape == frame.shape
    
    def test_3d_ls_image(self):
        """Test with 3D LS image (should average to 2D)."""
        frame = np.random.rand(100, 100, 10).astype(np.float32) * 1000
        ls_image = np.random.rand(100, 100, 5).astype(np.float32) * 0.7 + 0.5
        background = np.ones((100, 100), dtype=np.float32) * 40
        
        corrected = process_flatfield_correction_frame(
            frame, ls_image, background
        )
        
        assert corrected.shape == frame.shape
    
    def test_cropping(self):
        """Test with larger LS image (should crop)."""
        frame = np.random.rand(100, 100, 8).astype(np.float32) * 1000
        ls_image = np.random.rand(120, 120).astype(np.float32) * 0.8 + 0.5
        background = np.ones((110, 110), dtype=np.float32) * 35
        
        corrected = process_flatfield_correction_frame(
            frame, ls_image, background
        )
        
        # Should still work with cropping
        assert corrected.shape == frame.shape
    
    def test_const_offset(self):
        """Test with constant offset."""
        frame = np.random.rand(80, 80, 6).astype(np.float32) * 1000
        ls_image = np.random.rand(80, 80).astype(np.float32) * 0.7 + 0.6
        background = np.ones((80, 80), dtype=np.float32) * 45
        const_offset = 100.0
        
        corrected = process_flatfield_correction_frame(
            frame, ls_image, background, const_offset=const_offset
        )
        
        assert corrected.shape == frame.shape
    
    def test_no_background_removal(self):
        """Test without removing background from LS image."""
        frame = np.random.rand(90, 90, 7).astype(np.float32) * 1000
        ls_image = np.random.rand(90, 90).astype(np.float32) * 0.8 + 0.5
        background = np.ones((90, 90), dtype=np.float32) * 40
        
        corrected = process_flatfield_correction_frame(
            frame, ls_image, background, remove_ff_im_background=False
        )
        
        assert corrected.shape == frame.shape
    
    def test_no_ls_rescale(self):
        """Test without rescaling LS image."""
        frame = np.random.rand(85, 85, 5).astype(np.float32) * 1000
        ls_image = np.random.rand(85, 85).astype(np.float32) * 0.7 + 0.6
        background = np.ones((85, 85), dtype=np.float32) * 38
        
        corrected = process_flatfield_correction_frame(
            frame, ls_image, background, ls_rescale=False
        )
        
        assert corrected.shape == frame.shape
    
    def test_dtype_casting(self):
        """Test dtype casting."""
        # Test with uint16 input
        frame = (np.random.rand(70, 70, 4) * 10000).astype(np.uint16)
        ls_image = np.random.rand(70, 70).astype(np.float32) * 0.8 + 0.5
        background = np.ones((70, 70), dtype=np.float32) * 50
        
        corrected = process_flatfield_correction_frame(
            frame, ls_image, background, cast_data_type=True
        )
        
        # Should be cast back to uint16
        assert corrected.dtype == np.uint16
    
    def test_no_dtype_casting(self):
        """Test without dtype casting."""
        frame = (np.random.rand(70, 70, 4) * 10000).astype(np.uint16)
        ls_image = np.random.rand(70, 70).astype(np.float32) * 0.8 + 0.5
        background = np.ones((70, 70), dtype=np.float32) * 50
        
        corrected = process_flatfield_correction_frame(
            frame, ls_image, background, cast_data_type=False
        )
        
        # Should remain float32
        assert corrected.dtype == np.float32
    
    def test_lower_limit(self):
        """Test lower limit parameter."""
        frame = np.random.rand(75, 75, 5).astype(np.float32) * 1000
        ls_image = np.random.rand(75, 75).astype(np.float32) * 0.3  # Some low values
        background = np.ones((75, 75), dtype=np.float32) * 40
        lower_limit = 0.5
        
        corrected = process_flatfield_correction_frame(
            frame, ls_image, background, lower_limit=lower_limit
        )
        
        # Should complete without division by very small numbers
        assert corrected.shape == frame.shape
        assert not np.any(np.isnan(corrected))
        assert not np.any(np.isinf(corrected))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
