"""
Tests for visualization functions.
"""

import numpy as np
import pytest
from petakit5d.image_processing.visualization import rgb_overlay, z_proj_image


class TestRgbOverlay:
    """Tests for rgb_overlay function."""

    def test_basic_functionality(self):
        """Test basic RGB overlay."""
        img = np.random.rand(50, 50) * 255
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:20, 10:20] = True
        color = (1.0, 0.0, 0.0)  # Red

        rgb = rgb_overlay(img, mask, color)

        assert rgb.shape == (50, 50, 3)
        assert rgb.dtype == np.uint8

    def test_multiple_masks(self):
        """Test with multiple masks and colors."""
        img = np.random.rand(50, 50) * 255

        mask1 = np.zeros((50, 50), dtype=bool)
        mask1[10:20, 10:20] = True

        mask2 = np.zeros((50, 50), dtype=bool)
        mask2[30:40, 30:40] = True

        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]  # Red and green

        rgb = rgb_overlay(img, [mask1, mask2], colors)

        assert rgb.shape == (50, 50, 3)
        assert rgb.dtype == np.uint8

        # Check that masked regions have color
        assert rgb[15, 15, 0] > 0  # Red channel in first mask
        assert rgb[35, 35, 1] > 0  # Green channel in second mask

    def test_overlapping_masks(self):
        """Test with overlapping masks."""
        img = np.random.rand(50, 50) * 255

        mask1 = np.zeros((50, 50), dtype=bool)
        mask1[10:30, 10:30] = True

        mask2 = np.zeros((50, 50), dtype=bool)
        mask2[20:40, 20:40] = True

        colors = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]  # Red and blue

        rgb = rgb_overlay(img, [mask1, mask2], colors)

        assert rgb.shape == (50, 50, 3)
        # Overlapping region should have both colors
        assert rgb[25, 25, 0] > 0 or rgb[25, 25, 2] > 0

    def test_dynamic_range(self):
        """Test with specified dynamic range."""
        img = np.random.rand(50, 50) * 100 + 50  # Range [50, 150]
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:20, 10:20] = True

        rgb = rgb_overlay(img, mask, (1.0, 0.0, 0.0), i_range=(50, 150))

        assert rgb.shape == (50, 50, 3)
        assert rgb.dtype == np.uint8

    def test_single_mask_to_list_conversion(self):
        """Test that single mask is converted to list."""
        img = np.random.rand(50, 50) * 255
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:20, 10:20] = True

        # Pass single mask instead of list
        rgb = rgb_overlay(img, mask, (1.0, 0.0, 0.0))

        assert rgb.shape == (50, 50, 3)

    def test_mask_color_mismatch_error(self):
        """Test error when number of masks doesn't match colors."""
        img = np.random.rand(50, 50) * 255
        masks = [np.zeros((50, 50), dtype=bool), np.zeros((50, 50), dtype=bool)]
        colors = [(1.0, 0.0, 0.0)]  # Only one color

        with pytest.raises(ValueError, match="must match"):
            rgb_overlay(img, masks, colors)

    def test_zero_intensity_in_masked_regions(self):
        """Test that overlays are applied correctly."""
        # Create image with known values
        img = np.ones((50, 50)) * 128
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:20, 10:20] = True

        rgb = rgb_overlay(img, mask, (1.0, 0.0, 0.0))

        # Masked region should show some color
        assert rgb.shape == (50, 50, 3)
        # Unmasked region should show grayscale
        assert rgb[5, 5, 0] == rgb[5, 5, 1]  # R==G in background
        assert rgb[5, 5, 1] == rgb[5, 5, 2]  # G==B in background


class TestZProjImage:
    """Tests for z_proj_image function."""

    def test_max_projection(self):
        """Test maximum intensity projection."""
        stack = np.random.rand(10, 50, 50) * 255

        proj = z_proj_image(stack, 'max')

        assert proj.shape == (50, 50)
        # Maximum projection should be at least as large as any single slice
        assert np.all(proj >= stack[0])

    def test_mean_projection(self):
        """Test mean projection."""
        stack = np.random.rand(10, 50, 50) * 255

        proj = z_proj_image(stack, 'mean')

        assert proj.shape == (50, 50)
        # Mean should be close to manually computed mean
        manual_mean = np.mean(stack, axis=0)
        np.testing.assert_allclose(proj, manual_mean, rtol=1e-5)

    def test_median_projection(self):
        """Test median projection."""
        stack = np.random.rand(10, 50, 50) * 255

        proj = z_proj_image(stack, 'median')

        assert proj.shape == (50, 50)
        # Median should be close to manually computed median
        manual_median = np.nanmedian(stack, axis=0)
        np.testing.assert_allclose(proj, manual_median, rtol=1e-5)

    def test_min_projection(self):
        """Test minimum intensity projection."""
        stack = np.random.rand(10, 50, 50) * 255

        proj = z_proj_image(stack, 'min')

        assert proj.shape == (50, 50)
        # Minimum projection should be at most as large as any single slice
        assert np.all(proj <= stack[0])

    def test_with_nans(self):
        """Test projections with NaN values."""
        stack = np.random.rand(10, 50, 50) * 255
        stack[5, 20:30, 20:30] = np.nan

        # Mean and median should handle NaNs
        mean_proj = z_proj_image(stack, 'mean')
        median_proj = z_proj_image(stack, 'median')

        assert not np.all(np.isnan(mean_proj))
        assert not np.all(np.isnan(median_proj))

    def test_multi_channel(self):
        """Test with multi-channel stack."""
        stack = np.random.rand(10, 50, 50, 3) * 255

        proj = z_proj_image(stack, 'max')

        assert proj.shape == (50, 50, 3)

    def test_alternative_names(self):
        """Test alternative projection type names."""
        stack = np.random.rand(10, 50, 50) * 255

        # Test 'ave' as alternative to 'mean'
        proj_ave = z_proj_image(stack, 'ave')
        proj_mean = z_proj_image(stack, 'mean')
        np.testing.assert_array_equal(proj_ave, proj_mean)

        # Test 'med' as alternative to 'median'
        proj_med = z_proj_image(stack, 'med')
        proj_median = z_proj_image(stack, 'median')
        np.testing.assert_array_equal(proj_med, proj_median)

    def test_invalid_projection_type(self):
        """Test error with invalid projection type."""
        stack = np.random.rand(10, 50, 50)

        with pytest.raises(ValueError, match="Unknown projection type"):
            z_proj_image(stack, 'invalid')

    def test_dimension_error(self):
        """Test error with less than 3D input."""
        with pytest.raises(ValueError, match="at least 3-dimensional"):
            z_proj_image(np.random.rand(50, 50), 'max')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
