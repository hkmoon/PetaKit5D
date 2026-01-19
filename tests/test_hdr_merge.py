"""
Tests for high dynamic range (HDR) image merging functions.
"""

import numpy as np
import pytest
from petakit5d.image_processing.hdr_merge import (
    high_dynamic_range_merge,
    threshold_rosin
)


class TestThresholdRosin:
    """Test Rosin thresholding function."""
    
    def test_basic_threshold(self):
        """Test basic threshold computation."""
        # Create simple bimodal image
        img = np.concatenate([
            np.ones((50, 50)) * 100,
            np.ones((50, 50)) * 200
        ])
        
        threshold = threshold_rosin(img)
        assert 100 < threshold < 200
        
    def test_uniform_image(self):
        """Test with uniform image."""
        img = np.ones((100, 100)) * 150
        threshold = threshold_rosin(img)
        assert threshold == pytest.approx(150.0, rel=1e-1)
    
    def test_zeros(self):
        """Test with all zeros."""
        img = np.zeros((50, 50))
        threshold = threshold_rosin(img)
        assert threshold == 0.0


class TestHighDynamicRangeMerge:
    """Test HDR merging function."""
    
    def test_basic_merge(self):
        """Test basic HDR merging."""
        # Create low and high exposure images
        low = np.random.randint(0, 2000, (50, 50), dtype=np.uint16)
        high = np.random.randint(0, 4095, (50, 50), dtype=np.uint16)
        
        combined, log_mystery, log_img = high_dynamic_range_merge(low, high)
        
        # Check output shapes
        assert combined.shape == low.shape
        assert log_mystery.shape == low.shape
        assert log_img.shape == low.shape
        
        # Check types
        assert combined.dtype == np.float64
        assert log_mystery.dtype == np.float64
        assert log_img.dtype == np.float64
    
    def test_saturated_pixels(self):
        """Test handling of saturated pixels."""
        # Create images with known saturation
        low = np.ones((50, 50)) * 1000
        high = np.ones((50, 50)) * 4095  # All saturated
        
        combined, _, _ = high_dynamic_range_merge(low, high)
        
        # Combined should use low exposure data
        assert np.all(np.isfinite(combined))
    
    def test_no_saturation(self):
        """Test case with no saturated pixels."""
        low = np.random.randint(0, 1000, (50, 50), dtype=np.uint16)
        high = np.random.randint(1000, 3000, (50, 50), dtype=np.uint16)
        
        combined, log_mystery, log_img = high_dynamic_range_merge(low, high)
        
        # Should produce valid results
        assert np.all(np.isfinite(combined))
        assert np.all(np.isfinite(log_mystery))
        assert np.all(np.isfinite(log_img))
    
    def test_mystery_offset_factor(self):
        """Test different mystery offset factors."""
        low = np.random.randint(0, 2000, (50, 50), dtype=np.uint16)
        high = np.random.randint(0, 4095, (50, 50), dtype=np.uint16)
        
        combined1, _, _ = high_dynamic_range_merge(low, high, mystery_offset_factor=1.0)
        combined2, _, _ = high_dynamic_range_merge(low, high, mystery_offset_factor=1.5)
        
        # Different factors should produce different results
        assert not np.allclose(combined1, combined2)
    
    def test_custom_saturation_value(self):
        """Test custom saturation value."""
        low = np.ones((50, 50)) * 1000
        high = np.ones((50, 50)) * 2000
        
        # Use lower saturation threshold
        combined, _, _ = high_dynamic_range_merge(
            low, high, saturation_value=1500
        )
        
        assert np.all(np.isfinite(combined))
    
    def test_shape_mismatch_error(self):
        """Test error on shape mismatch."""
        low = np.ones((50, 50))
        high = np.ones((60, 60))
        
        with pytest.raises(ValueError, match="same shape"):
            high_dynamic_range_merge(low, high)
    
    def test_log_transform(self):
        """Test that log transform produces valid output."""
        low = np.random.randint(100, 2000, (50, 50), dtype=np.uint16)
        high = np.random.randint(1000, 4095, (50, 50), dtype=np.uint16)
        
        _, log_mystery, log_img = high_dynamic_range_merge(low, high)
        
        # Log values should be finite and reasonable
        assert np.all(np.isfinite(log_mystery))
        assert np.all(np.isfinite(log_img))
        assert np.all(log_mystery > 0)  # log of positive values
        assert np.all(log_img > 0)
    
    def test_integer_inputs(self):
        """Test with integer inputs."""
        low = np.array([[1000, 2000], [1500, 1800]], dtype=np.uint16)
        high = np.array([[3000, 4095], [3500, 3800]], dtype=np.uint16)
        
        combined, _, _ = high_dynamic_range_merge(low, high)
        
        assert combined.dtype == np.float64
        assert combined.shape == (2, 2)
    
    def test_float_inputs(self):
        """Test with float inputs."""
        low = np.random.rand(30, 30) * 2000
        high = np.random.rand(30, 30) * 4095
        
        combined, log_mystery, log_img = high_dynamic_range_merge(low, high)
        
        assert np.all(np.isfinite(combined))
        assert np.all(np.isfinite(log_mystery))
        assert np.all(np.isfinite(log_img))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
