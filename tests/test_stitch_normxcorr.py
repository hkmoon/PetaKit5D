"""
Tests for 3D normalized cross-correlation functions.
"""

import numpy as np
import pytest
from petakit5d.microscope_data_processing.stitch_normxcorr import (
    normxcorr3_fast,
    normxcorr3_max_shift,
)


class TestNormxcorr3Fast:
    """Tests for normxcorr3_fast function."""
    
    def test_basic_3d(self):
        """Test basic 3D normalized cross-correlation."""
        # Create simple template and image
        T = np.ones((5, 5, 5), dtype=np.float64)
        A = np.random.rand(20, 20, 20)
        
        C = normxcorr3_fast(T, A, 'full')
        
        # Check output shape (full correlation)
        assert C.shape == tuple(t + a - 1 for t, a in zip(T.shape, A.shape))
        
        # Check value range
        assert np.all(C >= -1 - 1e-6)
        assert np.all(C <= 1 + 1e-6)
    
    def test_shape_full(self):
        """Test full shape output."""
        T = np.ones((3, 3, 3), dtype=np.float64)
        A = np.ones((10, 10, 10), dtype=np.float64)
        
        C = normxcorr3_fast(T, A, 'full')
        
        # Full shape: (3+10-1, 3+10-1, 3+10-1) = (12, 12, 12)
        assert C.shape == (12, 12, 12)
    
    def test_shape_same(self):
        """Test same shape output."""
        T = np.ones((3, 3, 3), dtype=np.float64)
        A = np.ones((10, 10, 10), dtype=np.float64)
        
        C = normxcorr3_fast(T, A, 'same')
        
        # Same shape as A
        assert C.shape == A.shape
    
    def test_shape_valid(self):
        """Test valid shape output."""
        T = np.ones((3, 3, 3), dtype=np.float64)
        A = np.ones((10, 10, 10), dtype=np.float64)
        
        C = normxcorr3_fast(T, A, 'valid')
        
        # Valid shape: (10-3+1, 10-3+1, 10-3+1) = (8, 8, 8)
        assert C.shape == (8, 8, 8)
    
    def test_perfect_match(self):
        """Test correlation with perfect match."""
        # Create a pattern and embed it in a larger array
        pattern = np.random.rand(5, 5, 5)
        A = np.zeros((20, 20, 20))
        A[7:12, 7:12, 7:12] = pattern
        
        C = normxcorr3_fast(pattern, A, 'full')
        
        # Maximum should be close to 1 (perfect match)
        assert np.max(C) > 0.99
    
    def test_error_template_larger(self):
        """Test error when template is larger than image."""
        T = np.ones((10, 10, 10))
        A = np.ones((5, 5, 5))
        
        with pytest.raises(ValueError, match='template must be smaller'):
            normxcorr3_fast(T, A)
    
    def test_error_invalid_shape(self):
        """Test error with invalid shape parameter."""
        T = np.ones((3, 3, 3))
        A = np.ones((10, 10, 10))
        
        with pytest.raises(ValueError, match='unknown SHAPE'):
            normxcorr3_fast(T, A, 'invalid')
    
    def test_2d_input(self):
        """Test with 2D inputs (should be converted to 3D)."""
        T = np.ones((5, 5), dtype=np.float64)
        A = np.random.rand(20, 20)
        
        C = normxcorr3_fast(T, A, 'full')
        
        # Should work and produce 3D output
        assert C.ndim == 3


class TestNormxcorr3MaxShift:
    """Tests for normxcorr3_max_shift function."""
    
    def test_basic_max_shift(self):
        """Test basic max shift computation."""
        # Create template and image with known offset
        T = np.random.rand(5, 5, 5)
        A = np.zeros((20, 20, 20))
        A[10:15, 10:15, 10:15] = T
        
        # Allow shifts up to 10 pixels
        maxShifts = np.array([10, 10, 10])
        
        max_off, max_corr, C = normxcorr3_max_shift(T, A, maxShifts)
        
        # Check that we found high correlation
        assert max_corr > 0.9
        
        # Check offset shape
        assert max_off.shape == (3,)
    
    def test_symmetric_shifts(self):
        """Test with symmetric shift bounds."""
        T = np.ones((3, 3, 3), dtype=np.float64)
        A = np.random.rand(15, 15, 15)
        
        # Single row: symmetric bounds [-5, 5]
        maxShifts = np.array([5, 5, 5])
        
        max_off, max_corr, C = normxcorr3_max_shift(T, A, maxShifts)
        
        # Should return valid results
        assert isinstance(max_corr, (float, np.floating))
        assert max_off.shape == (3,)
    
    def test_asymmetric_shifts(self):
        """Test with asymmetric shift bounds."""
        T = np.ones((3, 3, 3), dtype=np.float64)
        A = np.random.rand(15, 15, 15)
        
        # Two rows: [lower, upper] bounds
        maxShifts = np.array([[-3, -3, -3], [5, 5, 5]])
        
        max_off, max_corr, C = normxcorr3_max_shift(T, A, maxShifts)
        
        # Should return valid results
        assert isinstance(max_corr, (float, np.floating))
        assert max_off.shape == (3,)
    
    def test_cropped_correlation(self):
        """Test that correlation is properly cropped."""
        T = np.ones((3, 3, 3), dtype=np.float64)
        A = np.ones((10, 10, 10), dtype=np.float64)
        
        maxShifts = np.array([2, 2, 2])
        
        max_off, max_corr, C = normxcorr3_max_shift(T, A, maxShifts)
        
        # C should be cropped based on maxShifts
        # The size depends on the constraint
        assert C.shape[0] <= 12  # Full would be 12
        assert C.shape[1] <= 12
        assert C.shape[2] <= 12
    
    def test_known_offset(self):
        """Test with known template offset."""
        # Create template
        T = np.zeros((5, 5, 5))
        T[1:4, 1:4, 1:4] = 1
        
        # Create image with template at known position
        A = np.zeros((20, 20, 20))
        offset = (5, 7, 3)  # Known offset
        A[offset[0]:offset[0]+5, 
          offset[1]:offset[1]+5, 
          offset[2]:offset[2]+5] = T
        
        maxShifts = np.array([10, 10, 10])
        
        max_off, max_corr, C = normxcorr3_max_shift(T, A, maxShifts)
        
        # Should find high correlation
        assert max_corr > 0.9
    
    def test_small_template(self):
        """Test with small template."""
        T = np.ones((2, 2, 2), dtype=np.float64)
        A = np.random.rand(10, 10, 10)
        
        maxShifts = np.array([3, 3, 3])
        
        max_off, max_corr, C = normxcorr3_max_shift(T, A, maxShifts)
        
        # Should complete without error
        assert max_corr >= -1 and max_corr <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
