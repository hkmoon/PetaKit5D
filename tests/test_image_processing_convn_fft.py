"""
Tests for FFT-based N-dimensional convolution.
"""

import numpy as np
import pytest
from petakit5d.image_processing import convn_fft


def test_convn_fft_1d_full():
    """Test 1D convolution with full output."""
    A = np.array([1, 2, 3, 4, 5])
    B = np.array([1, 1, 1])
    
    result = convn_fft(A, B, shape='full')
    
    # Check shape
    assert result.shape == (7,)  # 5 + 3 - 1
    
    # Compare with numpy convolve
    expected = np.convolve(A, B, mode='full')
    assert np.allclose(result, expected, rtol=1e-10)


def test_convn_fft_1d_same():
    """Test 1D convolution with same output size."""
    A = np.array([1, 2, 3, 4, 5])
    B = np.array([1, 1, 1])
    
    result = convn_fft(A, B, shape='same')
    
    # Check shape matches input
    assert result.shape == A.shape
    
    # Compare with numpy convolve
    expected = np.convolve(A, B, mode='same')
    assert np.allclose(result, expected, rtol=1e-10)


def test_convn_fft_1d_valid():
    """Test 1D convolution with valid output."""
    A = np.array([1, 2, 3, 4, 5])
    B = np.array([1, 1, 1])
    
    result = convn_fft(A, B, shape='valid')
    
    # Check shape
    assert result.shape == (3,)  # 5 - 3 + 1
    
    # Compare with numpy convolve
    expected = np.convolve(A, B, mode='valid')
    assert np.allclose(result, expected, rtol=1e-10)


def test_convn_fft_2d_full():
    """Test 2D convolution with full output."""
    A = np.random.rand(10, 10)
    B = np.random.rand(3, 3)
    
    result = convn_fft(A, B, shape='full')
    
    # Check shape
    assert result.shape == (12, 12)  # (10+3-1, 10+3-1)


def test_convn_fft_2d_same():
    """Test 2D convolution with same output size."""
    A = np.random.rand(10, 10)
    B = np.random.rand(3, 3)
    
    result = convn_fft(A, B, shape='same')
    
    # Check shape matches input
    assert result.shape == A.shape


def test_convn_fft_2d_valid():
    """Test 2D convolution with valid output."""
    A = np.random.rand(10, 10)
    B = np.random.rand(3, 3)
    
    result = convn_fft(A, B, shape='valid')
    
    # Check shape
    assert result.shape == (8, 8)  # (10-3+1, 10-3+1)


def test_convn_fft_3d():
    """Test 3D convolution."""
    A = np.random.rand(20, 20, 10)
    B = np.random.rand(5, 5, 3)
    
    result = convn_fft(A, B, shape='same')
    
    # Check shape matches input
    assert result.shape == A.shape


def test_convn_fft_selective_dims():
    """Test convolution along specific dimensions."""
    A = np.random.rand(10, 10, 5)
    B = np.random.rand(3, 3, 1)
    
    # Convolve only along first two dimensions
    result = convn_fft(A, B, shape='same', dims=(0, 1))
    
    assert result.shape == A.shape


def test_convn_fft_power_of_two():
    """Test convolution with power-of-two padding."""
    A = np.random.rand(10, 10)
    B = np.random.rand(3, 3)
    
    result = convn_fft(A, B, shape='same', use_power_of_two=True)
    
    # Should still get same output shape
    assert result.shape == A.shape


def test_convn_fft_complex_input():
    """Test convolution with complex input."""
    A = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
    B = np.random.rand(3, 3)
    
    result = convn_fft(A, B, shape='same')
    
    # Output should be complex
    assert np.iscomplexobj(result)
    assert result.shape == A.shape


def test_convn_fft_real_output():
    """Test that real inputs give real output."""
    A = np.random.rand(10, 10)
    B = np.random.rand(3, 3)
    
    result = convn_fft(A, B, shape='same')
    
    # Output should be real
    assert np.isrealobj(result)


def test_convn_fft_gaussian_kernel():
    """Test convolution with Gaussian kernel."""
    from scipy.ndimage import gaussian_filter
    
    img = np.random.rand(50, 50)
    
    # Create Gaussian kernel
    sigma = 2.0
    size = int(4 * sigma + 1)
    if size % 2 == 0:
        size += 1
    x = np.arange(size) - size // 2
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    kernel = np.outer(kernel_1d, kernel_1d)
    
    # Convolve
    result = convn_fft(img, kernel, shape='same')
    
    # Compare with scipy (approximately)
    expected = gaussian_filter(img, sigma=sigma, mode='constant')
    
    # Should be reasonably close (different boundary handling)
    central = slice(10, -10)
    assert np.allclose(result[central, central], expected[central, central], rtol=0.1)


def test_convn_fft_identity_kernel():
    """Test convolution with identity kernel."""
    A = np.random.rand(10, 10)
    B = np.array([[0, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0]])
    
    result = convn_fft(A, B, shape='same')
    
    # Should recover original image
    assert np.allclose(result, A, rtol=1e-10)


def test_convn_fft_invalid_shape():
    """Test with invalid shape parameter."""
    A = np.random.rand(10, 10)
    B = np.random.rand(3, 3)
    
    with pytest.raises(ValueError, match="Unknown shape"):
        convn_fft(A, B, shape='invalid')


def test_convn_fft_invalid_dims():
    """Test with invalid dimensions."""
    A = np.random.rand(10, 10)
    B = np.random.rand(3, 3)
    
    with pytest.raises(ValueError, match="Dimension .* out of range"):
        convn_fft(A, B, dims=(5,))


def test_convn_fft_single_dim():
    """Test convolution along a single dimension."""
    A = np.random.rand(10, 10)
    B = np.random.rand(3, 1)
    
    result = convn_fft(A, B, shape='same', dims=0)
    
    assert result.shape == A.shape


def test_convn_fft_zeros():
    """Test convolution with zeros."""
    A = np.zeros((10, 10))
    B = np.random.rand(3, 3)
    
    result = convn_fft(A, B, shape='same')
    
    # Should be all zeros
    assert np.allclose(result, 0, atol=1e-14)


def test_convn_fft_large_kernel():
    """Test convolution with large kernel."""
    A = np.random.rand(50, 50)
    B = np.random.rand(15, 15)
    
    result = convn_fft(A, B, shape='same')
    
    assert result.shape == A.shape


def test_convn_fft_mixed_dimensions_2d_1d():
    """Test 2D array with 1D kernel."""
    A = np.random.rand(10, 10)
    B = np.random.rand(3)
    
    # Should work - B gets reshaped to match A's dimensions
    result = convn_fft(A, B, shape='same')
    
    assert result.shape == A.shape


def test_convn_fft_mixed_dimensions_3d_2d():
    """Test 3D array with 2D kernel."""
    A = np.random.rand(10, 10, 10)
    B = np.random.rand(3, 3)
    
    # Should work - B gets reshaped to match A's dimensions
    result = convn_fft(A, B, shape='same')
    
    assert result.shape == A.shape


def test_convn_fft_dims_specific_axis():
    """Test convolution along specific axis when dimensions differ."""
    A = np.random.rand(10, 20)
    B = np.random.rand(3)
    
    # Convolve only along axis 0
    result = convn_fft(A, B, shape='same', dims=0)
    
    assert result.shape == A.shape


def test_convn_fft_odd_even_kernels():
    """Test with both odd and even sized kernels."""
    A = np.random.rand(20, 20)
    
    # Odd kernel
    B_odd = np.random.rand(5, 5)
    result_odd = convn_fft(A, B_odd, shape='same')
    assert result_odd.shape == A.shape
    
    # Even kernel
    B_even = np.random.rand(4, 4)
    result_even = convn_fft(A, B_even, shape='same')
    assert result_even.shape == A.shape

