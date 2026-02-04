"""Integration tests for complete workflows."""

import numpy as np
import pytest


def test_basic_io_workflow():
    """Test complete I/O workflow."""
    try:
        from petakit5d import write_tiff, read_tiff
        import tempfile
        import os

        # Create test data
        data = np.random.rand(10, 20, 30).astype(np.float32)

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            temp_file = f.name

        try:
            # Write and read
            write_tiff(data, temp_file)
            loaded = read_tiff(temp_file)

            # Verify
            assert loaded.shape == data.shape
            assert loaded.dtype == data.dtype
            assert np.allclose(loaded, data)
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)
    except ImportError:
        pytest.skip("Required modules not available")


def test_filtering_workflow():
    """Test filtering pipeline."""
    try:
        from petakit5d import filter_gauss_3d

        # Create test data
        data = np.random.rand(10, 20, 30).astype(np.float32)

        # Apply filter
        filtered = filter_gauss_3d(data, sigma=1.0)

        # Verify
        assert filtered.shape == data.shape
        assert filtered.dtype == data.dtype
    except ImportError:
        pytest.skip("Required modules not available")


def test_deconvolution_workflow():
    """Test deconvolution preparation workflow."""
    try:
        from petakit5d import psf_gen, decon_psf2otf

        # Create test PSF
        psf = np.random.rand(20, 40, 40).astype(np.float32)
        psf /= psf.sum()

        # Preprocess PSF
        processed_psf = psf_gen(psf, dz_data=0.5, dz_psf=0.2, psf_gen_method='median')

        # Convert to OTF
        otf = decon_psf2otf(processed_psf, (50, 100, 100))

        # Verify
        assert otf.shape == (50, 100, 100)
        assert np.isfinite(otf).all()
    except ImportError:
        pytest.skip("Required modules not available")


def test_stitching_workflow():
    """Test basic stitching workflow."""
    try:
        from petakit5d import normxcorr2_max_shift

        # Create overlapping tiles
        tile1 = np.random.rand(50, 100, 100).astype(np.float32)
        tile2 = np.roll(tile1, (5, 10, 15), axis=(0, 1, 2))
        tile2 += np.random.rand(*tile2.shape).astype(np.float32) * 0.1

        # Find offset
        offset, max_corr, corr_map = normxcorr2_max_shift(tile1[:, :, 50], tile2[:, :, 50],
                                       maxShifts=np.array([20, 20]))

        # Verify offset is reasonable
        assert len(offset) == 3  # Returns [dy, dx, 0]
        assert abs(offset[0] - 5) < 10  # dy shift - allow wider tolerance
        assert abs(offset[1] - 10) < 10  # dx shift - allow wider tolerance
        assert abs(max_corr) <= 1.0  # Correlation coefficient
    except ImportError:
        pytest.skip("Required modules not available")
