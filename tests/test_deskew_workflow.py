"""
Tests for deskew_workflow module.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from petakit5d.microscope_data_processing.deskew_workflow import (
    scmos_camera_flip,
    deskew_data
)
from petakit5d.microscope_data_processing.io import write_tiff


def test_scmos_camera_flip_modes():
    """Test different flip modes."""
    image = np.random.rand(20, 30).astype(np.float32)
    
    # None
    result_none = scmos_camera_flip(image, flip_mode='none')
    assert np.allclose(result_none, image)
    
    # Horizontal
    result_h = scmos_camera_flip(image, flip_mode='horizontal')
    assert np.allclose(result_h, np.flip(image, axis=1))
    
    # Vertical
    result_v = scmos_camera_flip(image, flip_mode='vertical')
    assert np.allclose(result_v, np.flip(image, axis=0))
    
    # Both
    result_both = scmos_camera_flip(image, flip_mode='both')
    assert np.allclose(result_both, np.flip(np.flip(image, axis=0), axis=1))


def test_scmos_camera_flip_3d():
    """Test flip on 3D volume."""
    volume = np.random.rand(10, 20, 30).astype(np.float32)
    
    result = scmos_camera_flip(volume, flip_mode='horizontal')
    
    # Should flip along X axis (axis 2)
    assert result.shape == volume.shape
    assert np.allclose(result, np.flip(volume, axis=2))


def test_deskew_data_basic():
    """Test basic deskew workflow."""
    # Create temporary directory and file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create test data
        data = np.random.rand(10, 20, 30).astype(np.float32)
        input_path = tmppath / 'test_input.tif'
        write_tiff(str(input_path), data)
        
        # Run deskew
        output_dir = tmppath / 'output'
        result = deskew_data(
            input_paths=str(input_path),
            output_dir=str(output_dir),
            angle=32.45,
            dz=0.5,
            rotate=False,  # Skip rotation for speed
            save_deskew=True
        )
        
        # Check results
        assert result['n_files'] == 1
        assert len(result['deskewed_files']) == 1
        assert Path(result['deskewed_files'][0]).exists()


def test_deskew_data_with_rotation():
    """Test deskew with rotation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create test data
        data = np.random.rand(10, 20, 30).astype(np.float32)
        input_path = tmppath / 'test_input.tif'
        write_tiff(str(input_path), data)
        
        # Run deskew with rotation
        output_dir = tmppath / 'output'
        result = deskew_data(
            input_paths=str(input_path),
            output_dir=str(output_dir),
            angle=32.45,
            dz=0.5,
            rotate=True,
            save_deskew=True,
            save_rotate=True
        )
        
        # Check results
        assert len(result['deskewed_files']) == 1
        assert len(result['rotated_files']) == 1
        assert Path(result['deskewed_files'][0]).exists()
        assert Path(result['rotated_files'][0]).exists()


def test_deskew_data_output_validation():
    """Test output file naming."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create test data
        data = np.random.rand(5, 10, 15).astype(np.float32)
        input_path = tmppath / 'test_data.tif'
        write_tiff(str(input_path), data)
        
        # Run deskew
        output_dir = tmppath / 'output'
        result = deskew_data(
            input_paths=str(input_path),
            output_dir=str(output_dir),
            angle=30.0,
            dz=0.4,
            rotate=False
        )
        
        # Check file naming
        deskewed_file = Path(result['deskewed_files'][0])
        assert 'deskewed' in deskewed_file.name
        assert deskewed_file.suffix == '.tif'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
