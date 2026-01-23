"""
Tests for microscope data processing utility functions.
"""

import pytest
import numpy as np
import os
import tempfile
from pathlib import Path

from petakit5d.microscope_data_processing.utils import (
    check_resample_setting,
    estimate_computing_memory,
    group_partial_volume_files
)


class TestCheckResampleSetting:
    """Tests for check_resample_setting function."""
    
    def test_isotropic_objective_scan(self):
        """Test isotropic resampling with objective scan."""
        resample, z_aniso = check_resample_setting(
            'isotropic', None, True, 32.8, 0.108, 0.3
        )
        assert np.allclose(resample, [1.0, 1.0, 1.0])
        assert z_aniso == pytest.approx(0.3 / 0.108)
    
    def test_isotropic_stage_scan(self):
        """Test isotropic resampling with stage scan."""
        resample, z_aniso = check_resample_setting(
            'isotropic', None, False, 32.8, 0.108, 0.3
        )
        assert np.allclose(resample, [1.0, 1.0, 1.0])
        expected_z_aniso = np.sin(np.deg2rad(32.8)) * 0.3 / 0.108
        assert z_aniso == pytest.approx(expected_z_aniso)
    
    def test_given_scalar(self):
        """Test given resample with scalar value."""
        resample, z_aniso = check_resample_setting(
            'given', 1.5, True, 32.8, 0.108, 0.3
        )
        assert np.allclose(resample, [1.5, 1.5, 1.5])
    
    def test_given_two_values(self):
        """Test given resample with two values (XY and Z)."""
        resample, z_aniso = check_resample_setting(
            'given', [1.5, 2.0], False, 32.8, 0.108, 0.3
        )
        assert np.allclose(resample, [1.5, 1.5, 2.0])
    
    def test_given_three_values(self):
        """Test given resample with three values (X, Y, Z)."""
        resample, z_aniso = check_resample_setting(
            'given', [1.0, 1.5, 2.0], True, 32.8, 0.108, 0.3
        )
        assert np.allclose(resample, [1.0, 1.5, 2.0])
    
    def test_given_empty_raises(self):
        """Test that empty resample with 'given' type raises error."""
        with pytest.raises(ValueError, match='must not be empty'):
            check_resample_setting('given', None, True, 32.8, 0.108, 0.3)
        
        with pytest.raises(ValueError, match='must not be empty'):
            check_resample_setting('given', [], True, 32.8, 0.108, 0.3)
    
    def test_xy_isotropic(self):
        """Test xy_isotropic resampling."""
        skew_angle = 32.8
        z_aniso = np.sin(np.deg2rad(skew_angle)) * 0.3 / 0.108
        
        resample, _ = check_resample_setting(
            'xy_isotropic', None, False, skew_angle, 0.108, 0.3
        )
        
        assert resample[0] == pytest.approx(1.0)
        assert resample[1] == pytest.approx(1.0)
        
        # Compute expected Z factor
        theta = np.deg2rad(skew_angle)
        expected_zf = np.sqrt(
            (np.sin(theta)**2 + z_aniso**2 * np.cos(theta)**2) /
            (np.cos(theta)**2 + z_aniso**2 * np.sin(theta)**2)
        )
        assert resample[2] == pytest.approx(expected_zf)
    
    def test_invalid_resample_type(self):
        """Test that invalid resample type raises error."""
        with pytest.raises(ValueError, match='Unknown resample_type'):
            check_resample_setting('invalid', None, True, 32.8, 0.108, 0.3)
    
    def test_given_numpy_array(self):
        """Test given resample with numpy array."""
        resample_in = np.array([1.2, 1.3, 1.4])
        resample, z_aniso = check_resample_setting(
            'given', resample_in, True, 32.8, 0.108, 0.3
        )
        assert np.allclose(resample, [1.2, 1.3, 1.4])


class TestEstimateComputingMemory:
    """Tests for estimate_computing_memory function."""
    
    def test_basic_estimation(self):
        """Test basic memory estimation with provided image size."""
        mem, gpu_mem, raw_size, im_size = estimate_computing_memory(
            'dummy.tif',
            steps=['deskew', 'rotate', 'deconvolution'],
            im_size=(1024, 1024, 500)
        )
        
        assert len(mem) == 3
        assert all(m > 0 for m in mem)
        assert raw_size > 0
        assert not np.isnan(gpu_mem)
        assert gpu_mem > 0
    
    def test_single_step(self):
        """Test estimation for single step."""
        mem, gpu_mem, raw_size, im_size = estimate_computing_memory(
            'dummy.tif',
            steps=['rotate'],
            im_size=(512, 512, 100),
            cuda_decon=False
        )
        
        assert len(mem) == 1
        assert mem[0] > 0
        assert np.isnan(gpu_mem)
    
    def test_large_z_dimension(self):
        """Test that large Z dimension increases deskew memory."""
        mem_small, _, _, _ = estimate_computing_memory(
            'dummy.tif',
            steps=['deskew'],
            im_size=(1024, 1024, 100)
        )
        
        mem_large, _, _, _ = estimate_computing_memory(
            'dummy.tif',
            steps=['deskew'],
            im_size=(1024, 1024, 600)
        )
        
        # Large Z should require more memory
        assert mem_large[0] > mem_small[0]
    
    def test_custom_mem_factors(self):
        """Test custom memory factors."""
        mem, _, _, _ = estimate_computing_memory(
            'dummy.tif',
            steps=['rotate'],
            im_size=(512, 512, 100),
            mem_factors=[10, 3, 10]
        )
        
        # Should use custom factor (3) instead of default (5)
        raw_size = 512 * 512 * 100 * 4 / (1024**3)
        expected = raw_size * 3
        assert mem[0] == pytest.approx(expected, rel=1e-3)
    
    def test_no_cuda_decon(self):
        """Test deconvolution without CUDA."""
        mem, gpu_mem, _, _ = estimate_computing_memory(
            'dummy.tif',
            steps=['deconvolution'],
            im_size=(512, 512, 100),
            cuda_decon=False
        )
        
        assert len(mem) == 1
        assert mem[0] > 0
        assert np.isnan(gpu_mem)
    
    def test_gpu_mem_factor(self):
        """Test custom GPU memory factor."""
        _, gpu_mem1, _, _ = estimate_computing_memory(
            'dummy.tif',
            steps=['deconvolution'],
            im_size=(512, 512, 100),
            cuda_decon=True,
            gpu_mem_factor=1.5
        )
        
        _, gpu_mem2, _, _ = estimate_computing_memory(
            'dummy.tif',
            steps=['deconvolution'],
            im_size=(512, 512, 100),
            cuda_decon=True,
            gpu_mem_factor=2.0
        )
        
        # Higher factor should give more GPU memory
        assert gpu_mem2 > gpu_mem1


class TestGroupPartialVolumeFiles:
    """Tests for group_partial_volume_files function."""
    
    def setup_method(self):
        """Setup temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_file(self, filename, size=1024):
        """Helper to create a test file."""
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(b'0' * size)
        return filepath
    
    def test_no_partial_volumes(self):
        """Test with files that have no parts."""
        self._create_file('image1.tif')
        self._create_file('image2.tif')
        self._create_file('image3.tif')
        
        has_parts, groups, dates, sizes = group_partial_volume_files(
            self.temp_dir, ext='.tif'
        )
        
        assert not has_parts
        assert len(groups) == 3
        assert all(len(g) == 1 for g in groups)
    
    def test_with_partial_volumes(self):
        """Test with files that have parts."""
        self._create_file('volume1.tif')
        self._create_file('volume1_part0001.tif')
        self._create_file('volume1_part0002.tif')
        self._create_file('volume2.tif')
        
        has_parts, groups, dates, sizes = group_partial_volume_files(
            self.temp_dir, ext='.tif'
        )
        
        assert has_parts
        assert len(groups) == 2
        # First group should have main file + 2 parts
        assert len(groups[0]) == 3
        # Second group should have only main file
        assert len(groups[1]) == 1
    
    def test_only_first_timepoint(self):
        """Test filtering for first timepoint only."""
        self._create_file('data_Iter_0000_CamA.tif')
        self._create_file('data_Iter_0001_CamA.tif')
        self._create_file('data_Iter_0002_CamA.tif')
        
        has_parts, groups, dates, sizes = group_partial_volume_files(
            self.temp_dir, ext='.tif', only_first_tp=True
        )
        
        assert not has_parts
        assert len(groups) == 1
        assert 'Iter_0000' in groups[0][0]
    
    def test_channel_patterns(self):
        """Test filtering by channel patterns."""
        self._create_file('data_CamA.tif')
        self._create_file('data_CamB.tif')
        self._create_file('data_CamC.tif')
        
        has_parts, groups, dates, sizes = group_partial_volume_files(
            self.temp_dir, ext='.tif', channel_patterns=['CamA', 'CamB']
        )
        
        assert not has_parts
        assert len(groups) == 2
        # Should only have CamA and CamB, not CamC
        filenames = [g[0] for g in groups]
        assert any('CamA' in f for f in filenames)
        assert any('CamB' in f for f in filenames)
        assert not any('CamC' in f for f in filenames)
    
    def test_empty_directory(self):
        """Test with empty directory."""
        has_parts, groups, dates, sizes = group_partial_volume_files(
            self.temp_dir, ext='.tif'
        )
        
        assert not has_parts
        assert len(groups) == 0
    
    def test_file_fullpath_list(self):
        """Test providing file list instead of directory."""
        file1 = self._create_file('test1.tif')
        file2 = self._create_file('test2.tif')
        
        has_parts, groups, dates, sizes = group_partial_volume_files(
            '', file_fullpath_list=[file1, file2]
        )
        
        assert not has_parts
        assert len(groups) == 2
    
    def test_multiple_volumes_with_parts(self):
        """Test multiple volumes each with parts."""
        # Volume 1
        self._create_file('vol1.tif')
        self._create_file('vol1_part0001.tif')
        
        # Volume 2
        self._create_file('vol2.tif')
        self._create_file('vol2_part0001.tif')
        self._create_file('vol2_part0002.tif')
        
        has_parts, groups, dates, sizes = group_partial_volume_files(
            self.temp_dir, ext='.tif'
        )
        
        assert has_parts
        assert len(groups) == 2
        assert len(groups[0]) == 2  # vol1 + 1 part
        assert len(groups[1]) == 3  # vol2 + 2 parts
    
    def test_sorted_parts(self):
        """Test that parts are sorted correctly."""
        self._create_file('data.tif')
        self._create_file('data_part0003.tif')
        self._create_file('data_part0001.tif')
        self._create_file('data_part0002.tif')
        
        has_parts, groups, dates, sizes = group_partial_volume_files(
            self.temp_dir, ext='.tif'
        )
        
        assert has_parts
        assert len(groups) == 1
        assert len(groups[0]) == 4
        # Check parts are in order
        assert 'part0001' in groups[0][1]
        assert 'part0002' in groups[0][2]
        assert 'part0003' in groups[0][3]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
