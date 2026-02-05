"""Test XY cropping in rotate_frame_3d"""
import numpy as np
import sys
sys.path.insert(0, '/home/runner/work/PetaKit5D/PetaKit5D')

from petakit5d.microscope_data_processing.deskew_rotate import rotate_frame_3d

# Test 1: Basic XY cropping
print("Test 1: Basic XY cropping after rotation")
data = np.random.rand(10, 20, 30).astype(np.float32)
print(f"Input shape: {data.shape}")

# Without XY cropping
rotated_no_crop = rotate_frame_3d(data, angle=32.45, dz=0.5, crop_xy=False)
print(f"Without XY crop: {rotated_no_crop.shape}")

# With XY cropping (default)
rotated_with_crop = rotate_frame_3d(data, angle=32.45, dz=0.5, crop_xy=True)
print(f"With XY crop: {rotated_with_crop.shape}")

# Verify XY dimensions are smaller or equal
assert rotated_with_crop.shape[1] <= rotated_no_crop.shape[1], "Y dimension should be <= original"
assert rotated_with_crop.shape[2] <= rotated_no_crop.shape[2], "X dimension should be <= original"
print("✓ XY dimensions reduced as expected\n")

# Test 2: No data loss
print("Test 2: Verify no data loss with XY cropping")
# The cropped version should have no zero rows/columns
has_zero_y = np.any(np.all(rotated_with_crop == 0, axis=(0, 2)))
has_zero_x = np.any(np.all(rotated_with_crop == 0, axis=(0, 1)))
print(f"Has zero Y rows: {has_zero_y}")
print(f"Has zero X columns: {has_zero_x}")
assert not has_zero_y, "Should have no all-zero Y rows"
assert not has_zero_x, "Should have no all-zero X columns"
print("✓ No zero rows/columns at boundaries\n")

# Test 3: Multi-channel support
print("Test 3: Multi-channel XY cropping")
multi_data = np.random.rand(3, 10, 20, 30).astype(np.float32)
print(f"Multi-channel input: {multi_data.shape}")

rotated_multi = rotate_frame_3d(multi_data, angle=32.45, dz=0.5, crop_xy=True, channel_axis=0)
print(f"Multi-channel output: {rotated_multi.shape}")

# Verify channels preserved and XY cropped
assert rotated_multi.shape[0] == 3, "Should preserve 3 channels"
assert rotated_multi.shape[2] <= 20, "Y should be cropped"
assert rotated_multi.shape[3] <= 30, "X should be cropped"
print("✓ Multi-channel XY cropping works\n")

print("=" * 50)
print("All tests passed! ✓")
print("=" * 50)
