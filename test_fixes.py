#!/usr/bin/env python3
"""Quick test script to verify the fixes."""

import numpy as np
import sys
import os
import tempfile

# Add the package to path
sys.path.insert(0, '/home/runner/work/PetaKit5D/PetaKit5D')

print("Testing fixes...")
print()

# Test 1: write_tiff argument order
print("Test 1: write_tiff argument order")
try:
    from petakit5d import write_tiff, read_tiff
    
    data = np.random.rand(10, 20, 30).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        temp_file = f.name
    
    try:
        # Should work with correct order: write_tiff(img, filepath)
        write_tiff(data, temp_file)
        loaded = read_tiff(temp_file)
        
        assert loaded.shape == data.shape
        assert loaded.dtype == data.dtype
        print("✓ write_tiff argument order is correct")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
except Exception as e:
    print(f"✗ write_tiff test failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 2: psf_gen parameter name
print("Test 2: psf_gen parameter name")
try:
    from petakit5d import psf_gen
    
    psf = np.random.rand(20, 40, 40).astype(np.float32)
    psf /= psf.sum()
    
    # Should work with psf_gen_method parameter
    processed_psf = psf_gen(psf, dz_psf=0.2, dz_data=0.5, psf_gen_method='median')
    print("✓ psf_gen parameter name is correct")
except Exception as e:
    print(f"✗ psf_gen test failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 3: normxcorr2_max_shift parameter name
print("Test 3: normxcorr2_max_shift parameter name")
try:
    from petakit5d import normxcorr2_max_shift
    
    tile1 = np.random.rand(50, 100).astype(np.float32)
    tile2 = np.roll(tile1, (5, 10), axis=(0, 1))
    
    # Should work with maxShifts parameter
    offset = normxcorr2_max_shift(tile1, tile2, maxShifts=np.array([20, 20]))
    print(f"✓ normxcorr2_max_shift parameter name is correct, offset: {offset[0]}")
except Exception as e:
    print(f"✗ normxcorr2_max_shift test failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 4: binterp validation for 2D with 1 coordinate
print("Test 4: binterp validation for 2D image with only 1 coordinate")
try:
    from petakit5d import binterp
    
    signal = np.ones((5, 5))
    xi = np.array([1.0])
    
    # Should raise ValueError for 2D image with only 1 coordinate
    try:
        result = binterp(signal, xi)
        print("✗ binterp did not raise ValueError as expected")
    except ValueError as e:
        if "2D interpolation" in str(e):
            print(f"✓ binterp correctly raises ValueError: {e}")
        else:
            print(f"✗ binterp raised wrong ValueError: {e}")
except Exception as e:
    print(f"✗ binterp test failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("All tests completed!")
