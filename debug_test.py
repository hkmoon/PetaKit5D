#!/usr/bin/env python3
"""Debug script for test_large_z_dimension failure."""

import sys
sys.path.insert(0, '/home/runner/work/PetaKit5D/PetaKit5D')

from petakit5d.microscope_data_processing.utils import estimate_computing_memory

# Run the exact test
print("Testing small Z dimension (100):")
mem_small, gpu_small, raw_small, size_small = estimate_computing_memory(
    'dummy.tif',
    steps=['deskew'],
    im_size=(1024, 1024, 100)
)
print(f"  mem_small: {mem_small}")
print(f"  mem_small[0]: {mem_small[0]}")
print(f"  gpu_small: {gpu_small}")
print(f"  raw_small: {raw_small}")
print(f"  size_small: {size_small}")

print("\nTesting large Z dimension (600):")
mem_large, gpu_large, raw_large, size_large = estimate_computing_memory(
    'dummy.tif',
    steps=['deskew'],
    im_size=(1024, 1024, 600)
)
print(f"  mem_large: {mem_large}")
print(f"  mem_large[0]: {mem_large[0]}")
print(f"  gpu_large: {gpu_large}")
print(f"  raw_large: {raw_large}")
print(f"  size_large: {size_large}")

print(f"\nComparison:")
print(f"  mem_large[0] > mem_small[0]: {mem_large[0] > mem_small[0]}")
print(f"  Assertion would be: assert {mem_large[0]} > {mem_small[0]}")
