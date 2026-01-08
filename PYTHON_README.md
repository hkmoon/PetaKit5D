# PetaKit5D Python Library

Python port of utility functions from the MATLAB PetaKit5D library for efficient and scalable processing of petabyte-scale 5D live images.

## Overview

This package provides Python implementations of commonly used utility functions from the original MATLAB PetaKit5D library. The functions have been carefully ported to maintain API compatibility while leveraging Python's native capabilities.

## Installation

### From source

```bash
# Install the package
pip install -e .

# Install with development dependencies
pip install -r requirements-dev.txt
```

### Requirements

- Python >= 3.8
- numpy >= 1.20.0

## Usage

```python
from petakit5d.utils import (
    get_uuid,
    mat2str_comma,
    read_text_file,
    write_text_file,
    write_json_file,
    data_type_to_byte_number,
    axis_order_mapping,
    get_hostname
)

# Generate a unique identifier
uuid = get_uuid()

# Convert array to comma-separated string
result = mat2str_comma([1, 2, 3])  # Returns '[1,2,3]'

# Read/write text files
lines = read_text_file('input.txt')
write_text_file(lines, 'output.txt')

# Write JSON files
data = {'key': 'value', 'number': 42}
write_json_file(data, 'output.json')

# Get data type byte size
bytes_count = data_type_to_byte_number('uint16')  # Returns 2

# Map axis ordering
mapping = axis_order_mapping('xyz', 'yxz')  # Returns (2, 1, 3)

# Get system hostname
hostname = get_hostname()
```

## Converted Functions

### UUID Utilities
- `get_uuid()` - Generate UUID strings (truncated on Windows for path compatibility)

### String Utilities
- `mat2str_comma(A, sn)` - Convert arrays to comma-separated strings

### File I/O Utilities
- `read_text_file(filename)` - Read text files line by line
- `write_text_file(text_lines, filename, batch_size)` - Write text files (with batching for large files)
- `write_json_file(data, filename)` - Write JSON files with pretty printing

### Data Type Utilities
- `data_type_to_byte_number(dtype)` - Get byte size for data types

### Axis Utilities
- `axis_order_mapping(input_axis_order, output_axis_order)` - Map axis orderings (e.g., 'xyz' to 'yxz')

### System Utilities
- `get_hostname()` - Get current machine hostname

## Testing

Run the test suite using pytest:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=petakit5d --cov-report=html

# Run specific test file
pytest tests/test_uuid_utils.py -v
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=petakit5d --cov-report=term-missing
```

### Code Structure

```
petakit5d/
├── __init__.py
└── utils/
    ├── __init__.py
    ├── uuid_utils.py
    ├── string_utils.py
    ├── file_utils.py
    ├── dtype_utils.py
    ├── axis_utils.py
    └── system_utils.py

tests/
├── __init__.py
├── test_uuid_utils.py
├── test_string_utils.py
├── test_file_utils.py
├── test_dtype_utils.py
├── test_axis_utils.py
└── test_system_utils.py
```

## Conversion Notes

This is a **starting point** for converting the PetaKit5D MATLAB library to Python. The conversion includes:

- ✅ Core utility functions (8 functions converted)
- ✅ Comprehensive unit tests (50 test cases)
- ✅ Python package structure with setup.py
- ✅ Documentation and examples

### What's Included

The initial conversion focuses on self-contained, reusable utility functions that:
- Have minimal dependencies
- Are frequently used throughout the codebase
- Provide good examples for future conversions

### Future Work

The complete PetaKit5D library contains 678 MATLAB files with 114K+ lines of code. Additional conversions would include:
- Image processing functions
- Microscope data processing pipelines
- Point detection and tracking algorithms
- Stitching and deconvolution routines
- Geometric transformations
- And many more specialized functions

## Original MATLAB Library

This Python library is a port of functions from [PetaKit5D](https://github.com/abcucberkeley/PetaKit5D), developed by Xiongtao Ruan and colleagues.

## Reference

If you use this software, please cite the original paper:

> Ruan, X., Mueller, M., Liu, G., Görlitz, F., Fu, T., Milkie, D.E., Lillvis, J.L., Kuhn, A., Chong, J.G., Hong, J.L., Herr, C.Y.A., Hercule, W., Nienhaus, M., Killilea, A.N., Betzig, E. and Upadhyayula, S. Image processing tools for petabyte-scale light sheet microscopy data. Nature Methods 21, 2342–2352 (2024). https://doi.org/10.1038/s41592-024-02475-4

## License

This project follows the same license as the original PetaKit5D project (GNU General Public License v3.0).
