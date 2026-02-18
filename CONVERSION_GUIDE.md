# MATLAB to Python Conversion Guide

## What Was Converted

This document describes the initial MATLAB to Python conversion completed for the PetaKit5D library.

### Scope

**Original Repository:**
- 678 MATLAB files
- 114,000+ lines of code
- Complex image processing, microscopy, and analysis functions

**Initial Conversion (Completed):**
- 8 core utility functions
- 761 lines of Python code
- 50 comprehensive unit tests
- Full Python package structure

### Files Converted

The following MATLAB files from the `utils/` directory have been converted to Python:

1. **get_uuid.m** → `petakit5d/utils/uuid_utils.py`
   - Generates unique identifiers
   - Handles platform-specific behavior (Windows path length limitations)

2. **mat2str_comma.m** → `petakit5d/utils/string_utils.py`
   - Converts arrays to comma-separated string representation
   - Supports custom precision formatting

3. **readTextFile.m** → `petakit5d/utils/file_utils.py` (read_text_file)
   - Reads text files line by line
   - Returns list of strings

4. **writeTextFile.m** → `petakit5d/utils/file_utils.py` (write_text_file)
   - Writes text files with automatic batching for large files
   - Supports both single strings and lists

5. **writeJsonFile.m** → `petakit5d/utils/file_utils.py` (write_json_file)
   - Writes JSON files with pretty printing
   - Handles nested dictionaries

6. **dataTypeToByteNumber.m** → `petakit5d/utils/dtype_utils.py`
   - Maps data types to byte sizes
   - Extended with Python-native types (float32, float64)

7. **axis_order_mapping.m** → `petakit5d/utils/axis_utils.py`
   - Maps axis orderings for array transformations
   - Returns 1-based indices to match MATLAB behavior

8. **get_hostname.m** → `petakit5d/utils/system_utils.py`
   - Gets system hostname
   - Cross-platform compatible

## Conversion Patterns

### Key Differences: MATLAB vs Python

| Aspect | MATLAB | Python |
|--------|--------|--------|
| Indexing | 1-based | 0-based (converted internally where needed) |
| Arrays | Built-in | NumPy arrays |
| File I/O | `fopen`, `fwrite`, `fclose` | Context managers (`with open()`) |
| Error Handling | `error()` | `raise Exception()` |
| Type Checking | `isa()`, `ischar()` | `isinstance()` |
| UUID Generation | Java-based or system call | `uuid` module |
| JSON Handling | `jsonencode()` | `json.dump()` |

### Conversion Guidelines

When converting MATLAB functions to Python, follow these patterns:

#### 1. Function Signature
```matlab
% MATLAB
function [output] = myFunction(input1, input2, optionalArg)
    if nargin < 3
        optionalArg = defaultValue;
    end
```

```python
# Python
def my_function(input1: type1, input2: type2, 
                optional_arg: type3 = default_value) -> return_type:
    """Docstring with description."""
```

#### 2. Array Handling
```matlab
% MATLAB
A = [1, 2, 3];
B = A(1);  % 1-based indexing
```

```python
# Python
import numpy as np
A = np.array([1, 2, 3])
B = A[0]  # 0-based indexing
```

#### 3. File I/O
```matlab
% MATLAB
fid = fopen(filename, 'w');
fprintf(fid, 'content');
fclose(fid);
```

```python
# Python
with open(filename, 'w', encoding='utf-8') as f:
    f.write('content')
```

#### 4. Error Handling
```matlab
% MATLAB
if ~condition
    error('Error message');
end
```

```python
# Python
if not condition:
    raise ValueError('Error message')
```

## Testing Strategy

Each converted function has comprehensive unit tests covering:

1. **Happy path** - Normal usage scenarios
2. **Edge cases** - Empty inputs, boundary values
3. **Error cases** - Invalid inputs, file not found, etc.
4. **Platform-specific behavior** - Windows vs Linux/Mac

Example test structure:
```python
class TestMyFunction:
    def test_normal_input(self):
        """Test with normal input."""
        result = my_function(valid_input)
        assert result == expected_output
    
    def test_edge_case(self):
        """Test edge case."""
        result = my_function(edge_case_input)
        assert result == expected_edge_output
    
    def test_invalid_input(self):
        """Test that invalid input raises exception."""
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

## How to Extend the Conversion

### Step 1: Choose Functions to Convert

Start with functions that are:
- Self-contained (minimal dependencies)
- Frequently used
- Have clear inputs and outputs
- Not heavily dependent on MATLAB-specific features

### Step 2: Create Python Module

```bash
# Create a new module file
touch petakit5d/utils/new_module.py
```

### Step 3: Convert Function

1. Copy the MATLAB function header comments
2. Convert function signature to Python
3. Replace MATLAB-specific calls with Python equivalents
4. Add type hints
5. Write comprehensive docstring

### Step 4: Update Package Imports

Edit `petakit5d/utils/__init__.py`:
```python
from .new_module import new_function

__all__ = [
    # ... existing functions
    "new_function",
]
```

### Step 5: Write Tests

```bash
# Create test file
touch tests/test_new_module.py
```

Follow the testing patterns in existing test files.

### Step 6: Verify

```bash
# Run tests
pytest tests/test_new_module.py -v

# Check code imports correctly
python3 -c "from petakit5d.utils import new_function; print(new_function())"
```

## Future Conversion Priorities

Based on usage patterns in the codebase, here are suggested priorities for future conversions:

### High Priority (Core Utilities)
- [ ] `getImageSize.m` - Image dimension retrieval
- [ ] `simplifyPath.m` - Path simplification
- [ ] `batch_file_exist.m` - Batch file existence checking
- [ ] `dir_recursive.m` - Recursive directory listing
- [ ] `getZarrInfo.m` - Zarr metadata reading

### Medium Priority (Image Processing)
- [ ] `getImageBoundingBox.m` - Bounding box calculation
- [ ] `check_cuboids_overlaps.m` - Overlap detection
- [ ] Image data type functions
- [ ] Coordinate extraction functions

### Lower Priority (Specialized Functions)
- [ ] Slurm job status checking
- [ ] Lock file management
- [ ] Color code utilities

### Complex Functions (Require More Effort)
- [ ] Image stitching algorithms
- [ ] Deconvolution routines
- [ ] Point detection and tracking
- [ ] Geometric transformations

## Package Management

### Installing the Package

```bash
# Development installation
pip install -e .

# With development dependencies
pip install -e . && pip install -r requirements-dev.txt
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=petakit5d --cov-report=html

# Specific test file
pytest tests/test_uuid_utils.py -v
```

### Building Distribution

```bash
# Build package
python setup.py sdist bdist_wheel

# Check distribution
twine check dist/*
```

## Notes and Recommendations

1. **Maintain API Compatibility**: Keep function names and parameters similar to MATLAB versions for easier migration

2. **Documentation**: Always include:
   - Original MATLAB function name
   - Author and date from original
   - Clear parameter descriptions
   - Usage examples

3. **Type Hints**: Use type hints throughout for better IDE support and documentation

4. **Testing**: Aim for >90% test coverage for all converted functions

5. **Performance**: While maintaining API compatibility, use Python idioms (list comprehensions, numpy operations) for better performance

6. **Platform Support**: Test on Windows, Linux, and macOS when possible

7. **Dependencies**: Minimize external dependencies. Currently only requires numpy.

## Contributing

When contributing additional conversions:

1. Follow the existing code structure and style
2. Write comprehensive unit tests
3. Update documentation
4. Update `petakit5d/utils/__init__.py` with new exports
5. Add examples to PYTHON_README.md
6. Ensure all tests pass before committing

## Summary

This initial conversion provides:
- ✅ Working Python package structure
- ✅ 8 core utility functions converted
- ✅ 50 comprehensive unit tests (all passing)
- ✅ Installation and testing infrastructure
- ✅ Clear patterns for future conversions
- ✅ Documentation and examples

The foundation is in place for continued conversion of the PetaKit5D MATLAB library to Python.
