# Contributing to PetaKit5D

Thank you for your interest in contributing to PetaKit5D! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository:**

```bash
git clone https://github.com/YOUR_USERNAME/PetaKit5D.git
cd PetaKit5D
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode:**

```bash
pip install -e .[dev]
```

4. **Install pre-commit hooks:**

```bash
pre-commit install
```

## Code Style

We use several tools to maintain code quality:

- **black**: Code formatter (line length: 100)
- **flake8**: Style guide enforcement
- **mypy**: Static type checking

These tools run automatically via pre-commit hooks. You can also run them manually:

```bash
# Format code with black
black petakit5d/ tests/

# Check style with flake8
flake8 petakit5d/ tests/

# Check types with mypy
mypy petakit5d/
```

## Testing

We aim for >95% test coverage. All new code should include tests.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=petakit5d --cov-report=html

# Run specific test file
pytest tests/test_filters.py -v

# Run specific test
pytest tests/test_filters.py::test_filter_gauss_3d -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive names
- Test edge cases and error conditions
- Include docstrings explaining what is tested

Example:

```python
def test_filter_gauss_3d_basic():
    """Test basic 3D Gaussian filtering."""
    image = np.random.rand(10, 10, 10).astype(np.float32)
    result = filter_gauss_3d(image, sigma=1.0)
    
    assert result.shape == image.shape
    assert result.dtype == np.float32
    assert not np.array_equal(result, image)  # Should be different
```

## Pull Request Process

1. **Create a new branch:**

```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes:**
   - Write clear, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Ensure all tests pass:**

```bash
pytest
```

4. **Commit your changes:**

```bash
git add .
git commit -m "Add feature: description"
```

The pre-commit hooks will automatically run. Fix any issues they report.

5. **Push to your fork:**

```bash
git push origin feature/your-feature-name
```

6. **Create a Pull Request:**
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Select your branch
   - Fill in the PR template
   - Link any related issues

## Code Review

All submissions require review. We use GitHub pull requests for this purpose.

Reviewers will check:
- Code quality and style
- Test coverage
- Documentation
- Performance implications
- Backward compatibility

## Documentation

Documentation is built with Sphinx. To build locally:

```bash
cd docs
make html
open _build/html/index.html
```

### Docstring Format

We use Google-style docstrings:

```python
def example_function(param1: int, param2: str) -> bool:
    """Short description of function.
    
    Longer description if needed, explaining the function's purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param1 is negative
    
    Example:
        >>> result = example_function(5, "test")
        >>> print(result)
        True
    """
    if param1 < 0:
        raise ValueError("param1 must be non-negative")
    return len(param2) > param1
```

## Reporting Issues

Use GitHub Issues to report bugs or request features.

### Bug Reports

Include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Relevant code snippets
- Error messages/stack traces

### Feature Requests

Include:
- Clear description of the feature
- Use cases and motivation
- Proposed API (if applicable)
- Any relevant examples

## Questions?

If you have questions:
- Check the documentation
- Search existing issues
- Open a new issue with your question

## License

By contributing, you agree that your contributions will be licensed under the GPL-3.0 License.

Thank you for contributing to PetaKit5D!
