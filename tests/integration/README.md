# Integration Tests

This directory contains integration tests that verify complete workflows and interactions between multiple functions.

## Running Integration Tests

```bash
pytest tests/integration/ -v
```

## Test Coverage

Integration tests cover:

- **I/O workflows**: Complete read/write cycles
- **Filtering pipelines**: Multi-step filtering operations
- **Deconvolution workflows**: PSF preprocessing and OTF conversion
- **Stitching workflows**: Tile registration and blending

## Writing Integration Tests

Integration tests should:

1. Test complete workflows, not individual functions
2. Verify interactions between multiple components
3. Use realistic data scenarios
4. Check end-to-end functionality
5. Include cleanup of temporary files

Example:

```python
def test_complete_workflow():
    """Test a complete processing workflow."""
    # Load data
    data = load_test_data()
    
    # Process through multiple steps
    filtered = filter_gauss_3d(data, sigma=1.0)
    cropped = crop_3d(filtered, bbox=[...])
    
    # Verify final result
    assert cropped.shape == expected_shape
    assert result_is_valid(cropped)
```
