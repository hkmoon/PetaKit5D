"""
Data type utilities.

Ported from MATLAB dataTypeToByteNumber.m
"""


def data_type_to_byte_number(dtype: str) -> int:
    """
    Get byte number for the given data type.
    
    Args:
        dtype: Data type string ('uint8', 'uint16', 'single', 'double')
        
    Returns:
        int: Number of bytes for the data type
        
    Raises:
        ValueError: If the data type is not supported
        
    Examples:
        >>> data_type_to_byte_number('uint8')
        1
        >>> data_type_to_byte_number('uint16')
        2
        >>> data_type_to_byte_number('single')
        4
        >>> data_type_to_byte_number('double')
        8
        
    Original MATLAB function: dataTypeToByteNumber.m
    Author: Xiongtao Ruan
    """
    dtype_map = {
        'uint8': 1,
        'uint16': 2,
        'single': 4,
        'float32': 4,  # Added common Python equivalent
        'double': 8,
        'float64': 8,  # Added common Python equivalent
    }
    
    if dtype not in dtype_map:
        raise ValueError(f'Unsupported data type {dtype}')
    
    return dtype_map[dtype]
