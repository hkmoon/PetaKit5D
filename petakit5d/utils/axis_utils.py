"""
Axis ordering utilities.

Ported from MATLAB axis_order_mapping.m
"""

from typing import Tuple


def axis_order_mapping(input_axis_order: str, 
                       output_axis_order: str = 'yxz') -> Tuple[int, int, int]:
    """
    Map axis order based on the axis order strings for input and output.
    
    Args:
        input_axis_order: Input axis order string (e.g., 'xyz', 'yxz', 'zyx')
        output_axis_order: Output axis order string (default: 'yxz')
        
    Returns:
        Tuple[int, int, int]: Mapping indices for axis reordering
        
    Raises:
        ValueError: If axis orders are invalid
        
    Examples:
        >>> axis_order_mapping('xyz', 'yxz')
        (2, 1, 3)
        >>> axis_order_mapping('zyx', 'xyz')
        (3, 2, 1)
        
    Original MATLAB function: axis_order_mapping.m
    Author: Xiongtao Ruan (10/08/2024)
    """
    valid_orders = ['xyz', 'yxz', 'zyx', 'zxy', 'yzx', 'xzy']
    
    input_axis_order = input_axis_order.lower()
    output_axis_order = output_axis_order.lower()
    
    # Validate input
    if len(input_axis_order) != 3 or not all(c in 'xyz' for c in input_axis_order):
        raise ValueError('Input axis order must contain exactly x, y, and z.')
    
    if input_axis_order not in valid_orders:
        raise ValueError(f'Input axis order must be one of {valid_orders}')
    
    if len(output_axis_order) != 3 or not all(c in 'xyz' for c in output_axis_order):
        raise ValueError('Output axis order must contain exactly x, y, and z.')
        
    if output_axis_order not in valid_orders:
        raise ValueError(f'Output axis order must be one of {valid_orders}')
    
    # Create mapping - MATLAB uses 1-based indexing, Python uses 0-based
    # Return 1-based indices to match MATLAB behavior
    order_mat = tuple(output_axis_order.index(c) + 1 for c in input_axis_order)
    
    return order_mat
