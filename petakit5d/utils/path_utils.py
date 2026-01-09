"""
Path utility functions.

This module provides path manipulation utilities for the PetaKit5D library.
"""

import os
from pathlib import Path


def simplify_path(path: str) -> str:
    """
    Simplify a path: make it absolute, and for folders, remove trailing separator.
    
    This function converts relative paths to absolute paths and normalizes
    the path format. For directories, it ensures there is no trailing
    separator.
    
    Args:
        path: Input path (can be relative or absolute, file or directory)
    
    Returns:
        Simplified absolute path without trailing separator for directories
    
    Raises:
        FileNotFoundError: If the path does not exist
    
    Examples:
        >>> simplify_path('.')  # doctest: +SKIP
        '/current/working/directory'
        >>> simplify_path('./somefile.txt')  # doctest: +SKIP
        '/current/working/directory/somefile.txt'
    """
    # Convert to Path object and resolve to absolute path
    p = Path(path).resolve()
    
    # Check if path exists
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    # Convert to string
    simple_path = str(p)
    
    # For directories, ensure no trailing separator
    # Path.resolve() already removes trailing separators, but be explicit
    if p.is_dir() and simple_path.endswith(os.sep):
        simple_path = simple_path.rstrip(os.sep)
    
    return simple_path
