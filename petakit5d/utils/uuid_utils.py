"""
UUID generation utilities.

Ported from MATLAB get_uuid.m
"""

import uuid
import random
import platform


def get_uuid() -> str:
    """
    Generate a UUID string.
    
    On Windows, returns only the first 4 characters to avoid long file path issues.
    On other platforms, returns the full UUID.
    
    Returns:
        str: UUID string
        
    Original MATLAB function: get_uuid.m
    Author: Xiongtao Ruan (02/22/2020)
    """
    try:
        uuid_str = str(uuid.uuid4())
    except Exception:
        # Fallback to random number if UUID generation fails
        uuid_str = str(random.randint(0, 2**53 - 1))
    
    # On Windows, truncate to first 4 characters to avoid long path issues
    if platform.system() == "Windows" and len(uuid_str) > 4:
        uuid_str = uuid_str[:4]
    
    return uuid_str
