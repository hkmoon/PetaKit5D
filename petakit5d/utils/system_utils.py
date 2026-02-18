"""
System utilities.

Ported from MATLAB get_hostname.m
"""

import socket
import platform


def get_hostname() -> str:
    """
    Get the hostname of the current machine.
    
    Returns:
        str: Hostname string
        
    Examples:
        >>> hostname = get_hostname()
        >>> isinstance(hostname, str)
        True
        
    Original MATLAB function: get_hostname.m
    Author: Xiongtao Ruan
    """
    try:
        # Try to get the full hostname
        hostname = socket.gethostname()
    except Exception:
        # Fallback to platform node name
        hostname = platform.node()
    
    return hostname
