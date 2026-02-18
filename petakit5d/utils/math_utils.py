"""
Mathematical utility functions.

This module provides mathematical utility functions for the PetaKit5D library.
"""

import numpy as np
from typing import Literal


def find_good_factor_number(
    given_num: int,
    direction: Literal[-1, 1] = 1,
    allow_odd_seven: bool = False
) -> int:
    """
    Find the nearest number that can only be factorized by 2, 3, 5, or 7.
    
    This is useful for FFT operations which perform better with numbers that
    have small prime factors. The function searches for a "good" number by
    checking if all prime factors are <= 7.
    
    Args:
        given_num: Starting number to search from
        direction: Search direction (1 for higher, -1 for lower)
        allow_odd_seven: If True, allow odd numbers with factor 7.
                        If False, numbers with factor 7 must be even.
    
    Returns:
        The nearest number with only small prime factors (2, 3, 5, 7)
    
    Notes:
        - Some CUDA FFT implementations work better with even numbers
        - The allow_odd_seven flag controls this behavior
    
    Examples:
        >>> find_good_factor_number(100)
        100
        >>> find_good_factor_number(101)
        105
        >>> find_good_factor_number(100, direction=-1)
        100
    """
    if not isinstance(given_num, (int, np.integer)):
        raise TypeError("given_num must be an integer")
    
    if direction not in (-1, 1):
        raise ValueError("direction must be 1 or -1")
    
    good_num = int(given_num)
    
    while True:
        prime_factors = _prime_factors(good_num)
        
        if len(prime_factors) == 0:
            # Handle edge case of 1
            if good_num == 1:
                return 1
            good_num += direction
            continue
        
        max_factor = prime_factors[-1]
        min_factor = prime_factors[0]
        
        # Check conditions:
        # 1. No prime factors > 7
        # 2. If factor 7 exists and allow_odd_seven is False, must be even (min factor > 2 means odd)
        if max_factor <= 7 and (allow_odd_seven or not (min_factor > 2 and max_factor == 7)):
            return good_num
        
        good_num += direction


def _prime_factors(n: int) -> list:
    """
    Compute prime factorization of n in ascending order.
    
    Args:
        n: Integer to factorize
    
    Returns:
        List of prime factors in ascending order
    """
    if n <= 1:
        return []
    
    factors = []
    
    # Check for 2s
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    
    # Check odd factors from 3 onwards
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        i += 2
    
    # If n is a prime greater than 2
    if n > 2:
        factors.append(n)
    
    return factors
