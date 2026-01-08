"""
File I/O utilities.

Ported from MATLAB readTextFile.m, writeTextFile.m, and writeJsonFile.m
"""

import json
from typing import Union, List
from pathlib import Path


def read_text_file(filename: str) -> List[str]:
    """
    Read a text file line by line into a list of strings.
    
    Args:
        filename: Path to the text file
        
    Returns:
        List[str]: List of lines from the file
        
    Raises:
        FileNotFoundError: If the file does not exist
        
    Original MATLAB function: readTextFile.m
    Author: Xiongtao Ruan (04/04/2024)
    """
    filepath = Path(filename)
    
    if not filepath.exists():
        raise FileNotFoundError(f'{filename} does not exist, please check the path!')
    
    with open(filepath, 'r', encoding='utf-8') as f:
        file_lines = [line.rstrip('\n\r') for line in f]
    
    return file_lines


def write_text_file(text_lines: Union[str, List[str]], filename: str, 
                   batch_size: int = 10000) -> None:
    """
    Write text to a file. Accepts either a single string or a list of strings.
    
    Args:
        text_lines: String or list of strings to write
        filename: Path to the output file
        batch_size: Batch size for writing large lists (default: 10000)
        
    Original MATLAB function: writeTextFile.m
    Author: Xiongtao Ruan (04/04/2024)
    """
    filepath = Path(filename)
    
    if isinstance(text_lines, str):
        # Single string
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text_lines)
    elif isinstance(text_lines, list):
        if len(text_lines) <= batch_size:
            # Small list - write all at once
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(text_lines))
        else:
            # Large list - write in batches
            if filepath.exists():
                filepath.unlink()
            
            with open(filepath, 'a', encoding='utf-8') as f:
                n_lines = len(text_lines)
                n_batches = (n_lines + batch_size - 1) // batch_size
                
                for b in range(n_batches):
                    start = b * batch_size
                    end = min((b + 1) * batch_size, n_lines)
                    batch = '\n'.join(text_lines[start:end])
                    f.write(batch)
                    if end < n_lines:
                        f.write('\n')


def write_json_file(data: dict, filename: str) -> None:
    """
    Write a dictionary to a JSON file with pretty printing.
    
    Args:
        data: Dictionary to write to JSON
        filename: Path to the output JSON file
        
    Original MATLAB function: writeJsonFile.m
    Author: Xiongtao Ruan (05/01/2025)
    """
    filepath = Path(filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
