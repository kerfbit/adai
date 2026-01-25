#!/usr/bin/env python3
"""
Project Gutenberg Text Trimmer

This script removes Project Gutenberg headers and footers from text files,
extracting only the actual book content.
"""

import re
import os
import argparse
from pathlib import Path


def find_content_boundaries(text):
    """
    Find the start and end boundaries of the actual book content.
    
    Args:
        text (str): The full text of the Project Gutenberg file
        
    Returns:
        tuple: (start_index, end_index) or (None, None) if not found
    """
    
    # Common start markers (case-insensitive)
    start_patterns = [
        r'\*\*\*\s*START OF TH[EI]S PROJECT GUTENBERG EBOOK.*?\*\*\*',
        r'\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*',
        r'START OF THIS PROJECT GUTENBERG EBOOK',
        r'START OF THE PROJECT GUTENBERG EBOOK',
        r'\*\*\* START OF THIS PROJECT GUTENBERG',
        r'*** START OF THE PROJECT GUTENBERG'
    ]
    
    # Common end markers (case-insensitive)
    end_patterns = [
        r'\*\*\*\s*END OF TH[EI]S PROJECT GUTENBERG EBOOK.*?\*\*\*',
        r'\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*',
        r'END OF THIS PROJECT GUTENBERG EBOOK',
        r'END OF THE PROJECT GUTENBERG EBOOK',
        r'\*\*\* END OF THIS PROJECT GUTENBERG',
        r'\*\*\* END OF THE PROJECT GUTENBERG'
    ]
    
    start_pos = None
    end_pos = None
    
    # Find start position
    for pattern in start_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            start_pos = match.end()
            break
    
    # Find end position
    for pattern in end_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            end_pos = match.start()
            break
    
    return start_pos, end_pos


def trim_gutenberg_text(input_text):
    """
    Trim Project Gutenberg headers and footers from text.
    
    Args:
        input_text (str): The full Project Gutenberg text
        
    Returns:
        str: The trimmed text containing only the book content
    """
    
    start_pos, end_pos = find_content_boundaries(input_text)
    
    if start_pos is not None and end_pos is not None:
        # Extract content between markers
        content = input_text[start_pos:end_pos]
    elif start_pos is not None:
        # Only start marker found
        content = input_text[start_pos:]
    elif end_pos is not None:
        # Only end marker found
        content = input_text[:end_pos]
    else:
        # No markers found - return original text with a warning
        print("Warning: No Project Gutenberg markers found. Returning original text.")
        content = input_text
    
    # Clean up the content
    content = content.strip()
    
    # Remove excessive whitespace while preserving paragraph breaks
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    
    # Remove trailing whitespace from lines
    content = re.sub(r' +$', '', content, flags=re.MULTILINE)
    
    # Remove lines starting with [ and ending with ]
    content = re.sub(r'^.*\[.*\].*$\n?', '', content, flags=re.MULTILINE)
    
    # Remove lines containing 'produced by' or proofreading notices
    content = re.sub(r'^.*produced by.*$\n?', '', content, flags=re.MULTILINE | re.IGNORECASE)
    content = re.sub(r'^.*proofreading.*$\n?', '', content, flags=re.MULTILINE | re.IGNORECASE)
    content = re.sub(r'^.*pgdp\.net.*$\n?', '', content, flags=re.MULTILINE | re.IGNORECASE)
    content = re.sub(r'^illustrated by.*$\n?', '', content, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove lines containing only spaced out asterisks (replace with single newline)
    content = re.sub(r'^\s*\*\s*\*\s*\*\s*\*\s*\*\s*$\n?', '\n', content, flags=re.MULTILINE)
    
    # Remove line numbering (e.g., "1:1", "12:34")
    content = re.sub(r'\b\d+:\d+\b\s*', '', content)
    
    # Remove indented metadata blocks (like production/copyright notices)
    # Pattern: multiple consecutive lines starting with whitespace
    content = re.sub(r'(^[ \t]+.+$\n?){2,}', '', content, flags=re.MULTILINE)
    
    return content


def process_file(input_path, output_path=None):
    """
    Process a single Project Gutenberg file.
    
    Args:
        input_path (str): Path to the input file
        output_path (str): Path to the output file (optional)
        
    Returns:
        bool: True if successful, False otherwise
    """
    
    try:
        # Read the input file
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Trim the text
        trimmed_text = trim_gutenberg_text(text)
        
        # Determine output path
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}_trimmed{input_file.suffix}"
        
        # Write the trimmed text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(trimmed_text)
        
        print(f"Successfully trimmed: {input_path} -> {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def main():
    """Main function to handle command-line arguments and process files."""
    
    parser = argparse.ArgumentParser(
        description="Trim Project Gutenberg text files to extract only book content"
    )
    
    parser.add_argument(
        'input',
        help='Input file or directory path'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output file or directory path (optional)'
    )
    
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Process directories recursively'
    )
    
    parser.add_argument(
        '--suffix',
        default='_trimmed',
        help='Suffix to add to output filenames (default: _trimmed)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single file
        process_file(str(input_path), args.output)
        
    elif input_path.is_dir():
        # Process directory
        pattern = '**/*.txt' if args.recursive else '*.txt'
        txt_files = list(input_path.glob(pattern))
        
        if not txt_files:
            print(f"No .txt files found in {input_path}")
            return
        
        output_dir = Path(args.output) if args.output else input_path
        output_dir.mkdir(exist_ok=True)
        
        for txt_file in txt_files:
            if args.output:
                # Maintain directory structure in output
                relative_path = txt_file.relative_to(input_path)
                output_file = output_dir / relative_path.parent / f"{relative_path.stem}{args.suffix}{relative_path.suffix}"
                output_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_file = txt_file.parent / f"{txt_file.stem}{args.suffix}{txt_file.suffix}"
            
            process_file(str(txt_file), str(output_file))
    
    else:
        print(f"Error: {input_path} is not a valid file or directory")


if __name__ == "__main__":
    main()
