#!/usr/bin/env python3
"""
Enhanced markdown linter fix for training-internals.md
Handles all MD rules comprehensively
"""

import re

def is_heading(line):
    """Check if line is a markdown heading"""
    stripped = line.strip()
    if not stripped:
        return False
    # Must start with # but not code/comment
    if stripped.startswith('#') and not stripped.startswith('```') and not stripped.startswith('#include'):
        # Make sure it's not just '#' in middle of line
        if stripped[0] == '#':
            return True
    return False

def is_code_fence(line):
    """Check if line is a code fence"""
    return line.strip().startswith('```')

def is_list_item(line):
    """Check if line is a list item"""
    stripped = line.lstrip()  # Keep indentation consideration
    if not stripped:
        return False
    # Bullet lists
    if len(stripped) > 0 and stripped[0] in ['-', '*', '+']:
        if len(stripped) == 1 or (len(stripped) > 1 and stripped[1] == ' '):
            return True
    # Numbered lists  
    match = re.match(r'^\d+[\.)]\s', stripped)
    if match:
        return True
    return False

def get_list_indent_level(line):
    """Get indentation level of list item"""
    return len(line) - len(line.lstrip())

def fix_markdown(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Ensure file ends with newline
    if lines and not lines[-1].endswith('\n'):
        lines[-1] += '\n'
    
    fixed = []
    in_code_block = False
    prev_was_list = False
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # MD009: Remove trailing whitespace (but keep newline)
        line = line.rstrip()
        if i < len(lines) - 1 or line:  # Keep newlines except maybe last line
            line += '\n'
        
        # Track code block state
        if is_code_fence(line):
            # MD040: Add language specification if missing
            if line.strip() == '```' and not in_code_block:
                line = '```text\n'
            
            # MD031: Blank line before code fence (when opening)
            if not in_code_block:
                if fixed and fixed[-1].strip() != '':
                    # Check what's before
                    if not is_heading(fixed[-1]) and not is_code_fence(fixed[-1]):
                        fixed.append('\n')
            
            fixed.append(line)
            
            # Toggle state
            was_closing = in_code_block
            in_code_block = not in_code_block
            
            # MD031: Blank line after code fence (when closing)
            if was_closing and i + 1 < len(lines):
                next_line = lines[i + 1]
                if next_line.strip() != '' and not is_heading(next_line) and not is_code_fence(next_line):
                    # Check if next line is already blank
                    if i + 1 < len(lines) and lines[i + 1].strip() != '':
                        fixed.append('\n')
            
            i += 1
            continue
        
        # Skip normal processing inside code blocks
        if in_code_block:
            fixed.append(line)
            i += 1
            continue
        
        # Check if current line is list
        curr_is_list = is_list_item(line)
        
        # MD032: Blank line before list starts
        if curr_is_list and not prev_was_list:
            if fixed and fixed[-1].strip() != '':
                if not is_heading(fixed[-1]) and not is_code_fence(fixed[-1]):
                    fixed.append('\n')
        
        # MD032: Blank line after list ends  
        if prev_was_list and not curr_is_list and line.strip() != '':
            if not is_heading(line) and not is_code_fence(line):
                if fixed and fixed[-1].strip() != '':
                    fixed.append('\n')
        
        # MD022: Blank line before heading
        if is_heading(line):
            if fixed and fixed[-1].strip() != '':
                # Don't add if previous was also heading or code fence
                if not is_heading(fixed[-1]) and not is_code_fence(fixed[-1]):
                    fixed.append('\n')
        
        # Add the current line
        fixed.append(line)
        
        # MD022: Blank line after heading
        if is_heading(line) and i + 1 < len(lines):
            next_line = lines[i + 1]
            if next_line.strip() != '' and not is_heading(next_line) and not is_code_fence(next_line):
                # Add blank line after heading
                fixed.append('\n')
        
        # Update state
        prev_was_list = curr_is_list
        i += 1
    
    # Fix table formatting (MD060)
    final_fixed = []
    for line in fixed:
        if '|' in line and not line.strip().startswith('```'):
            # Check if it's actually a table (has more than one |)
            if line.count('|') >= 2:
                # Ensure spaces around pipes
                parts = line.rstrip('\n').split('|')
                formatted_parts = []
                for j, part in enumerate(parts):
                    if j == 0 or j == len(parts) - 1:
                        # Keep leading/trailing parts as-is (usually empty)
                        formatted_parts.append(part)
                    else:
                        # Ensure single space around content
                        formatted_parts.append(' ' + part.strip() + ' ')
                line = '|'.join(formatted_parts).rstrip() + '\n'
        
        final_fixed.append(line)
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(final_fixed)

if __name__ == '__main__':
    input_file = '/home/rodney/Repos/adai/docs/guides/training-internals.md.backup'
    output_file = '/home/rodney/Repos/adai/docs/guides/training-internals.md'
    
    print(f"Fixing markdown lint issues in {input_file}...")
    fix_markdown(input_file, output_file)
    print(f"✅ Fixed file saved to {output_file}")
    print("Run linter again to verify fixes")
