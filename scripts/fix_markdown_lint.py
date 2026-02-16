#!/usr/bin/env python3
"""Fix markdown lint issues in training-internals.md"""

import re

def fix_markdown_lint(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if current line is a heading
        is_heading = line.strip().startswith('#') and not line.strip().startswith('```')
        
        # Check if previous line exists and is not blank
        prev_line_not_blank = (i > 0 and lines[i-1].strip() != '')
        
        # Check if next line exists
        has_next = i + 1 < len(lines)
        next_line_not_blank = has_next and lines[i+1].strip() != ''
        
        # Check for code fence
        is_code_fence = line.strip().startswith('```')
        
        # Check for list item
        is_list = line.strip() and (line.strip()[0] in ['-', '*'] or 
                                     (len(line.strip()) > 1 and line.strip()[0].isdigit() and line.strip()[1] in ['.', ')']))
        
        # Check for table row
        is_table = '|' in line and i > 0 and '|' in lines[i-1]
        
        # MD022: Add blank line before heading if needed
        if is_heading and prev_line_not_blank and i > 0:
            # Don't add if previous line is also a heading or code fence
            if not lines[i-1].strip().startswith('#') and not lines[i-1].strip().startswith('```'):
                fixed_lines.append('\n')
        
        # MD009: Remove trailing spaces
        line = line.rstrip() + '\n' if line.strip() else '\n'
        
        # Add current line
        fixed_lines.append(line)
        
        # MD022: Add blank line after heading if needed
        if is_heading and has_next and next_line_not_blank:
            next_line = lines[i+1].strip()
            # Don't add if next line is another heading, blank, or code fence
            if not next_line.startswith('#') and not next_line.startswith('```'):
                # Check if there's already a blank line
                if has_next and lines[i+1].strip() != '':
                    # Peek ahead to add blank line before next processing
                    pass
        
        i += 1
    
    # Second pass: fix code blocks, lists, and tables
    final_lines = []
    i = 0
    
    while i < len(fixed_lines):
        line = fixed_lines[i]
        
        # Check if this is a code fence
        if line.strip().startswith('```'):
            # Check if language is specified (MD040)
            if line.strip() == '```':
                # Check context to determine language
                # For now, add 'text' as default for mathematical formulas/plain text
                if i > 0:
                    prev_context = fixed_lines[i-1].lower()
                    if 'formula' in prev_context or 'gradient' in prev_context or 'equation' in prev_context:
                        line = '```text\n'
                    else:
                        line = '```text\n'
            
            # MD031: Ensure blank line before code fence
            if i > 0 and fixed_lines[i-1].strip() != '':
                prev_line = fixed_lines[i-1].strip()
                if not prev_line.startswith('#'):
                    final_lines.append('\n')
            
            final_lines.append(line)
            i += 1
            
            # Find closing fence
            while i < len(fixed_lines):
                final_lines.append(fixed_lines[i])
                if fixed_lines[i].strip().startswith('```'):
                    # MD031: Ensure blank line after code fence
                    if i + 1 < len(fixed_lines) and fixed_lines[i+1].strip() != '':
                        next_line = fixed_lines[i+1].strip()
                        if not next_line.startswith('#') and not next_line.startswith('```'):
                            final_lines.append('\n')
                    break
                i += 1
            i += 1
            continue
        
        # Check for list items (MD032)
        is_list = line.strip() and len(line.strip()) > 0 and (
            line.strip()[0] in ['-', '*'] or 
            (len(line.strip()) > 1 and line.strip()[0].isdigit() and line.strip()[1] in ['.', ')'])
        )
        
        if is_list:
            # Check if this is the start of a list
            prev_is_list = i > 0 and len(fixed_lines[i-1].strip()) > 0 and (
                fixed_lines[i-1].strip()[0] in ['-', '*'] or
                (len(fixed_lines[i-1].strip()) > 1 and fixed_lines[i-1].strip()[0].isdigit() and fixed_lines[i-1].strip()[1] in ['.', ')'])
            )
            
            if not prev_is_list and i > 0 and fixed_lines[i-1].strip() != '':
                # This is start of list, ensure blank line before
                final_lines.append('\n')
        
        # Check if previous line was end of list
        if i > 0:
            prev_is_list = len(fixed_lines[i-1].strip()) > 0 and (
                fixed_lines[i-1].strip()[0] in ['-', '*'] or
                (len(fixed_lines[i-1].strip()) > 1 and fixed_lines[i-1].strip()[0].isdigit() and fixed_lines[i-1].strip()[1] in ['.', ')'])
            )
            curr_is_list = is_list
            
            if prev_is_list and not curr_is_list and line.strip() != '' and not line.strip().startswith('#') and not line.strip().startswith('```'):
                # Previous was list, current is not, ensure blank line
                if i > 0 and final_lines and final_lines[-1].strip() != '':
                    final_lines.append('\n')
        
        final_lines.append(line)
        i += 1
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(final_lines)

if __name__ == '__main__':
    fix_markdown_lint(
        '/home/rodney/Repos/adai/docs/guides/training-internals.md',
        '/home/rodney/Repos/adai/docs/guides/training-internals.md.fixed'
    )
    print("Fixed file written to training-internals.md.fixed")
