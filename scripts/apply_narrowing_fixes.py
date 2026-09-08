#!/usr/bin/env python3

# @adai-status: experimental        (one-off codemod for a specific past clang-tidy cleanup; capped by TD-046 — see TECHNICAL_DEBT.md)
# @adai-version: 0.3.0
# @adai-reviewed: 2026-09-07

"""Apply static_cast fixes for cppcoreguidelines-narrowing-conversions warnings.

Reads a clang-tidy warning file and applies fixes to source files.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

# Parse warning file
WARNING_RE = re.compile(
    r'^(.*?):(\d+):(\d+): warning: narrowing conversion from \'([^\']+)\' to (?:signed type |)\'([^\']+)\'.*?(?:\[(.*?)\])?$'
)
BUGPRONE_RE = re.compile(
    r'^(.*?):(\d+):(\d+): warning: (?:result of integer division|performing an implicit widening|either cast from).*?(?:\[(.*?)\])?$'
)


def parse_warnings(warning_file):
    """Parse clang-tidy warning lines into structured data."""
    warnings = defaultdict(list)
    with open(warning_file) as f:
        for line in f:
            line = line.strip()
            m = WARNING_RE.match(line)
            if m:
                filepath, lineno, col, from_type, to_type, check = m.groups()
                warnings[filepath].append({
                    'line': int(lineno),
                    'col': int(col),
                    'from': from_type,
                    'to': to_type,
                    'check': check or 'cppcoreguidelines-narrowing-conversions',
                })
                continue
            m = BUGPRONE_RE.match(line)
            if m:
                filepath, lineno, col, check = m.groups()
                warnings[filepath].append({
                    'line': int(lineno),
                    'col': int(col),
                    'from': 'int',
                    'to': 'float',
                    'check': check or 'bugprone',
                })
    return warnings


def find_expression_end(src, start_col):
    """Find the end of an expression starting at start_col (0-indexed).
    
    Returns end index (exclusive) of the token/expression to wrap.
    Handles: identifiers, member calls (a.b()), method chains, etc.
    """
    i = start_col
    n = len(src)
    
    # Skip whitespace
    while i < n and src[i] in ' \t':
        i += 1
    
    start = i
    depth = 0  # paren depth
    
    # Handle negative number or unary minus
    if i < n and src[i] == '-':
        i += 1
    
    while i < n:
        ch = src[i]
        if ch in '(':
            depth += 1
            i += 1
        elif ch in ')':
            if depth == 0:
                break
            depth -= 1
            i += 1
        elif ch in '[':
            # array subscript - skip to matching ]
            depth2 = 1
            i += 1
            while i < n and depth2 > 0:
                if src[i] == '[':
                    depth2 += 1
                elif src[i] == ']':
                    depth2 -= 1
                i += 1
        elif ch in ',;':
            if depth == 0:
                break
            i += 1
        elif ch in ' \t\n':
            # Stop at whitespace only if not inside parens
            if depth == 0:
                break
            i += 1
        elif ch in '+-*/':
            if depth == 0:
                break
            i += 1
        else:
            i += 1
    
    return start, i


def get_token_at_col(src_line, col_1indexed):
    """Get the token/identifier starting at the given 1-indexed column."""
    col = col_1indexed - 1  # Convert to 0-indexed
    if col >= len(src_line):
        return None, None, None
    
    # Skip forward to start of meaningful token
    i = col
    while i < len(src_line) and src_line[i] in ' \t':
        i += 1
    
    start = i
    
    # Determine token type and find end
    if i < len(src_line) and (src_line[i].isalpha() or src_line[i] == '_'):
        # Identifier - scan full expression including . -> () chains
        while i < len(src_line):
            ch = src_line[i]
            if ch.isalnum() or ch == '_':
                i += 1
            elif ch == '.' and i + 1 < len(src_line) and (src_line[i+1].isalpha() or src_line[i+1] == '_'):
                i += 1  # include the dot
            elif ch == '-' and i + 1 < len(src_line) and src_line[i+1] == '>':
                i += 2  # include ->
            elif ch == ':' and i + 1 < len(src_line) and src_line[i+1] == ':':
                i += 2  # include ::
            elif ch == '(':
                # Function call - find matching )
                depth = 1
                i += 1
                while i < len(src_line) and depth > 0:
                    if src_line[i] == '(':
                        depth += 1
                    elif src_line[i] == ')':
                        depth -= 1
                    i += 1
            elif ch == '[':
                # Array subscript
                depth = 1
                i += 1
                while i < len(src_line) and depth > 0:
                    if src_line[i] == '[':
                        depth += 1
                    elif src_line[i] == ']':
                        depth -= 1
                    i += 1
            else:
                break
        return start, i, src_line[start:i]
    elif i < len(src_line) and src_line[i].isdigit():
        # Number literal
        while i < len(src_line) and (src_line[i].isalnum() or src_line[i] in '.'):
            i += 1
        return start, i, src_line[start:i]
    
    return None, None, None


def determine_cast_type(from_type, to_type):
    """Determine what type to cast to."""
    to_clean = to_type.strip().replace("signed type '", "").replace("'", "")
    if 'float' in to_type:
        return 'float'
    elif 'int' in to_type and 'unsigned' not in to_type:
        return 'int'
    elif 'long' in to_type and 'unsigned' not in to_type:
        return 'long'
    elif 'difference_type' in to_type:
        return 'std::ptrdiff_t'
    elif 'size_t' in to_type or 'unsigned long' in to_type:
        return 'std::size_t'
    elif 'char' in to_type:
        return 'char'
    elif 'double' in to_type:
        return 'double'
    return None


def apply_fixes_to_file(filepath, file_warnings):
    """Apply all fixes to a single file."""
    path = Path(filepath)
    if not path.exists():
        print(f"  SKIP (not found): {filepath}")
        return 0
    
    with open(path) as f:
        lines = f.readlines()
    
    fixed = 0
    modified_lines = {}
    
    # Sort warnings by line desc then col desc so later fixes don't shift positions
    sorted_warnings = sorted(file_warnings, key=lambda w: (w['line'], w['col']), reverse=True)
    
    for w in sorted_warnings:
        lineno = w['line']  # 1-indexed
        col = w['col']      # 1-indexed
        
        if lineno < 1 or lineno > len(lines):
            continue
        
        line_idx = lineno - 1
        src_line = modified_lines.get(line_idx, lines[line_idx]).rstrip('\n')
        
        cast_type = determine_cast_type(w['from'], w['to'])
        if not cast_type:
            print(f"  SKIP (unknown type): {filepath}:{lineno}:{col}")
            continue
        
        # Get token at column
        start, end, token = get_token_at_col(src_line, col)
        if token is None or not token.strip():
            print(f"  SKIP (no token): {filepath}:{lineno}:{col} line={src_line[:80]!r}")
            continue
        
        # Skip if already has static_cast
        if 'static_cast' in token:
            continue
        
        # Build the replacement
        prefix = src_line[:start]
        suffix = src_line[end:]
        new_token = f"static_cast<{cast_type}>({token})"
        new_line = prefix + new_token + suffix + '\n'
        
        modified_lines[line_idx] = new_line
        print(f"  FIX {filepath}:{lineno}:{col}: {token!r} -> {new_token!r}")
        fixed += 1
    
    if fixed > 0:
        # Apply modifications
        result_lines = []
        for i, line in enumerate(lines):
            result_lines.append(modified_lines.get(i, line))
        with open(path, 'w') as f:
            f.writelines(result_lines)
    
    return fixed


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <warnings_file>")
        sys.exit(1)
    
    warnings_file = sys.argv[1]
    all_warnings = parse_warnings(warnings_file)
    
    total_fixed = 0
    for filepath, file_warnings in sorted(all_warnings.items()):
        print(f"\n{filepath} ({len(file_warnings)} warnings)")
        fixed = apply_fixes_to_file(filepath, file_warnings)
        total_fixed += fixed
    
    print(f"\nTotal fixes applied: {total_fixed}")


if __name__ == '__main__':
    main()
