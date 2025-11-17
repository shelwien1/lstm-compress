#!/usr/bin/env python3
"""
Extract function names from GCC linker map files (generated with -Wl,--cref,-Map=output.map)
Filters out library functions and prints only user-defined function names.
"""

import sys
import re
from pathlib import Path


def is_library_path(path):
    """Check if a path is from a system library."""
    if not path:
        return False

    library_indicators = [
        '/lib/',
        '/usr/lib/',
        '/lib64/',
        '/usr/lib64/',
        'libc.a',
        'libgcc',
        'libstdc++',
        'libm.a',
        'libpthread',
        'crt1.o',
        'crti.o',
        'crtn.o',
        'crtbegin',
        'crtend',
    ]

    path_lower = path.lower()
    return any(indicator in path_lower for indicator in library_indicators)


def is_valid_function_name(name):
    """Check if a name looks like a valid function name."""
    if not name:
        return False

    # Skip special symbols
    if name.startswith('_') and name.startswith('__'):
        # Double underscore usually indicates compiler-generated symbols
        if not name.startswith('__Z'):  # But keep C++ mangled names
            return False

    # Skip common non-function symbols
    skip_patterns = [
        r'^\..*',  # Section names starting with .
        r'.*\$.*',  # Special symbols with $
        r'^_init$',
        r'^_fini$',
        r'^_start$',
        r'^__.*_start$',
        r'^__.*_end$',
        r'.*\.eh_frame.*',
        r'.*_GLOBAL_.*',
        r'^DW\..*',
    ]

    for pattern in skip_patterns:
        if re.match(pattern, name):
            return False

    return True


def extract_functions_from_map(map_file_path):
    """
    Extract non-library function names from GCC map file.

    GCC map files with --cref option have several sections:
    1. Memory Configuration
    2. Linker script and memory map
    3. Cross Reference Table
    """
    functions = set()

    try:
        with open(map_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return functions

    # Split into lines for processing
    lines = content.split('\n')

    # Track what section we're in
    in_cross_ref = False
    in_memory_map = False

    # Pattern to match function entries in different sections
    # Functions typically appear as:
    # - In memory map: address Symbol file.o
    # - In cross reference: Symbol file.o (address)

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # Detect sections
        if 'Cross Reference Table' in line:
            in_cross_ref = True
            in_memory_map = False
            continue
        elif 'Linker script and memory map' in line:
            in_memory_map = True
            in_cross_ref = False
            continue
        elif line_stripped.startswith('Memory Configuration') or \
             line_stripped.startswith('Memory Map') or \
             line_stripped.startswith('Discarded input sections'):
            in_cross_ref = False
            # Don't disable memory map here as we might still be in it

        # Parse cross reference table
        if in_cross_ref:
            # Format is typically: Symbol  file.o (or file.cpp.o)
            # Lines with symbols usually don't start with whitespace
            if line and not line[0].isspace() and line_stripped:
                parts = line_stripped.split()
                if parts:
                    symbol = parts[0]
                    # Check if there's a file reference
                    source_file = ''
                    if len(parts) > 1:
                        source_file = parts[1]

                    # Only include if not from library
                    if not is_library_path(source_file):
                        if is_valid_function_name(symbol):
                            functions.add(symbol)

        # Parse memory map section
        elif in_memory_map:
            # Look for lines with addresses and symbols
            # Format: .text.symbolname or just symbolname
            #         0x0000000000401234   symbolname
            #         file.o

            # Match lines with addresses followed by symbol names
            match = re.match(r'^\s*0x[0-9a-fA-F]+\s+(\S+)', line)
            if match:
                symbol = match.group(1)

                # Look ahead for the source file (usually next line or nearby)
                source_file = ''
                for j in range(i+1, min(i+5, len(lines))):
                    next_line = lines[j].strip()
                    if next_line.endswith('.o') or next_line.endswith('.cpp.o'):
                        source_file = next_line
                        break

                # Filter out library functions
                if not is_library_path(source_file):
                    if is_valid_function_name(symbol):
                        functions.add(symbol)

            # Also look for .text.function_name patterns
            match = re.match(r'^\s*\.text\.(\S+)', line)
            if match:
                symbol = match.group(1)

                # Check context for source file
                source_file = ''
                for j in range(max(0, i-3), min(i+3, len(lines))):
                    ctx_line = lines[j].strip()
                    if ctx_line.endswith('.o') or ctx_line.endswith('.cpp.o'):
                        source_file = ctx_line
                        break

                if not is_library_path(source_file):
                    if is_valid_function_name(symbol):
                        functions.add(symbol)

    return functions


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output.map>", file=sys.stderr)
        sys.exit(1)

    map_file = sys.argv[1]

    if not Path(map_file).exists():
        print(f"Error: File '{map_file}' not found", file=sys.stderr)
        sys.exit(1)

    functions = extract_functions_from_map(map_file)

    # Sort and print
    for func in sorted(functions):
        print(func)

    # Print summary to stderr so it doesn't interfere with piped output
    print(f"\n# Total functions found: {len(functions)}", file=sys.stderr)


if __name__ == '__main__':
    main()
