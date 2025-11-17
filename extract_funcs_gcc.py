#!/usr/bin/env python3
"""
Extract function names from GCC linker map files (generated with -Wl,--cref,-Map=output.map)
Filters out library functions and prints only user-defined function names.
"""

import sys
import re
from pathlib import Path


def is_library_path(path):
    """Check if a path is from a system library or compiler runtime."""
    if not path:
        return True  # Empty path treated as library

    # Normalize path
    path_lower = path.lower().replace('\\', '/')

    # System/library indicators
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
        'libmingw',
        'libmsvcrt',
        'libkernel',
        'libadvapi',
        'libshell',
        'libuser',
        'crt1.o',
        'crti.o',
        'crtn.o',
        'crtbegin',
        'crtend',
        'crtfastmath',
        'crt2.o',
    ]

    for indicator in library_indicators:
        if indicator in path_lower:
            return True

    return False


def is_valid_function_name(name):
    """Check if a name looks like a valid user function name."""
    if not name:
        return False

    # Skip addresses
    if name.startswith('0x'):
        return False

    # Skip section markers
    if name.startswith('.'):
        return False

    # Skip *fill* and similar linker-generated names
    if name.startswith('*'):
        return False

    # Skip common compiler-generated symbols (but keep user symbols starting with single underscore)
    if name.startswith('__'):
        # Except C++ mangled names
        if not name.startswith('__Z'):
            return False

    # Skip common runtime symbols
    skip_patterns = [
        r'^_init$',
        r'^_fini$',
        r'^_start$',
        r'^mainCRTStartup$',
        r'^WinMainCRTStartup$',
        r'^atexit$',
        r'.*_GLOBAL_.*',
        r'^DW\..*',
        r'^Symbol$',  # Header word
        r'^File$',    # Header word
    ]

    for pattern in skip_patterns:
        if re.match(pattern, name):
            return False

    return True


def extract_functions_from_memory_map(lines):
    """Extract functions from the memory map section."""
    functions = set()
    in_text_section = False
    current_file = ''
    i = 0

    while i < len(lines):
        line = lines[i]
        line_stripped = line.strip()

        # Detect .text section
        if line_stripped.startswith('.text'):
            in_text_section = True

        # Exit .text section
        elif in_text_section and line_stripped and not line[0].isspace() and not line_stripped.startswith('.text'):
            if not line_stripped.startswith('0x') and not line_stripped.startswith('*'):
                in_text_section = False

        # In .text section, look for file references and function definitions
        if in_text_section:
            # Check if this line defines a source file
            # Format: " .text          0xADDR     SIZE file.o"
            match = re.match(r'^\s+\.text\s+0x[0-9a-fA-F]+\s+0x[0-9a-fA-F]+\s+(.+)$', line)
            if match:
                current_file = match.group(1).strip()
                # Next lines may contain function definitions
                j = i + 1
                while j < len(lines):
                    func_line = lines[j]
                    # Function definition format: "                0xADDR                functionName"
                    func_match = re.match(r'^\s+0x[0-9a-fA-F]+\s+(\S.*)$', func_line)
                    if func_match:
                        func_name = func_match.group(1).strip()
                        # Stop if we hit another file reference
                        if func_name.endswith('.o') or func_name.endswith('.a)'):
                            break
                        # Add function if from user code
                        if not is_library_path(current_file) and is_valid_function_name(func_name):
                            functions.add(func_name)
                        j += 1
                    elif func_line.strip().startswith('.text') or (func_line.strip() and not func_line[0].isspace()):
                        break
                    else:
                        j += 1

        i += 1

    return functions


def extract_functions_from_cross_ref(lines):
    """Extract functions from the Cross Reference Table section."""
    functions = set()
    in_cross_ref = False

    for line in lines:
        line_stripped = line.strip()

        # Detect Cross Reference Table
        if 'Cross Reference Table' in line:
            in_cross_ref = True
            continue

        if not in_cross_ref:
            continue

        # Skip header line
        if line_stripped.startswith('Symbol') and 'File' in line_stripped:
            continue

        # Skip empty lines
        if not line_stripped:
            continue

        # Parse cross reference entries
        # Format: "Symbol    File" or "Symbol    File1\n      File2\n      File3"
        # Symbol lines don't start with whitespace
        if line and not line[0].isspace():
            # Split by multiple spaces to separate symbol from file
            parts = re.split(r'\s{2,}', line_stripped, maxsplit=1)
            if len(parts) >= 2:
                symbol = parts[0].strip()
                source_file = parts[1].strip()

                # Check if it's from user code
                if not is_library_path(source_file) and is_valid_function_name(symbol):
                    functions.add(symbol)
            elif len(parts) == 1:
                # Symbol without file on same line - might have file on next line
                symbol = parts[0].strip()
                if is_valid_function_name(symbol):
                    # This might be a valid symbol, add it tentatively
                    # (We could check next lines for file reference if needed)
                    pass

    return functions


def extract_functions_from_map(map_file_path):
    """Extract non-library function names from GCC map file."""
    try:
        with open(map_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return set()

    # Extract from both sections and merge
    functions_map = extract_functions_from_memory_map(lines)
    functions_cref = extract_functions_from_cross_ref(lines)

    # Prefer memory map results as they're more reliable
    # But include cross-ref if they add new functions
    all_functions = functions_map.union(functions_cref)

    return all_functions


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
