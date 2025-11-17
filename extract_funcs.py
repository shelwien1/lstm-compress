#!/usr/bin/env python3
"""
Extract function names from clang -ast-dump output for a specific source file.

Usage: python extract_funcs.py <astdump_file> <target_source_file>
"""

import sys
import re


def parse_ast_dump(astdump_file, target_source):
    """
    Parse clang AST dump and extract function names from the target source file.

    Args:
        astdump_file: Path to the AST dump file
        target_source: Name of the target source file to filter by

    Returns:
        List of function names found in the target source file
    """
    current_file = None
    function_names = []

    # Pattern to match source location like <filename:line:col> or <line:line:col>
    loc_pattern = re.compile(r'<([^>]+)>')

    # Pattern to match FunctionDecl lines
    # Format: [|-|`-]FunctionDecl 0xADDR <location> [line:N:M] [col:N] [used] function_name 'type'
    # The function name can appear after various combinations of line: and col: markers
    func_decl_pattern = re.compile(r'[|`]-?FunctionDecl\s+\S+\s+<[^>]+>(?:\s+(?:line:\d+:\d+|col:\d+))*\s+(?:used\s+|static\s+|inline\s+|extern\s+|constexpr\s+)*(\w+)\s+')

    try:
        with open(astdump_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                # Look for source location information
                loc_match = loc_pattern.search(line)
                if loc_match:
                    location = loc_match.group(1)

                    # Check if this is a full path (contains filename)
                    # Format: "filename:line:col" or "filename:line:col, col:end"
                    # But NOT "line:N:M" or "col:N"
                    parts = location.split(':')
                    if len(parts) >= 2 and parts[0] and not parts[0].startswith('line') and not parts[0].startswith('col') and not parts[0] == '<invalid sloc>':
                        # This is a filename - extract just the basename
                        current_file = parts[0].split('/')[-1].split('\\')[-1]

                # If we're in the target source file, look for FunctionDecl
                if current_file == target_source:
                    func_match = func_decl_pattern.search(line)
                    if func_match:
                        func_name = func_match.group(1)
                        if func_name not in function_names:
                            function_names.append(func_name)

    except FileNotFoundError:
        print(f"Error: File '{astdump_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    return function_names


def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_funcs.py <astdump_file> <target_source_file>", file=sys.stderr)
        sys.exit(1)

    astdump_file = sys.argv[1]
    target_source = sys.argv[2]

    function_names = parse_ast_dump(astdump_file, target_source)

    # Print each function name on a separate line
    for func_name in function_names:
        print(func_name)


if __name__ == '__main__':
    main()
