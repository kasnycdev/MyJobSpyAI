import re
import os
from pathlib import Path

def fix_title_underline(file_path, line_number, title_line):
    """Fix a title underline in a file by ensuring it matches the title text length exactly."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Get the title length
    title_length = len(title_line)
    
    # Check if there's an overline (previous line)
    if line_number > 1:
        overline_line = lines[line_number - 2].strip()
        if overline_line:
            # Get the overline character
            overline_char = overline_line[0]
            # Create a new overline of the exact title length
            new_overline = overline_char * title_length
            if overline_line != new_overline:
                lines[line_number - 2] = new_overline + '\n'
    
    # Check the underline
    if line_number >= len(lines):
        print(f"Warning: No underline found for title at line {line_number}")
        return
    
    underline_line = lines[line_number].strip()
    
    # If there's no underline, create one using the same character as the overline
    if not underline_line:
        if line_number > 1:
            overline_line = lines[line_number - 2].strip()
            if overline_line:
                underline_char = overline_line[0]
            else:
                underline_char = '='
        else:
            underline_char = '='
        new_underline = underline_char * title_length
        lines[line_number] = new_underline + '\n'
    else:
        # Get the underline character
        underline_char = underline_line[0]
        # Create a new underline of the exact title length
        new_underline = underline_char * title_length
        if underline_line != new_underline:
            lines[line_number] = new_underline + '\n'
    
    # Write the changes back to the file if any were made
    with open(file_path, 'w') as f:
        f.writelines(lines)
    print(f"  Fixed line {line_number}")

def get_title_underline_warnings():
    """Get all title underline warnings from rstcheck."""
    import subprocess
    result = subprocess.run(['rstcheck', '-r', '--warn-unknown-settings', 'docs/'],
                          capture_output=True, text=True)
    warnings = set()  # Use set to avoid duplicates
    pattern = r'^(.*):([0-9]+):.*Title underline too short.$'
    
    for line in result.stderr.split('\n'):
        match = re.match(pattern, line)
        if match:
            file_path = match.group(1)
            line_number = int(match.group(2))
            # Store both the line number and the title line content
            with open(file_path, 'r') as f:
                lines = f.readlines()
                title_line = lines[line_number - 1].strip()
            warnings.add((file_path, line_number, title_line))  # Add tuple to set
    
    # Convert set back to sorted list
    return sorted(warnings, key=lambda x: (x[0], x[1]))

def main():
    # Get all warnings from rstcheck
    warnings = get_title_underline_warnings()
    
    # Group warnings by file
    warnings_by_file = {}
    for file_path, line_number, title_line in warnings:
        if file_path not in warnings_by_file:
            warnings_by_file[file_path] = []
        warnings_by_file[file_path].append((line_number, title_line))
    
    # Process each file
    for file_path, warning_list in warnings_by_file.items():
        print(f"\nProcessing {file_path}...")
        # Sort warnings by line number to process from bottom to top
        for line_number, title_line in sorted(warning_list, reverse=True, key=lambda x: x[0]):
            try:
                fix_title_underline(file_path, line_number, title_line)
                print(f"  Fixed line {line_number}")
            except Exception as e:
                print(f"  Error fixing line {line_number}: {str(e)}")
    
    # Verify all warnings are fixed
    remaining_warnings = get_title_underline_warnings()
    if remaining_warnings:
        print("\nRemaining warnings:")
        for file_path, line_number, title_line in remaining_warnings:
            print(f"{file_path}:{line_number}")
    else:
        print("\nAll title underline warnings have been fixed!")

if __name__ == "__main__":
    main()
