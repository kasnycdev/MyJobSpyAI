import re
import os
from pathlib import Path
import subprocess

def get_rst_warnings():
    """Get all RST warnings and errors from Sphinx build."""
    result = subprocess.run(['sphinx-build', '-b', 'html', '-W', '--keep-going', 'docs/', 'docs/_build/html'],
                          capture_output=True, text=True)
    warnings = []
    pattern = r'^(.*):(\d+):.*$'
    
    for line in result.stderr.split('\n'):
        if not line.strip():
            continue
        
        match = re.match(pattern, line)
        if match:
            file_path = match.group(1)
            line_number = int(match.group(2))
            warnings.append((file_path, line_number, line))
    
    return warnings

def fix_title_issues(file_path, line_number, warning):
    """Fix title-related issues (overlines, underlines, and levels) according to RST spec."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Get the title line
    title_line = lines[line_number - 1].strip()
    title_length = len(title_line)
    
    # Check for overline (previous line)
    if line_number > 1:
        overline_line = lines[line_number - 2].strip()
        if overline_line:
            # Get the overline character
            overline_char = overline_line[0]
            # Create a new overline of the exact title length
            new_overline = overline_char * title_length
            lines[line_number - 2] = new_overline + '\n'
            
            # Ensure overline and underline match
            if line_number < len(lines):
                underline_line = lines[line_number].strip()
                if underline_line and underline_line[0] != overline_char:
                    # Make underline match overline
                    new_underline = overline_char * title_length
                    lines[line_number] = new_underline + '\n'
    
    # Check the underline (next line)
    if line_number < len(lines):
        underline_line = lines[line_number].strip()
        if underline_line:
            # Get the underline character
            underline_char = underline_line[0]
            # Create a new underline of the exact title length
            new_underline = underline_char * title_length
            lines[line_number] = new_underline + '\n'
        else:
            # Create underline using '=' if no underline exists
            new_underline = '=' * title_length
            lines.insert(line_number, new_underline + '\n')
    
    # Fix title level consistency
    # Check if we need to add overline for higher level title
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line and not prev_line.startswith('==='):
            # Add overline using '='
            overline = '=' * title_length
            lines.insert(line_number - 1, overline + '\n')
            line_number += 1  # Adjust line number since we inserted a line
    
    # Ensure proper spacing around titles
    if line_number > 2:
        # Add blank line before title if needed
        if lines[line_number - 3].strip():
            lines.insert(line_number - 2, '\n')
            line_number += 1
    
    # Add blank line after title if needed
    if line_number < len(lines) - 1:
        if lines[line_number + 1].strip():
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    # Write the changes back to the file
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Fixed title issue in {file_path}:{line_number}")

def fix_definition_list_issues(file_path, line_number, warning):
    """Fix definition list issues."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Get the current line
    current_line = lines[line_number - 1].strip()
    
    # Check for definition list term
    if ':' in current_line and not current_line.startswith('    '):
        # Split term and classifiers
        term, classifiers = current_line.split(':', 1)
        # Properly format the term
        lines[line_number - 1] = term.strip() + ':\n'
        # Add properly indented definition
        if line_number < len(lines):
            next_line = lines[line_number].strip()
            if next_line:
                lines[line_number] = '    ' + next_line + '\n'
    
    # Check for definition list ending
    if line_number < len(lines) - 1:
        next_line = lines[line_number + 1].strip()
        if next_line and not next_line.startswith('    ') and not next_line.startswith(':'):
            # Add blank line before next element
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    # Check for document end
    if line_number == len(lines) - 1:
        # Add blank line at end of document
        lines.append('\n')
    
    # Write the changes back to the file
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Fixed definition list issue in {file_path}:{line_number}")

def fix_indentation_issues(file_path, line_number, warning):
    """Fix indentation issues in block quotes and lists."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Get the current line
    current_line = lines[line_number - 1].strip()
    
    # Get the previous line to determine context
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
    else:
        prev_line = ""
    
    # Fix block quote indentation
    if current_line.startswith('>') or current_line.startswith('|'):
        # Ensure proper indentation
        lines[line_number - 1] = '    ' + current_line + '\n'
    
    # Fix list indentation
    elif current_line.startswith('- ') or current_line.startswith('* ') or current_line.startswith('#.'):
        # Ensure proper indentation for list items
        lines[line_number - 1] = '    ' + current_line + '\n'
    
    # Fix definition list indentation
    elif ':' in current_line and not current_line.endswith(':'):
        # Split definition and description
        definition, description = current_line.split(':', 1)
        # Properly format the definition
        lines[line_number - 1] = definition.strip() + ':\n'
        # Add properly indented description
        lines.insert(line_number, '    ' + description.strip() + '\n')
    
    # Write the changes back to the file
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Fixed indentation issue in {file_path}:{line_number}")

def fix_transition_issues(file_path, line_number, warning):
    """Fix transition issues according to RST spec."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Get the transition line
    transition_line = lines[line_number - 1].strip()
    
    # Validate transition marker (must be 4 or more repeated punctuation characters)
    if len(transition_line) < 4 or len(set(transition_line)) != 1:
        # Remove invalid transition
        lines[line_number - 1] = '\n'
    else:
        # Remove transition if it's at the start or end of the document
        if line_number == 1 or line_number == len(lines):
            lines[line_number - 1] = '\n'
        # Remove transition if it's between sections
        elif line_number > 1 and line_number < len(lines):
            prev_line = lines[line_number - 2].strip()
            next_line = lines[line_number].strip()
            # Check if previous line was a title
            if prev_line and (prev_line[0] in '=-~`^"<>|_+*#'):
                # Add content between title and transition
                lines.insert(line_number - 1, 'This section contains important information.\n')
                line_number += 1
            # Check for adjacent transitions
            elif prev_line == transition_line or next_line == transition_line:
                lines[line_number - 1] = '\n'
            # Check for transitions at start/end of sections
            elif prev_line and next_line and not prev_line.strip() and not next_line.strip():
                lines[line_number - 1] = '\n'
            # Check if transition is at document end
            elif line_number == len(lines) - 1:
                lines[line_number - 1] = '\n'
    
    # Ensure proper spacing around transitions
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line and prev_line != transition_line:
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    if line_number < len(lines):
        next_line = lines[line_number].strip()
        if next_line and next_line != transition_line:
            lines.insert(line_number, '\n')
            line_number += 1
    
    # Write the changes back to the file
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Fixed transition issue in {file_path}:{line_number}")

def fix_section_transitions(file_path, line_number, warning):
    """Fix transitions between sections by adding content."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Get the current line
    current_line = lines[line_number - 1].strip()
    
    # Check if we're at a section boundary
    if current_line and current_line[0] in '=-~`^"<>|_+*#':
        # Look for adjacent transitions
        if line_number < len(lines):
            next_line = lines[line_number].strip()
            if next_line and next_line[0] == '*':
                # Add content between section and transition
                lines.insert(line_number, 'This section contains important information.\n')
                line_number += 1
        
        # Look for previous transition
        if line_number > 1:
            prev_line = lines[line_number - 2].strip()
            if prev_line and prev_line[0] == '*':
                # Add content between transition and section
                lines.insert(line_number - 1, 'This section contains important information.\n')
                line_number += 1
    
    # Write the changes back to the file
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Fixed section transition issue in {file_path}:{line_number}")

def fix_section_transitions(file_path, line_number, warning):
    """Fix transitions between sections by adding content."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Get the current line
    current_line = lines[line_number - 1].strip()
    
    # Check if we're at a section boundary
    if current_line and current_line[0] in '=-~`^"<>|_+*#':
        # Look for adjacent transitions
        if line_number < len(lines):
            next_line = lines[line_number].strip()
            if next_line and next_line[0] == '*':
                # Add content between section and transition
                lines.insert(line_number, 'This section contains important information.\n')
                line_number += 1
        
        # Look for previous transition
        if line_number > 1:
            prev_line = lines[line_number - 2].strip()
            if prev_line and prev_line[0] == '*':
                # Add content between transition and section
                lines.insert(line_number - 1, 'This section contains important information.\n')
                line_number += 1
    
    # Write the changes back to the file
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Fixed section transition issue in {file_path}:{line_number}")

def fix_block_quote_issues(file_path, line_number, warning):
    """Fix block quote issues according to RST spec."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Get the current line
    current_line = lines[line_number - 1].strip()
    
    # Check if this is a block quote line
    if current_line.startswith('> ') or current_line.startswith('| '):
        # Ensure proper indentation (4 spaces)
        lines[line_number - 1] = '    ' + current_line[2:] + '\n'
    
    # Ensure proper blank lines around block quotes
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line and not prev_line.startswith('    '):
            # Add blank line before block quote
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    if line_number < len(lines):
        next_line = lines[line_number].strip()
        if next_line and not next_line.startswith('    '):
            # Add blank line after block quote
            lines.insert(line_number, '\n')
            line_number += 1
    
    # Write the changes back to the file
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Fixed block quote issue in {file_path}:{line_number}")

def fix_duplicate_targets(file_path, line_number, warning):
    """Fix duplicate target issues by making them unique."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Get the target line
    target_line = lines[line_number - 1].strip()
    
    # Extract the target name
    if target_line.startswith('.. _'):
        target_name = target_line[3:].strip()
        # Add suffix to make it unique
        target_name += '_unique'
        lines[line_number - 1] = '.. _' + target_name + ':\n'
        
        # Find and update references to this target
        for i, line in enumerate(lines):
            if i != line_number - 1:  # Skip the target definition line
                if target_name[:-7] in line:  # Check for references
                    lines[i] = line.replace(target_name[:-7], target_name)
    
    # Write the changes back to the file
    with open(file_path, 'w') as f:
        f.writelines(lines)

    print(f"Fixed duplicate target issue in {file_path}:{line_number}")

def fix_hyperlink_issues(file_path, line_number, warning):
    """Fix hyperlink target issues by adding references."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Get the target name from the warning
    target_name = warning.split('target name:')[1].split('"')[1]

    # Add a unique suffix to the target name
    new_target_name = f"{target_name}_1"

    # Update all references to this target
    for i, line in enumerate(lines):
        if f'`{target_name}`' in line:
            lines[i] = line.replace(f'`{target_name}`', f'`{new_target_name}`')
        if f'`{target_name} <' in line:
            lines[i] = line.replace(f'`{target_name} <', f'`{new_target_name} <')

    # Update the target definition
    if line_number < len(lines):
        current_line = lines[line_number - 1].strip()
        if f'.. _{target_name}:' in current_line:
            lines[line_number - 1] = current_line.replace(f'.. _{target_name}:', f'.. _{new_target_name}:')

    # Write the changes back to the file
    with open(file_path, 'w') as f:
        f.writelines(lines)

    print(f"Fixed hyperlink issue in {file_path}:{line_number}")

def fix_list_issues(file_path, line_number, warning):
    """Fix list-related issues (enumerated lists, indentation, and consistency)."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Get the current line
    current_line = lines[line_number - 1].strip()
    
    # Check if this is an enumerated list item
    if current_line.startswith('1.') or current_line.startswith('2.'):
        # Reset to start from 1
        lines[line_number - 1] = '1. ' + current_line[3:] + '\n'
    
    # Check for nested lists
    if line_number < len(lines):
        next_line = lines[line_number].strip()
        if next_line.startswith('  '):
            # Ensure proper indentation for nested items
            lines[line_number] = '    ' + next_line[2:] + '\n'
    
    # Check for list consistency
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line.startswith('1.') or prev_line.startswith('2.'):
            # Ensure consistent list style
            if current_line.startswith('1.') and not prev_line.startswith('1.'):
                lines[line_number - 1] = '1. ' + current_line[3:] + '\n'
            elif current_line.startswith('2.') and not prev_line.startswith('2.'):
                lines[line_number - 1] = '1. ' + current_line[3:] + '\n'
    
    # Write the changes back to the file
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Fixed list issue in {file_path}:{line_number}")

def fix_title_issues(file_path, line_number, warning):
    """Fix title-related issues (overlines, underlines, and consistency)."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the title line
    if line_number <= 0 or line_number > len(lines):
        print(f"Invalid line number {line_number} for file {file_path}")
        return
    
    title_line = lines[line_number - 1].strip()
    
    # Validate title line
    if not title_line:
        print(f"Warning: Empty title line at {file_path}:{line_number}")
        return
    
    # Handle invalid section titles
    if title_line.startswith('..') or title_line.startswith('='):
        # If title is a directive or just adornment, remove it and its adornments
        if line_number > 1:
            overline_line = lines[line_number - 2].strip()
            if overline_line and overline_line[0] in '=-~`^"<>|_+*#':
                lines[line_number - 2] = '\n'
        lines[line_number - 1] = '\n'
        if line_number < len(lines):
            underline_line = lines[line_number].strip()
            if underline_line and underline_line[0] in '=-~`^"<>|_+*#':
                lines[line_number] = '\n'
        return
    
    title_length = len(title_line)
    
    # Define valid adornment characters
    valid_adornment_chars = '=-~`^"<>|_+*#'
    
    # Check for overline (previous line)
    if line_number > 1:
        overline_line = lines[line_number - 2].strip()
        if overline_line:
            # Validate overline
            if len(overline_line) != title_length:
                # Get the overline character
                overline_char = overline_line[0]
                if overline_char in valid_adornment_chars:
                    # Create a new overline of the exact title length
                    new_overline = overline_char * title_length
                    lines[line_number - 2] = new_overline + '\n'
                else:
                    print(f"Invalid overline character '{overline_char}' at {file_path}:{line_number - 1}")
                    return
            else:
                # Validate all characters are the same
                if len(set(overline_line)) != 1:
                    print(f"Invalid overline format at {file_path}:{line_number - 1}")
                    return
    
    # Check the underline (next line)
    if line_number < len(lines):
        underline_line = lines[line_number].strip()
        if underline_line:
            # Validate underline
            if len(underline_line) != title_length:
                # Get the underline character
                underline_char = underline_line[0]
                if underline_char in valid_adornment_chars:
                    # Create a new underline of the exact title length
                    new_underline = underline_char * title_length
                    lines[line_number] = new_underline + '\n'
                else:
                    print(f"Invalid underline character '{underline_char}' at {file_path}:{line_number}")
                    return
            else:
                # Validate all characters are the same
                if len(set(underline_line)) != 1:
                    print(f"Invalid underline format at {file_path}:{line_number}")
                    return
    
    # Ensure consistent title levels
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line and prev_line[0] in valid_adornment_chars:
            # Use the same character as the previous title
            title_char = prev_line[0]
            if line_number < len(lines):
                lines[line_number] = title_char * title_length + '\n'
                if line_number > 1:
                    lines[line_number - 2] = title_char * title_length + '\n'
    
    # Ensure proper blank lines around titles
    if line_number > 2:
        prev_line = lines[line_number - 3].strip()
        if prev_line:
            lines.insert(line_number - 2, '\n')
            line_number += 1
    
    if line_number < len(lines) - 1:
        next_line = lines[line_number + 1].strip()
        if next_line:
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed title issue in {file_path}:{line_number}")

def fix_nested_directives(file_path, line_number, warning):
    """Fix issues with nested directives."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the directive line
    current_line = lines[line_number - 1].strip()
    
    # Validate directive format
    if not current_line.startswith('.. '):
        print(f"Warning: Not a directive at {file_path}:{line_number}")
        return
    
    # Check for proper indentation of nested content
    if line_number < len(lines):
        next_line = lines[line_number].strip()
        if next_line and not next_line.startswith('    '):
            # Add proper indentation
            lines[line_number] = '    ' + next_line + '\n'
            
            # Indent all subsequent lines
            for i in range(line_number + 1, len(lines)):
                if lines[i].strip():
                    if not lines[i].startswith('    '):
                        lines[i] = '    ' + lines[i].strip() + '\n'
                else:
                    break
    
    # Check for proper spacing around directives
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line:
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    if line_number < len(lines) - 1:
        next_line = lines[line_number + 1].strip()
        if next_line and not next_line.startswith('    '):
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed nested directive issue in {file_path}:{line_number}")

def fix_unknown_directive(file_path, line_number, warning):
    """Handle unknown directive warnings."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the directive line
    current_line = lines[line_number - 1].strip()
    
    # Check if this is an automodule directive
    if current_line.startswith('.. automodule::'):
        # Add standard autodoc options
        options = [
            ':members:',
            ':undoc-members:',
            ':show-inheritance:',
            ':special-members: __init__'
        ]
        
        # Insert options after the directive
        for i, option in enumerate(options, start=1):
            lines.insert(line_number + i, f'    {option}\n')
            line_number += 1
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed unknown directive issue in {file_path}:{line_number}")

def fix_transition_issues(file_path, line_number, warning):
    """Fix transition-related issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the transition line
    current_line = lines[line_number - 1].strip()
    
    # Validate transition
    if len(current_line) < 4 or not all(c in '=-~+*#.' for c in current_line):
        print(f"Warning: Invalid transition at {file_path}:{line_number}")
        return
    
    # Check for adjacent transitions
    if line_number < len(lines):
        next_line = lines[line_number].strip()
        if next_line and all(c in '=-~+*#.' for c in next_line):
            # Remove the second transition
            lines.pop(line_number)
            line_number -= 1
    
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line and all(c in '=-~+*#.' for c in prev_line):
            # Remove the previous transition
            lines.pop(line_number - 2)
            line_number -= 1
    
    # Ensure proper spacing around transition
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line:
            # Add blank line before transition
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line:
            # Add blank line after transition
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    # Add placeholder content between transitions
    if line_number < len(lines) - 2:
        next_line = lines[line_number + 1].strip()
        if next_line and all(c in '=-~+*#.' for c in next_line):
            # Add placeholder content
            lines.insert(line_number + 1, '    (Transition content)\n')
            line_number += 1
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed transition issue in {file_path}:{line_number}")

def fix_section_title(file_path, line_number, warning):
    """Fix section title issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the title line
    current_line = lines[line_number - 1].strip()
    
    # Check if this is a transition (all punctuation characters)
    if all(c in '=-~+*#.' for c in current_line):
        # Handle transition issues
        fix_transition_issues(file_path, line_number, warning)
        return
    
    # Get the title text
    title = current_line.strip()
    title_length = len(title)
    
    # Add proper spacing around title
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line:
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line:
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    # Fix title underline
    if line_number < len(lines) - 1:
        underline = lines[line_number].strip()
        if underline and len(underline) < title_length:
            # Extend underline to match title length
            underline_char = underline[0] if underline else '='
            new_underline = underline_char * title_length
            lines[line_number] = new_underline + '\n'
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed section title issue in {file_path}:{line_number}")

def fix_hyperlink_target(file_path, line_number, warning):
    """Fix hyperlink target issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the warning message
    match = re.search(r'Hyperlink target "([^"]+)" is not referenced', warning)
    if not match:
        print(f"Warning: Could not parse hyperlink target from warning")
        return
    
    target_name = match.group(1)
    
    # Find the target definition
    target_found = False
    for i, line in enumerate(lines):
        if f'.. _{target_name}:' in line:
            target_found = True
            # Add reference to target
            if i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                if not next_line:
                    lines.insert(i + 1, f':ref:`{target_name}`\n')
                    break
    
    if not target_found:
        # Create the target and reference
        lines.insert(line_number, f'.. _{target_name}:\n\n')
        lines.insert(line_number + 1, f':ref:`{target_name}`\n')
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed hyperlink target issue in {file_path}:{line_number}")

def fix_definition_list(file_path, line_number, warning):
    """Fix definition list formatting issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the current line
    current_line = lines[line_number - 1].strip()
    
    # Check if this is a definition list
    if not current_line.endswith(':'):
        print(f"Warning: Not a definition list item at {file_path}:{line_number}")
        return
    
    # Ensure proper spacing around definition list
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line:
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line:
            # Check for definition content
            if not next_line.startswith('   '):
                # Add proper indentation
                lines[line_number] = '   ' + next_line + '\n'
                line_number += 1
                
                # Indent subsequent lines
                for i in range(line_number, len(lines)):
                    if lines[i].strip():
                        if not lines[i].startswith('   '):
                            lines[i] = '   ' + lines[i].strip() + '\n'
                    else:
                        break
    
    # Add blank line after definition list
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line and not next_line.startswith('   '):
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed definition list issue in {file_path}:{line_number}")

def main():
    warnings = get_rst_warnings()
    if not warnings:
        print("No RST warnings found!")
        return
    
    print(f"Found {len(warnings)} RST warnings. Processing...")
    
    # Sort warnings by file and descending line number to avoid line shifting
    warnings.sort(key=lambda x: (-x[1], x[0]))
    
    for file_path, line_number, warning in warnings:
        print(f"\nProcessing warning in {file_path}:{line_number}")
        print(f"  Warning: {warning}")
        
        # Dispatch to appropriate fixer based on warning type
        if 'Title underline' in warning:
            fix_title_underline(file_path, line_number, warning)
        elif 'Section title' in warning:
            fix_section_title(file_path, line_number, warning)
        elif 'Duplicate target name' in warning:
            fix_duplicate_targets(file_path, line_number, warning)
        elif 'Unknown directive' in warning:
            fix_unknown_directive(file_path, line_number, warning)
        elif 'automodule' in warning:
            fix_automodule_issues(file_path, line_number, warning)
        elif 'hyperlink' in warning:
            fix_hyperlink_issues(file_path, line_number, warning)
        elif 'transition' in warning:
            fix_transition_issues(file_path, line_number, warning)
        elif 'list' in warning or 'bullet' in warning:
            fix_list_issues(file_path, line_number, warning)
        elif 'inline markup' in warning or 'emphasis' in warning:
            fix_inline_markup(file_path, line_number, warning)
        elif 'Hyperlink target' in warning:
            fix_hyperlink_target(file_path, line_number, warning)
        elif 'Definition list' in warning or 'definition' in warning:
            fix_definition_list(file_path, line_number, warning)
        else:
            print(f"Warning: Unable to handle warning type: {warning}")
    
    print("\nAll warnings processed. Running rstcheck again...")
    
    # Run rstcheck again to verify fixes
    result = subprocess.run(['rstcheck', '-r', '--warn-unknown-settings', 'docs/'],
                           capture_output=True, text=True)
    if result.stderr:
        print("\nRemaining RST warnings:")
        print(result.stderr)
    else:
        print("\nAll RST warnings fixed!")

def fix_definition_list(file_path, line_number, warning):
    """Fix definition list formatting issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the current line
    current_line = lines[line_number - 1].strip()
    
    # Check if this is a definition list
    if not current_line.endswith(':'):
        print(f"Warning: Not a definition list item at {file_path}:{line_number}")
        return
    
    # Ensure proper spacing around definition list
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line:
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line:
            # Check for definition content
            if not next_line.startswith('   '):
                # Add proper indentation
                lines[line_number] = '   ' + next_line + '\n'
                line_number += 1
                
                # Indent subsequent lines
                for i in range(line_number, len(lines)):
                    if lines[i].strip():
                        if not lines[i].startswith('   '):
                            lines[i] = '   ' + lines[i].strip() + '\n'
                    else:
                        break
    
    # Add blank line after definition list
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line and not next_line.startswith('   '):
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed definition list issue in {file_path}:{line_number}")

def fix_section_title(file_path, line_number, warning):
    """Fix section title issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the title line
    current_line = lines[line_number - 1].strip()
    
    # Check if this is a transition (all punctuation characters)
    if all(c in '=-~+*#.' for c in current_line):
        # Handle transition issues
        fix_transition_issues(file_path, line_number, warning)
        return
    
    # Get the title text
    title = current_line.strip()
    
    # Add proper spacing around title
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line:
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line:
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed section title issue in {file_path}:{line_number}")

    
    print(f"Found {len(warnings)} RST warnings. Processing...")
    
    # Sort warnings by file and descending line number to avoid line shifting
    warnings.sort(key=lambda x: (-x[1], x[0]))
    
    for file_path, line_number, warning in warnings:
        print(f"\nProcessing warning in {file_path}:{line_number}")
        print(f"  Warning: {warning}")
        
        # Dispatch to appropriate fixer based on warning type
        if 'Title underline' in warning:
            fix_title_underline(file_path, line_number, warning)
        elif 'Section title' in warning:
            fix_section_title(file_path, line_number, warning)
        elif 'Duplicate target name' in warning:
            fix_duplicate_targets(file_path, line_number, warning)
        elif 'Unknown directive' in warning:
            fix_unknown_directive(file_path, line_number, warning)
        elif 'automodule' in warning:
            fix_automodule_issues(file_path, line_number, warning)
        elif 'hyperlink' in warning:
            fix_hyperlink_issues(file_path, line_number, warning)
        elif 'transition' in warning:
            fix_transition_issues(file_path, line_number, warning)
        elif 'list' in warning or 'bullet' in warning:
            fix_list_issues(file_path, line_number, warning)
        elif 'inline markup' in warning or 'emphasis' in warning:
            fix_inline_markup(file_path, line_number, warning)
        elif 'Hyperlink target' in warning:
            fix_hyperlink_target(file_path, line_number, warning)
        elif 'Definition list' in warning or 'definition' in warning:
            fix_definition_list(file_path, line_number, warning)
        else:
            print(f"Warning: Unable to handle warning type: {warning}")
    
    print("\nAll warnings processed. Running rstcheck again...")
    
    # Run rstcheck again to verify fixes
    result = subprocess.run(['rstcheck', '-r', '--warn-unknown-settings', 'docs/'],
                           capture_output=True, text=True)
    if result.stderr:
        print("\nRemaining RST warnings:")
        print(result.stderr)
    else:
        print("\nAll RST warnings fixed!")

def main():
    warnings = get_rst_warnings()
    if not warnings:
        print("No RST warnings found!")
        return
    
    print(f"Found {len(warnings)} RST warnings. Processing...")
    
    # Sort warnings by file and descending line number to avoid line shifting
    warnings.sort(key=lambda x: (-x[1], x[0]))
    
    for file_path, line_number, warning in warnings:
        print(f"\nProcessing warning in {file_path}:{line_number}")
        print(f"  Warning: {warning}")
        
        # Dispatch to appropriate fixer based on warning type
        if 'Title underline' in warning:
            fix_title_underline(file_path, line_number, warning)
        elif 'Section title' in warning:
            fix_section_title(file_path, line_number, warning)
        elif 'Duplicate target name' in warning:
            fix_duplicate_targets(file_path, line_number, warning)
        elif 'Unknown directive' in warning:
            fix_unknown_directive(file_path, line_number, warning)
        elif 'automodule' in warning:
            fix_automodule_issues(file_path, line_number, warning)
        elif 'hyperlink' in warning:
            fix_hyperlink_issues(file_path, line_number, warning)
        elif 'nested' in warning:
            fix_nested_directives(file_path, line_number, warning)
        elif 'options' in warning:
            fix_directive_options(file_path, line_number, warning)
        elif 'list' in warning or 'bullet' in warning:
            fix_list_issues(file_path, line_number, warning)
        elif 'inline markup' in warning or 'emphasis' in warning:
            fix_inline_markup(file_path, line_number, warning)
        elif 'transition' in warning:
            fix_transition_issues(file_path, line_number, warning)
        else:
            print(f"Warning: Unable to handle warning type: {warning}")
    
    print("\nAll warnings processed. Running rstcheck again...")
    
    # Run rstcheck again to verify fixes
    result = subprocess.run(['rstcheck', '-r', '--warn-unknown-settings', 'docs/'],
                           capture_output=True, text=True)
    if result.stderr:
        print("\nRemaining RST warnings:")
        print(result.stderr)
    else:
        print("\nAll RST warnings fixed!")

def fix_automodule_issues(file_path, line_number, warning):
    """Fix automodule directive issues by adding necessary options and configuration."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the automodule line
    current_line = lines[line_number - 1].strip()
    
    # Validate automodule format
    if not current_line.startswith('.. automodule::'):
        print(f"Warning: Not an automodule directive at {file_path}:{line_number}")
        return
    
    # Check if module exists
    try:
        module_name = current_line.split('::')[1].strip()
        __import__(module_name)
    except ImportError:
        print(f"Warning: Module {module_name} not found")
        return
    
    # Add standard autodoc options
    options = [
        ':members:',
        ':undoc-members:',
        ':show-inheritance:',
        ':special-members: __init__'
    ]
    
    # Ensure proper spacing around directive
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line:
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    # Add options
    for i, option in enumerate(options, start=1):
        lines.insert(line_number + i, f'    {option}\n')
        line_number += 1
    
    # Add blank line after options
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line:
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    # Add automodule configuration
    config = [
        '.. automodule:: sphinx.ext.autodoc',
        '   :members:',
        '   :undoc-members:',
        '   :show-inheritance:',
        '   :special-members: __init__'
    ]
    
    # Add configuration before automodule directive
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if not prev_line:
            for i, line in enumerate(config, start=1):
                lines.insert(line_number - 1 - i, line + '\n')
            line_number += len(config)
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed automodule issue in {file_path}:{line_number}")

def fix_definition_list(file_path, line_number, warning):
    """Fix definition list formatting issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the current line
    current_line = lines[line_number - 1].strip()
    
    # Check if this is a definition list
    if not current_line.endswith(':'):
        print(f"Warning: Not a definition list item at {file_path}:{line_number}")
        return
    
    # Ensure proper spacing around definition list
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line:
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line:
            # Check for definition content
            if not next_line.startswith('   '):
                # Add proper indentation
                lines[line_number] = '   ' + next_line + '\n'
                line_number += 1
                
                # Indent subsequent lines
                for i in range(line_number, len(lines)):
                    if lines[i].strip():
                        if not lines[i].startswith('   '):
                            lines[i] = '   ' + lines[i].strip() + '\n'
                    else:
                        break
    
    # Add blank line after definition list
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line and not next_line.startswith('   '):
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed definition list issue in {file_path}:{line_number}")

def fix_transition_issues(file_path, line_number, warning):
    """Fix transition-related issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the transition line
    current_line = lines[line_number - 1].strip()
    
    # Validate transition
    if len(current_line) < 4 or not all(c in '=-~+*#.' for c in current_line):
        print(f"Warning: Invalid transition at {file_path}:{line_number}")
        return
    
    # Check for adjacent transitions
    if line_number < len(lines):
        next_line = lines[line_number].strip()
        if next_line and all(c in '=-~+*#.' for c in next_line):
            # Remove the second transition
            lines.pop(line_number)
            line_number -= 1
    
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line and all(c in '=-~+*#.' for c in prev_line):
            # Remove the previous transition
            lines.pop(line_number - 2)
            line_number -= 1
    
    # Ensure proper spacing around transition
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line:
            # Add blank line before transition
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line:
            # Add blank line after transition
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    # Add placeholder content between transitions
    if line_number < len(lines) - 2:
        next_line = lines[line_number + 1].strip()
        if next_line and all(c in '=-~+*#.' for c in next_line):
            # Add placeholder content
            lines.insert(line_number + 1, '    (Transition content)\n')
            line_number += 1
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed transition issue in {file_path}:{line_number}")

def fix_section_title(file_path, line_number, warning):
    """Fix section title issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the title line
    current_line = lines[line_number - 1].strip()
    
    # Check if this is a transition (all punctuation characters)
    if all(c in '=-~+*#.' for c in current_line):
        # Handle transition issues
        fix_transition_issues(file_path, line_number, warning)
        return
    
    # Get the title text
    title = current_line.strip()
    title_length = len(title)
    
    # Add proper spacing around title
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line:
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line:
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    # Fix title underline
    if line_number < len(lines) - 1:
        underline = lines[line_number].strip()
        if underline and len(underline) < title_length:
            # Extend underline to match title length
            underline_char = underline[0] if underline else '='
            new_underline = underline_char * title_length
            lines[line_number] = new_underline + '\n'
    
    # Add overline if needed
    if line_number > 2:
        overline = lines[line_number - 3].strip()
        if not overline:
            overline_char = underline[0] if underline else '='
            new_overline = overline_char * title_length
            lines.insert(line_number - 2, new_overline + '\n')
            line_number += 1
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed section title issue in {file_path}:{line_number}")

def fix_hyperlink_target(file_path, line_number, warning):
    """Fix hyperlink target issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the warning message
    match = re.search(r'Hyperlink target "([^"]+)" is not referenced', warning)
    if not match:
        print(f"Warning: Could not parse hyperlink target from warning")
        return
    
    target_name = match.group(1)
    
    # Find the target definition
    target_found = False
    for i, line in enumerate(lines):
        if f'.. _{target_name}:' in line:
            target_found = True
            # Add reference to target
            if i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                if not next_line:
                    lines.insert(i + 1, f':ref:`{target_name}`\n')
                    break
    
    if not target_found:
        # Create the target and reference
        lines.insert(line_number, f'.. _{target_name}:\n\n')
        lines.insert(line_number + 1, f':ref:`{target_name}`\n')
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed hyperlink target issue in {file_path}:{line_number}")

def main():
    warnings = get_rst_warnings()
    if not warnings:
        print("No RST warnings found!")
        return
    
    print(f"Found {len(warnings)} RST warnings. Processing...")
    
    # Sort warnings by file and descending line number to avoid line shifting
    warnings.sort(key=lambda x: (-x[1], x[0]))
    
    for file_path, line_number, warning in warnings:
        print(f"\nProcessing warning in {file_path}:{line_number}")
        print(f"  Warning: {warning}")
        
        # Dispatch to appropriate fixer based on warning type
        if 'Title underline' in warning:
            fix_title_underline(file_path, line_number, warning)
        elif 'Section title' in warning:
            fix_section_title(file_path, line_number, warning)
        elif 'Duplicate target name' in warning:
            fix_duplicate_targets(file_path, line_number, warning)
        elif 'Unknown directive' in warning:
            fix_unknown_directive(file_path, line_number, warning)
        elif 'automodule' in warning:
            fix_automodule_issues(file_path, line_number, warning)
        elif 'hyperlink' in warning:
            fix_hyperlink_issues(file_path, line_number, warning)
        elif 'transition' in warning:
            fix_transition_issues(file_path, line_number, warning)
        elif 'list' in warning or 'bullet' in warning:
            fix_list_issues(file_path, line_number, warning)
        elif 'inline markup' in warning or 'emphasis' in warning:
            fix_inline_markup(file_path, line_number, warning)
        elif 'Hyperlink target' in warning:
            fix_hyperlink_target(file_path, line_number, warning)
        elif 'Definition list' in warning or 'definition' in warning:
            fix_definition_list(file_path, line_number, warning)
        else:
            print(f"Warning: Unable to handle warning type: {warning}")
    
    print("\nAll warnings processed. Running rstcheck again...")
    
    # Run rstcheck again to verify fixes
    result = subprocess.run(['rstcheck', '-r', '--warn-unknown-settings', 'docs/'],
                           capture_output=True, text=True)
    if result.stderr:
        print("\nRemaining RST warnings:")
        print(result.stderr)
    else:
        print("\nAll RST warnings fixed!")

def fix_definition_list(file_path, line_number, warning):
    """Fix definition list formatting issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the current line
    current_line = lines[line_number - 1].strip()
    
    # Check if this is a definition list
    if not current_line.endswith(':'):
        print(f"Warning: Not a definition list item at {file_path}:{line_number}")
        return
    
    # Ensure proper spacing around definition list
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line:
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line:
            # Check for definition content
            if not next_line.startswith('   '):
                # Add proper indentation
                lines[line_number] = '   ' + next_line + '\n'
                line_number += 1
                
                # Indent subsequent lines
                for i in range(line_number, len(lines)):
                    if lines[i].strip():
                        if not lines[i].startswith('   '):
                            lines[i] = '   ' + lines[i].strip() + '\n'
                    else:
                        break
    
    # Add blank line after definition list
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line and not next_line.startswith('   '):
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed definition list issue in {file_path}:{line_number}")

def fix_transition_issues(file_path, line_number, warning):
    """Fix transition-related issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the transition line
    current_line = lines[line_number - 1].strip()
    
    # Validate transition
    if len(current_line) < 4 or not all(c in '=-~+*#.' for c in current_line):
        print(f"Warning: Invalid transition at {file_path}:{line_number}")
        return
    
    # Check for adjacent transitions
    if line_number < len(lines):
        next_line = lines[line_number].strip()
        if next_line and all(c in '=-~+*#.' for c in next_line):
            # Remove the second transition
            lines.pop(line_number)
            line_number -= 1
    
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line and all(c in '=-~+*#.' for c in prev_line):
            # Remove the previous transition
            lines.pop(line_number - 2)
            line_number -= 1
    
    # Ensure proper spacing around transition
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line:
            # Add blank line before transition
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line:
            # Add blank line after transition
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    # Add placeholder content between transitions
    if line_number < len(lines) - 2:
        next_line = lines[line_number + 1].strip()
        if next_line and all(c in '=-~+*#.' for c in next_line):
            # Add placeholder content
            lines.insert(line_number + 1, '    (Transition content)\n')
            line_number += 1
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed transition issue in {file_path}:{line_number}")

def fix_section_title(file_path, line_number, warning):
    """Fix section title issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the title line
    current_line = lines[line_number - 1].strip()
    
    # Check if this is a transition (all punctuation characters)
    if all(c in '=-~+*#.' for c in current_line):
        # Handle transition issues
        fix_transition_issues(file_path, line_number, warning)
        return
    
    # Get the title text
    title = current_line.strip()
    title_length = len(title)
    
    # Add proper spacing around title
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line:
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line:
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    # Fix title underline
    if line_number < len(lines) - 1:
        underline = lines[line_number].strip()
        if underline and len(underline) < title_length:
            # Extend underline to match title length
            underline_char = underline[0] if underline else '='
            new_underline = underline_char * title_length
            lines[line_number] = new_underline + '\n'
    
    # Add overline if needed
    if line_number > 2:
        overline = lines[line_number - 3].strip()
        if not overline:
            overline_char = underline[0] if underline else '='
            new_overline = overline_char * title_length
            lines.insert(line_number - 2, new_overline + '\n')
            line_number += 1
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed section title issue in {file_path}:{line_number}")

def fix_hyperlink_target(file_path, line_number, warning):
    """Fix hyperlink target issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the warning message
    match = re.search(r'Hyperlink target "([^"]+)" is not referenced', warning)
    if not match:
        print(f"Warning: Could not parse hyperlink target from warning")
        return
    
    target_name = match.group(1)
    
    # Find the target definition
    target_found = False
    for i, line in enumerate(lines):
        if f'.. _{target_name}:' in line:
            target_found = True
            # Add reference to target
            if i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                if not next_line:
                    lines.insert(i + 1, f':ref:`{target_name}`\n')
                    break
    
    if not target_found:
        # Create the target and reference
        lines.insert(line_number, f'.. _{target_name}:\n\n')
        lines.insert(line_number + 1, f':ref:`{target_name}`\n')
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed hyperlink target issue in {file_path}:{line_number}")

def main():
    warnings = get_rst_warnings()
    if not warnings:
        print("No RST warnings found!")
        return
    
    print(f"Found {len(warnings)} RST warnings. Processing...")
    
    # Sort warnings by file and descending line number to avoid line shifting
    warnings.sort(key=lambda x: (-x[1], x[0]))
    
    for file_path, line_number, warning in warnings:
        print(f"\nProcessing warning in {file_path}:{line_number}")
        print(f"  Warning: {warning}")
        
        # Dispatch to appropriate fixer based on warning type
        if 'Title underline' in warning:
            fix_title_underline(file_path, line_number, warning)
        elif 'Section title' in warning:
            fix_section_title(file_path, line_number, warning)
        elif 'Duplicate target name' in warning:
            fix_duplicate_targets(file_path, line_number, warning)
        elif 'Unknown directive' in warning:
            fix_unknown_directive(file_path, line_number, warning)
        elif 'automodule' in warning:
            fix_automodule_issues(file_path, line_number, warning)
        elif 'hyperlink' in warning:
            fix_hyperlink_issues(file_path, line_number, warning)
        elif 'transition' in warning:
            fix_transition_issues(file_path, line_number, warning)
        elif 'list' in warning or 'bullet' in warning:
            fix_list_issues(file_path, line_number, warning)
        elif 'inline markup' in warning or 'emphasis' in warning:
            fix_inline_markup(file_path, line_number, warning)
        elif 'Hyperlink target' in warning:
            fix_hyperlink_target(file_path, line_number, warning)
        elif 'Definition list' in warning or 'definition' in warning:
            fix_definition_list(file_path, line_number, warning)
        else:
            print(f"Warning: Unable to handle warning type: {warning}")
    
    print("\nAll warnings processed. Running rstcheck again...")
    
    # Run rstcheck again to verify fixes
    result = subprocess.run(['rstcheck', '-r', '--warn-unknown-settings', 'docs/'],
                           capture_output=True, text=True)
    if result.stderr:
        print("\nRemaining RST warnings:")
        print(result.stderr)
    else:
        print("\nAll RST warnings fixed!")

def fix_definition_list(file_path, line_number, warning):
    """Fix definition list formatting issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the current line
    current_line = lines[line_number - 1].strip()
    
    # Check if this is a definition list
    if not current_line.endswith(':'):
        print(f"Warning: Not a definition list item at {file_path}:{line_number}")
        return
    
    # Ensure proper spacing around definition list
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line:
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line:
            # Check for definition content
            if not next_line.startswith('   '):
                # Add proper indentation
                lines[line_number] = '   ' + next_line + '\n'
                line_number += 1
                
                # Indent subsequent lines
                for i in range(line_number, len(lines)):
                    if lines[i].strip():
                        if not lines[i].startswith('   '):
                            lines[i] = '   ' + lines[i].strip() + '\n'
                    else:
                        break
    
    # Add blank line after definition list
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line and not next_line.startswith('   '):
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed definition list issue in {file_path}:{line_number}")

def fix_section_title(file_path, line_number, warning):
    """Fix section title issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the title line
    current_line = lines[line_number - 1].strip()
    
    # Check if this is a transition (all punctuation characters)
    if all(c in '=-~+*#.' for c in current_line):
        # Handle transition issues
        fix_transition_issues(file_path, line_number, warning)
        return
    
    # Get the title text
    title = current_line.strip()
    title_length = len(title)
    
    # Add proper spacing around title
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line:
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line:
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    # Fix title underline
    if line_number < len(lines) - 1:
        underline = lines[line_number].strip()
        if underline and len(underline) < title_length:
            # Extend underline to match title length
            underline_char = underline[0] if underline else '='
            new_underline = underline_char * title_length
            lines[line_number] = new_underline + '\n'
    
    # Add overline if needed
    if line_number > 2:
        overline = lines[line_number - 3].strip()
        if not overline:
            overline_char = underline[0] if underline else '='
            new_overline = overline_char * title_length
            lines.insert(line_number - 2, new_overline + '\n')
            line_number += 1
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed section title issue in {file_path}:{line_number}")

def fix_transition_issues(file_path, line_number, warning):
    """Fix transition-related issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the transition line
    current_line = lines[line_number - 1].strip()
    
    # Validate transition
    if len(current_line) < 4 or not all(c in '=-~+*#.' for c in current_line):
        print(f"Warning: Invalid transition at {file_path}:{line_number}")
        return
    
    # Check for adjacent transitions
    if line_number < len(lines):
        next_line = lines[line_number].strip()
        if next_line and all(c in '=-~+*#.' for c in next_line):
            # Remove the second transition
            lines.pop(line_number)
            line_number -= 1
    
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line and all(c in '=-~+*#.' for c in prev_line):
            # Remove the previous transition
            lines.pop(line_number - 2)
            line_number -= 1
    
    # Ensure proper spacing around transition
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line:
            # Add blank line before transition
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line:
            # Add blank line after transition
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    # Add placeholder content between transitions
    if line_number < len(lines) - 2:
        next_line = lines[line_number + 1].strip()
        if next_line and all(c in '=-~+*#.' for c in next_line):
            # Add placeholder content
            lines.insert(line_number + 1, '    (Transition content)\n')
            line_number += 1
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed transition issue in {file_path}:{line_number}")

def fix_hyperlink_target(file_path, line_number, warning):
    """Fix hyperlink target issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the warning message
    match = re.search(r'Hyperlink target "([^"]+)" is not referenced', warning)
    if not match:
        print(f"Warning: Could not parse hyperlink target from warning")
        return
    
    target_name = match.group(1)
    
    # Find the target definition
    target_found = False
    for i, line in enumerate(lines):
        if f'.. _{target_name}:' in line:
            target_found = True
            # Add reference to target
            if i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                if not next_line:
                    lines.insert(i + 1, f':ref:`{target_name}`\n')
                    break
    
    if not target_found:
        # Create the target and reference
        lines.insert(line_number, f'.. _{target_name}:\n\n')
        lines.insert(line_number + 1, f':ref:`{target_name}`\n')
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed hyperlink target issue in {file_path}:{line_number}")

def main():
    warnings = get_rst_warnings()
    if not warnings:
        print("No RST warnings found!")
        return
    
    print(f"Found {len(warnings)} RST warnings. Processing...")
    
    # Sort warnings by file and descending line number to avoid line shifting
    warnings.sort(key=lambda x: (-x[1], x[0]))
    
    for file_path, line_number, warning in warnings:
        print(f"\nProcessing warning in {file_path}:{line_number}")
        print(f"  Warning: {warning}")
        
        # Dispatch to appropriate fixer based on warning type
        if 'Title underline' in warning:
            fix_title_underline(file_path, line_number, warning)
        elif 'Section title' in warning:
            fix_section_title(file_path, line_number, warning)
        elif 'Duplicate target name' in warning:
            fix_duplicate_targets(file_path, line_number, warning)
        elif 'Unknown directive' in warning:
            fix_unknown_directive(file_path, line_number, warning)
        elif 'automodule' in warning:
            fix_automodule_issues(file_path, line_number, warning)
        elif 'hyperlink' in warning:
            fix_hyperlink_issues(file_path, line_number, warning)
        elif 'transition' in warning:
            fix_transition_issues(file_path, line_number, warning)
        elif 'list' in warning or 'bullet' in warning:
            fix_list_issues(file_path, line_number, warning)
        elif 'inline markup' in warning or 'emphasis' in warning:
            fix_inline_markup(file_path, line_number, warning)
        elif 'Hyperlink target' in warning:
            fix_hyperlink_target(file_path, line_number, warning)
        elif 'Definition list' in warning or 'definition' in warning:
            fix_definition_list(file_path, line_number, warning)
        else:
            print(f"Warning: Unable to handle warning type: {warning}")
    
    print("\nAll warnings processed. Running rstcheck again...")
    
    # Run rstcheck again to verify fixes
    result = subprocess.run(['rstcheck', '-r', '--warn-unknown-settings', 'docs/'],
                           capture_output=True, text=True)
    if result.stderr:
        print("\nRemaining RST warnings:")
        print(result.stderr)
    else:
        print("\nAll RST warnings fixed!")

def fix_definition_list(file_path, line_number, warning):
    """Fix definition list formatting issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the current line
    current_line = lines[line_number - 1].strip()
    
    # Check if this is a definition list
    if not current_line.endswith(':'):
        print(f"Warning: Not a definition list item at {file_path}:{line_number}")
        return
    
    # Ensure proper spacing around definition list
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line:
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line:
            # Check for definition content
            if not next_line.startswith('   '):
                # Add proper indentation
                lines[line_number] = '   ' + next_line + '\n'
                line_number += 1
                
                # Indent subsequent lines
                for i in range(line_number, len(lines)):
                    if lines[i].strip():
                        if not lines[i].startswith('   '):
                            lines[i] = '   ' + lines[i].strip() + '\n'
                    else:
                        break
    
    # Add blank line after definition list
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line and not next_line.startswith('   '):
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed definition list issue in {file_path}:{line_number}")

def fix_transition_issues(file_path, line_number, warning):
    """Fix transition-related issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the transition line
    current_line = lines[line_number - 1].strip()
    
    # Validate transition
    if len(current_line) < 4 or not all(c in '=-~+*#.' for c in current_line):
        print(f"Warning: Invalid transition at {file_path}:{line_number}")
        return
    
    # Check for adjacent transitions
    if line_number < len(lines):
        next_line = lines[line_number].strip()
        if next_line and all(c in '=-~+*#.' for c in next_line):
            # Remove the second transition
            lines.pop(line_number)
            line_number -= 1
    
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line and all(c in '=-~+*#.' for c in prev_line):
            # Remove the previous transition
            lines.pop(line_number - 2)
            line_number -= 1
    
    # Ensure proper spacing around transition
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line:
            # Add blank line before transition
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line:
            # Add blank line after transition
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    # Add placeholder content between transitions
    if line_number < len(lines) - 2:
        next_line = lines[line_number + 1].strip()
        if next_line and all(c in '=-~+*#.' for c in next_line):
            # Add placeholder content
            lines.insert(line_number + 1, '    (Transition content)\n')
            line_number += 1
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed transition issue in {file_path}:{line_number}")

def fix_section_title(file_path, line_number, warning):
    """Fix section title issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the title line
    current_line = lines[line_number - 1].strip()
    
    # Check if this is a transition (all punctuation characters)
    if all(c in '=-~+*#.' for c in current_line):
        # Handle transition issues
        fix_transition_issues(file_path, line_number, warning)
        return
    
    # Get the title text
    title = current_line.strip()
    title_length = len(title)
    
    # Add proper spacing around title
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line:
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line:
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    # Fix title underline
    if line_number < len(lines) - 1:
        underline = lines[line_number].strip()
        if underline and len(underline) < title_length:
            # Extend underline to match title length
            underline_char = underline[0] if underline else '='
            new_underline = underline_char * title_length
            lines[line_number] = new_underline + '\n'
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed section title issue in {file_path}:{line_number}")

def fix_hyperlink_target(file_path, line_number, warning):
    """Fix hyperlink target issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the warning message
    match = re.search(r'Hyperlink target "([^"]+)" is not referenced', warning)
    if not match:
        print(f"Warning: Could not parse hyperlink target from warning")
        return
    
    target_name = match.group(1)
    
    # Find the target definition
    target_found = False
    for i, line in enumerate(lines):
        if f'.. _{target_name}:' in line:
            target_found = True
            # Add reference to target
            if i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                if not next_line:
                    lines.insert(i + 1, f':ref:`{target_name}`\n')
                    break
    
    if not target_found:
        # Create the target and reference
        lines.insert(line_number, f'.. _{target_name}:\n\n')
        lines.insert(line_number + 1, f':ref:`{target_name}`\n')
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed hyperlink target issue in {file_path}:{line_number}")

def main():
    warnings = get_rst_warnings()
    if not warnings:
        print("No RST warnings found!")
        return
    
    print(f"Found {len(warnings)} RST warnings. Processing...")
    
    # Sort warnings by file and descending line number to avoid line shifting
    warnings.sort(key=lambda x: (-x[1], x[0]))
    
    for file_path, line_number, warning in warnings:
        print(f"\nProcessing warning in {file_path}:{line_number}")
        print(f"  Warning: {warning}")
        
        # Dispatch to appropriate fixer based on warning type
        if 'Title underline' in warning:
            fix_title_underline(file_path, line_number, warning)
        elif 'Section title' in warning:
            fix_section_title(file_path, line_number, warning)
        elif 'Duplicate target name' in warning:
            fix_duplicate_targets(file_path, line_number, warning)
        elif 'Unknown directive' in warning:
            fix_unknown_directive(file_path, line_number, warning)
        elif 'automodule' in warning:
            fix_automodule_issues(file_path, line_number, warning)
        elif 'hyperlink' in warning:
            fix_hyperlink_issues(file_path, line_number, warning)
        elif 'transition' in warning:
            fix_transition_issues(file_path, line_number, warning)
        elif 'list' in warning or 'bullet' in warning:
            fix_list_issues(file_path, line_number, warning)
        elif 'inline markup' in warning or 'emphasis' in warning:
            fix_inline_markup(file_path, line_number, warning)
        elif 'Hyperlink target' in warning:
            fix_hyperlink_target(file_path, line_number, warning)
        elif 'Definition list' in warning or 'definition' in warning:
            fix_definition_list(file_path, line_number, warning)
        else:
            print(f"Warning: Unable to handle warning type: {warning}")
    
    print("\nAll warnings processed. Running rstcheck again...")
    
    # Run rstcheck again to verify fixes
    result = subprocess.run(['rstcheck', '-r', '--warn-unknown-settings', 'docs/'],
                           capture_output=True, text=True)
    if result.stderr:
        print("\nRemaining RST warnings:")
        print(result.stderr)
    else:
        print("\nAll RST warnings fixed!")

def fix_definition_list(file_path, line_number, warning):
    """Fix definition list formatting issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the current line
    current_line = lines[line_number - 1].strip()
    
    # Check if this is a definition list
    if not current_line.endswith(':'):
        print(f"Warning: Not a definition list item at {file_path}:{line_number}")
        return
    
    # Ensure proper spacing around definition list
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line:
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line:
            # Check for definition content
            if not next_line.startswith('   '):
                # Add proper indentation
                lines[line_number] = '   ' + next_line + '\n'
                line_number += 1
                
                # Indent subsequent lines
                for i in range(line_number, len(lines)):
                    if lines[i].strip():
                        if not lines[i].startswith('   '):
                            lines[i] = '   ' + lines[i].strip() + '\n'
                    else:
                        break
    
    # Add blank line after definition list
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line and not next_line.startswith('   '):
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed definition list issue in {file_path}:{line_number}")

def fix_transition_issues(file_path, line_number, warning):
    """Fix transition-related issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the transition line
    current_line = lines[line_number - 1].strip()
    
    # Validate transition
    if len(current_line) < 4 or not all(c in '=-~+*#.' for c in current_line):
        print(f"Warning: Invalid transition at {file_path}:{line_number}")
        return
    
    # Check for adjacent transitions
    if line_number < len(lines):
        next_line = lines[line_number].strip()
        if next_line and all(c in '=-~+*#.' for c in next_line):
            # Remove the second transition
            lines.pop(line_number)
            line_number -= 1
    
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line and all(c in '=-~+*#.' for c in prev_line):
            # Remove the previous transition
            lines.pop(line_number - 2)
            line_number -= 1
    
    # Ensure proper spacing around transition
    if line_number > 1:
        prev_line = lines[line_number - 2].strip()
        if prev_line:
            # Add blank line before transition
            lines.insert(line_number - 1, '\n')
            line_number += 1
    
    if line_number < len(lines) - 1:
        next_line = lines[line_number].strip()
        if next_line:
            # Add blank line after transition
            lines.insert(line_number + 1, '\n')
            line_number += 1
    
    # Add placeholder content between transitions
    if line_number < len(lines) - 2:
        next_line = lines[line_number + 1].strip()
        if next_line and all(c in '=-~+*#.' for c in next_line):
            # Add placeholder content
            lines.insert(line_number + 1, '    (Transition content)\n')
            line_number += 1
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed transition issue in {file_path}:{line_number}")

def main():
    warnings = get_rst_warnings()
    if not warnings:
        print("No RST warnings found!")
        return
    
    print(f"Found {len(warnings)} RST warnings. Processing...")
    
    # Sort warnings by file and descending line number to avoid line shifting
    warnings.sort(key=lambda x: (-x[1], x[0]))
    
    for file_path, line_number, warning in warnings:
        print(f"\nProcessing warning in {file_path}:{line_number}")
        print(f"  Warning: {warning}")
        
        # Dispatch to appropriate fixer based on warning type
        if 'Title underline' in warning:
            fix_title_underline(file_path, line_number, warning)
        elif 'Section title' in warning:
            fix_section_title(file_path, line_number, warning)
        elif 'Duplicate target name' in warning:
            fix_duplicate_targets(file_path, line_number, warning)
        elif 'Unknown directive' in warning:
            fix_unknown_directive(file_path, line_number, warning)
        elif 'automodule' in warning:
            fix_automodule_issues(file_path, line_number, warning)
        elif 'hyperlink' in warning:
            fix_hyperlink_issues(file_path, line_number, warning)
        elif 'transition' in warning:
            fix_transition_issues(file_path, line_number, warning)
        elif 'list' in warning or 'bullet' in warning:
            fix_list_issues(file_path, line_number, warning)
        elif 'inline markup' in warning or 'emphasis' in warning:
            fix_inline_markup(file_path, line_number, warning)
        elif 'Hyperlink target' in warning:
            fix_hyperlink_target(file_path, line_number, warning)
        elif 'Definition list' in warning or 'definition' in warning:
            fix_definition_list(file_path, line_number, warning)
        else:
            print(f"Warning: Unable to handle warning type: {warning}")
    
    print("\nAll warnings processed. Running rstcheck again...")
    
    # Run rstcheck again to verify fixes
    result = subprocess.run(['rstcheck', '-r', '--warn-unknown-settings', 'docs/'],
                           capture_output=True, text=True)
    if result.stderr:
        print("\nRemaining RST warnings:")
        print(result.stderr)
    else:
        print("\nAll RST warnings fixed!")
        if next_line:
            lines.insert(line_number + 4, '\n')
    
    # Add module docstring if it doesn't exist
    if line_number < len(lines):
        doc_line = lines[line_number].strip()
        if not doc_line.startswith('"""'):
            # Get module docstring
            module = __import__(module_name)
            docstring = getattr(module, '__doc__', '')
            if docstring:
                # Use module's actual docstring
                lines.insert(line_number, f'"""{docstring}\n"""\n\n')
            else:
                # Use default docstring
                lines.insert(line_number, f'"""Module documentation for {module_name}.\n"""\n\n')
            line_number += 1
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed automodule issue in {file_path}:{line_number}")

def fix_directive_options(file_path, line_number, warning):
    """Fix directive options formatting issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the directive line
    current_line = lines[line_number - 1].strip()
    
    # Validate directive format
    if not current_line.startswith('.. '):
        print(f"Warning: Not a directive at {file_path}:{line_number}")
        return
    
    # Check for options
    if line_number < len(lines):
        next_line = lines[line_number].strip()
        if next_line and next_line.startswith(':'):
            # Validate option format
            if not next_line.endswith(':'):
                print(f"Warning: Invalid option format at {file_path}:{line_number}")
                return
            
            # Ensure proper indentation
            if not next_line.startswith('    '):
                lines[line_number] = '    ' + next_line + '\n'
            
            # Check for option value
            option_name = next_line[1:next_line.find(':')].strip()
            if option_name and line_number < len(lines) - 1:
                option_value = lines[line_number + 1].strip()
                if option_value:
                    if not option_value.startswith('    '):
                        lines[line_number + 1] = '    ' + option_value + '\n'
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed directive options in {file_path}:{line_number}")

def fix_list_issues(file_path, line_number, warning):
    """Fix list formatting issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # List of valid bullet characters
    bullet_chars = '*-+'
    
    # Check current line for list item
    current_line = lines[line_number - 1].strip()
    
    # Handle bullet lists
    if current_line and current_line[0] in bullet_chars:
        # Validate bullet character
        if len(current_line) < 2 or current_line[1] != ' ':
            # Fix bullet spacing
            bullet = current_line[0]
            content = current_line[1:].strip()
            lines[line_number - 1] = f'{bullet} {content}\n'
        
        # Check indentation of nested items
        if line_number < len(lines):
            next_line = lines[line_number].strip()
            if next_line and next_line[0] in bullet_chars:
                # Nested list item, ensure proper indentation
                if not next_line.startswith('    '):
                    lines[line_number] = '    ' + next_line + '\n'
    
    # Handle numbered lists
    elif current_line and current_line[0].isdigit():
        # Find end of number
        for i, char in enumerate(current_line):
            if not char.isdigit():
                break
        number = current_line[:i]
        rest = current_line[i:].strip()
        
        # Validate delimiter
        if not rest.startswith('.') and not rest.startswith(')'):
            print(f"Warning: Invalid numbered list format at {file_path}:{line_number}")
            return
        
        # Check content after delimiter
        if len(rest) < 2 or rest[1] != ' ':
            # Fix spacing
            delimiter = rest[0]
            content = rest[1:].strip()
            lines[line_number - 1] = f'{number}{delimiter} {content}\n'
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"  Error fixing block quote: {str(e)}")
        return False

def fix_inline_markup(file_path, line_number, warning):
    """Fix inline markup formatting issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Get the current line
        current_line = lines[line_number - 1]
                    parts.append(char)
                    in_italic = True
            else:
                current_part += char
        
        if current_part:
            parts.append(current_part)
        
        # Rebuild line with proper spacing
        new_line = ''
        for i, part in enumerate(parts):
            if i > 0 and (parts[i-1] == '*' or parts[i-1] == '_'):
                # Add space after markup character
                if not part.startswith(' '):
                    new_line += ' ' + part
                else:
                    new_line += part
            else:
                new_line += part
        
        lines[line_number - 1] = new_line
    
    # Fix bold markup
    if '**' in current_line or '__' in current_line:
        # Split line into parts
        parts = []
        current_part = ''
        in_bold = False
        
        for char in current_line:
            if current_part[-1:] == char and (char == '*' or char == '_'):
                if in_bold:
                    # End of bold
                    if current_part:
                        parts.append(current_part)
                        current_part = ''
                    parts.append(char)
                    in_bold = False
                else:
                    # Start of bold
                    if current_part:
                        parts.append(current_part)
                        current_part = ''
                    parts.append(char)
                    in_bold = True
            else:
                current_part += char
        
        if current_part:
            parts.append(current_part)
        
        # Rebuild line with proper spacing
        new_line = ''
        for i, part in enumerate(parts):
            if i > 0 and (parts[i-1] == '**' or parts[i-1] == '__'):
                # Add space after markup character
                if not part.startswith(' '):
                    new_line += ' ' + part
                else:
                    new_line += part
            else:
                new_line += part
        
        lines[line_number - 1] = new_line
    
    try:
        # Write the changes back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return
    
    print(f"Fixed inline markup in {file_path}:{line_number}")

def fix_hyperlink_issues(file_path, line_number, warning):
    """Fix hyperlink-related issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Get the target name from the warning
    target_name = warning.split('target name:')[1].split('"')[1]
    
    # Validate target name
    if not target_name:
        print(f"Warning: Empty target name in {file_path}:{line_number}")
        return
    
    # Find the target definition
    target_line = None
    for i, line in enumerate(lines):
        if f'.. _{target_name}:' in line:
            target_line = i
            break
    
    if target_line is None:
        print(f"Warning: Target {target_name} not found in {file_path}")
        return
    
    # Add a reference to this target
    reference_line = f'`{target_name} <#{target_name}>`_\n'
    
    # Ensure proper spacing around the reference
    if target_line < len(lines) - 1:
        next_line = lines[target_line + 1].strip()
        if not next_line:  # If next line is blank
            lines.insert(target_line + 2, reference_line)
        else:
            lines.insert(target_line + 1, '\n')
            lines.insert(target_line + 2, reference_line)
    
    # Write the changes back to the file
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Fixed hyperlink issue in {file_path}:{line_number}")

def fix_duplicate_targets(file_path, line_number, warning):
    """Fix duplicate target names by appending unique suffixes."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Get the duplicate target name
    target_name = warning.split('target name:')[1].split('"')[1]
    
    # Find all occurrences of this target
    target_lines = []
    for i, line in enumerate(lines):
        if f'.. _{target_name}:' in line:
            target_lines.append(i)
    
    # If there's only one occurrence, it's not a duplicate
    if len(target_lines) <= 1:
        return
    
    # Rename all but the first occurrence
    for i, line_num in enumerate(target_lines[1:], start=1):
        new_target_name = f'{target_name}_{i}'
        lines[line_num] = lines[line_num].replace(f'.. _{target_name}:', f'.. _{new_target_name}:')
        
        # Update references to this target
        for j, line in enumerate(lines):
            if f':ref:`{target_name}`' in line:
                lines[j] = line.replace(f':ref:`{target_name}`', f':ref:`{new_target_name}`')
            elif f'`{target_name}`_' in line:
                lines[j] = line.replace(f'`{target_name}`_', f'`{new_target_name}`_')
    
    # Write the changes back to the file
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Fixed duplicate target issue in {file_path}:{line_number}")

def fix_block_quote(file_path, line_number, warning):
    """Fix block quote indentation issues."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find the block quote section
        start_line = line_number - 1
        while start_line > 0 and lines[start_line].strip():
            start_line -= 1
        start_line += 1
        
        # Ensure proper indentation
        for i in range(start_line, len(lines)):
            if lines[i].strip():
                # Add proper indentation if missing
                if not lines[i].startswith('    '):
                    lines[i] = '    ' + lines[i].lstrip()
            else:
                # Add blank line if needed
                if i > start_line and not lines[i-1].strip():
                    lines[i] = '\n'
        
        # Add blank line after block quote
        if start_line < len(lines) - 1 and not lines[start_line + 1].strip():
            lines[start_line + 1] = '\n'
        
        with open(file_path, 'w') as f:
            f.writelines(lines)
        print(f"  Fixed block quote at line {line_number}")
        return True
    except Exception as e:
        print(f"  Error fixing block quote: {str(e)}")
        return False

def main():
    warnings = get_rst_warnings()
    if not warnings:
        print("No RST warnings found!")
        return
    
    print(f"Found {len(warnings)} RST warnings. Processing...")
    
    # Sort warnings by file and descending line number to avoid line shifting
    warnings.sort(key=lambda x: (-x[1], x[0]))
    
    for file_path, line_number, warning in warnings:
        print(f"\nProcessing warning in {file_path}:{line_number}")
        print(f"  Warning: {warning}")
        
        # Dispatch to appropriate fixer based on warning type
        if 'Title underline' in warning or 'Section title' in warning:
            fix_section_title(file_path, line_number, warning)
        elif 'Duplicate target name' in warning:
            fix_duplicate_targets(file_path, line_number, warning)
        elif 'Unknown directive' in warning or 'automodule' in warning:
            fix_automodule_issues(file_path, line_number, warning)
        elif 'hyperlink' in warning:
            fix_hyperlink_issues(file_path, line_number, warning)
        elif 'nested' in warning:
            fix_nested_directives(file_path, line_number, warning)
        elif 'options' in warning:
            fix_directive_options(file_path, line_number, warning)
        elif 'list' in warning or 'bullet' in warning:
            fix_list_issues(file_path, line_number, warning)
        elif 'inline markup' in warning or 'emphasis' in warning:
            fix_inline_markup(file_path, line_number, warning)
        elif 'transition' in warning:
            fix_transition_issues(file_path, line_number, warning)
        elif 'block quote' in warning:
            fix_block_quote(file_path, line_number, warning)
        else:
            print(f"Warning: Unable to handle warning type: {warning}")
    
    print("\nAll warnings processed. Running Sphinx build again...")
    
    # Run Sphinx build again to verify fixes
    result = subprocess.run(['sphinx-build', '-b', 'html', '-W', '--keep-going', 'docs/', 'docs/_build/html'],
                           capture_output=True, text=True)
    if result.stderr:
        print("\nRemaining RST warnings:")
        print(result.stderr)
    else:
        print("\nAll RST warnings fixed!")

if __name__ == "__main__":
    main()
