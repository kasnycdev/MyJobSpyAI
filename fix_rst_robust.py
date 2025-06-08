import os

def fix_rst_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\r\n') for line in f]
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
            
        # Check if this is a title line
        if i + 1 < len(lines):
            next_line = lines[i+1].rstrip()
            if (next_line and 
                len(set(next_line)) == 1 and  # All same character
                next_line[0] in '=+*^~`#"' and  # Valid title underline chars
                len(next_line) >= 4):  # Reasonable minimum length
                
                # Fix the title line if it has multiple spaces
                if '  ' in line:
                    words = line.split()
                    fixed_line = ' '.join(words)
                    lines[i] = fixed_line
                    line = fixed_line  # Update for length check
                
                # Ensure underline is at least as long as the title
                if len(next_line) < len(line):
                    lines[i+1] = next_line[0] * len(line)
                
                i += 2  # Skip the next line as we've processed it
                continue
        
        i += 1
    
    # Ensure consistent line endings
    content = '\n'.join(lines) + '\n'
    
    # Write the fixed content back to the file with consistent line endings
    with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)

# Fix the problematic files
fix_rst_file('docs/api/myjobspyai.rst')
fix_rst_file('docs/api/myjobspyai.analysis.rst')
