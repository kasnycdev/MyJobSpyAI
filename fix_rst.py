import os

def fix_rst_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix title underlines
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        next_line = lines[i+1].strip() if i+1 < len(lines) else ''
        
        # Check if this is a title underline
        if (i+1 < len(lines) and 
            len(next_line) >= 2 and 
            all(c == next_line[0] for c in next_line) and 
            next_line[0] in '=+*^~`#"'):
            
            # Ensure underline is at least as long as the text
            if len(next_line) < len(line):
                lines[i+1] = next_line[0] * len(line) + '\n'
            i += 2  # Skip the next line as we've processed it
        else:
            i += 1
    
    # Write the fixed content back to the file
    with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(lines)

# Fix the problematic files
fix_rst_file('docs/api/myjobspyai.rst')
fix_rst_file('docs/api/myjobspyai.analysis.rst')
