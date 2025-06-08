#!/bin/bash
# Fix file permissions for the entire repository

# Make files non-executable if they don't have a shebang
find . -type f -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "*.rst" -o -name "*.yaml" -o -name "*.yml" -o -name "*.toml" -o -name "*.json" -o -name "*.css" -o -name "*.ini" -o -name "*.cfg" | while read -r file; do
    # Skip files that should be executable
    if [[ "$file" == *"/scripts/"* ]] && [[ "$file" != *".sh" ]]; then
        continue
    fi
    
    # Check if file has a shebang
    if ! head -n 1 "$file" | grep -q '^#!'; then
        # Check if file is executable
        if [ -x "$file" ]; then
            echo "Fixing permissions for: $file"
            chmod -x "$file"
        fi
    fi
done

echo "All file permissions have been fixed."
