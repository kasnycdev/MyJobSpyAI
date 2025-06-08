#!/bin/bash
# Remove executable bit from files that shouldn't be executable

# List of file patterns that should not be executable
PATTERNS=(
    "*.md"
    "*.txt"
    "*.yaml"
    "*.yml"
    "*.rst"
    "*.py"  # Python files should not be executable unless they have a shebang
    "*.toml"
    "*.ini"
    "*.cfg"
    "*.json"
)

# Build find command
FIND_CMD="find . -type f"
for pattern in "${PATTERNS[@]}"; do
    FIND_CMD+=" -o -name \"$pattern\""
done
FIND_CMD+=' -exec test -x {} \; -print'

# Find and fix permissions
echo "Fixing permissions for the following files:"
eval "$FIND_CMD" | while read -r file; do
    # Check if file has a shebang
    if ! head -n 1 "$file" | grep -q '^#!' && [ "$(file -b --mime-type "$file" | cut -d/ -f1)" != "text" ]; then
        echo "  $file"
        chmod -x "$file"
    fi
done

echo "Permissions fixed."
