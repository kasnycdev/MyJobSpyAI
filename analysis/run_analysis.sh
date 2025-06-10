#!/bin/bash
# Code Analysis Script

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create reports directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="reports/$TIMESTAMP"
mkdir -p "$REPORT_DIR"

echo -e "${GREEN}Starting code analysis...${NC}"

# Run Pylint
echo -e "${YELLOW}Running Pylint...${NC}"
if command -v pylint &> /dev/null; then
    pylint --rcfile=../.pylintrc src/ > "$REPORT_DIR/pylint.txt" 2>&1
    PYLINT_EXIT=$?
    if [ $PYLINT_EXIT -eq 0 ]; then
        echo -e "${GREEN}Pylint analysis complete${NC}"
    else
        echo -e "${YELLOW}Pylint found some issues (exit code: $PYLINT_EXIT)${NC}"
    fi
else
    echo "Pylint not found. Install with: pip install pylint"
fi

# Run Flake8
echo -e "${YELLOW}Running Flake8...${NC}"
if command -v flake8 &> /dev/null; then
    flake8 --config=../.flake8 src/ > "$REPORT_DIR/flake8.txt"
    FLAKE8_EXIT=$?
    if [ $FLAKE8_EXIT -eq 0 ]; then
        echo -e "${GREEN}Flake8 analysis complete${NC}"
    else
        echo -e "${YELLOW}Flake8 found some issues (exit code: $FLAKE8_EXIT)${NC}"
    fi
else
    echo "Flake8 not found. Install with: pip install flake8"
fi

# Generate summary
echo -e "${GREEN}Generating summary...${NC}"
{
    echo "# Code Analysis Report - $(date)"
    echo "## Summary"
    echo "- Files Analyzed: $(find ../src -name "*.py" | wc -l)"
    echo "- Total LOC: $(find ../src -name "*.py" -type f -exec wc -l {} + | awk '{total += $1} END{print total}')"

    if [ -f "$REPORT_DIR/pylint.txt" ]; then
        echo -e "\n## Pylint Results"
        echo "Score: $(grep -oP "(?<=Your code has been rated at ).*?(?=/10)" "$REPORT_DIR/pylint.txt" || echo "N/A")/10"
        echo -e "\n### Top Issues"
        grep -E "^[A-Z]:" "$REPORT_DIR/pylint.txt" | sort | uniq -c | sort -nr | head -5
    fi

    if [ -f "$REPORT_DIR/flake8.txt" ]; then
        echo -e "\n## Flake8 Results"
        echo "Total Issues: $(wc -l < "$REPORT_DIR/flake8.txt")"
        echo -e "\n### Issue Types"
        cut -d: -f4 "$REPORT_DIR/flake8.txt" | sort | uniq -c | sort -nr
    fi
} > "$REPORT_DIR/summary.md"

echo -e "${GREEN}Analysis complete!${NC}"
echo "Report generated at: $REPORT_DIR/summary.md"
