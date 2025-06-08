import os
import re
import shutil
from pathlib import Path
import subprocess

def convert_file(rst_path, md_path):
    """Convert a single RST file to Markdown."""
    try:
        # Run rst2md conversion
        result = subprocess.run(['pandoc', '-f', 'rst', '-t', 'markdown', str(rst_path)],
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error converting {rst_path}: {result.stderr}")
            return False
        
        # Write the converted content
        with open(md_path, 'w') as f:
            f.write(result.stdout)
        
        print(f"Converted {rst_path} to {md_path}")
        return True
    except Exception as e:
        print(f"Error converting {rst_path}: {str(e)}")
        return False

def convert_directory(rst_dir, md_dir):
    """Convert all RST files in a directory to Markdown."""
    # Create MD directory if it doesn't exist
    os.makedirs(md_dir, exist_ok=True)
    
    # Convert each RST file
    for rst_file in rst_dir.glob('*.rst'):
        md_file = md_dir / f"{rst_file.stem}.md"
        convert_file(rst_file, md_file)

def main():
    # Define paths
    docs_dir = Path('docs')
    md_docs_dir = Path('docs-md')
    
    # Create MD docs directory
    if md_docs_dir.exists():
        shutil.rmtree(md_docs_dir)
    md_docs_dir.mkdir()
    
    # Convert main documentation files
    convert_directory(docs_dir, md_docs_dir)
    
    # Convert API documentation
    api_dir = docs_dir / 'api'
    if api_dir.exists():
        api_md_dir = md_docs_dir / 'api'
        convert_directory(api_dir, api_md_dir)
    
    # Convert enhancement plans
    plans_dir = docs_dir / 'enhancement_plans'
    if plans_dir.exists():
        plans_md_dir = md_docs_dir / 'enhancement_plans'
        convert_directory(plans_dir, plans_md_dir)

if __name__ == "__main__":
    main()
