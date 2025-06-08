import os
import glob
import subprocess
from pathlib import Path

def convert_rst_to_md(rst_file):
    """Convert a single RST file to Markdown using pandoc."""
    md_file = rst_file.replace('.rst', '.md')
    try:
        subprocess.run(['pandoc', rst_file, '-f', 'rst', '-t', 'markdown', '-o', md_file], check=True)
        print(f"Converted {rst_file} to {md_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {rst_file}: {e}")
        return False

def main():
    docs_dir = Path("docs")
    rst_files = list(docs_dir.glob("**/*.rst"))
    
    # Create a backup directory
    backup_dir = docs_dir / "backup_rst"
    backup_dir.mkdir(exist_ok=True)
    
    for rst_file in rst_files:
        # Skip specific files we don't want to convert
        if rst_file.name in ['README.rst', 'test.rst']:
            continue
            
        # Move original RST to backup
        backup_path = backup_dir / rst_file.relative_to(docs_dir)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        rst_file.rename(backup_path)
        
        # Convert from backup
        if convert_rst_to_md(str(backup_path)):
            print(f"Successfully converted {backup_path.name}")
        else:
            # If conversion fails, restore the original RST
            backup_path.rename(rst_file)
            print(f"Failed to convert {backup_path.name}, restored original")

if __name__ == "__main__":
    main()
