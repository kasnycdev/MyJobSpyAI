import os
import shutil
from pathlib import Path

def organize_docs():
    docs_dir = Path("docs")
    backup_dir = docs_dir / "backup_rst"
    
    # Create necessary directories
    directories = [
        "getting_started",
        "features",
        "development",
        "enhancement_plans",
        "api/core",
        "api/analysis",
        "api/filtering",
        "api/llm",
        "api/scrapers",
        "api/models",
        "api/utils"
    ]
    
    for dir_name in directories:
        (docs_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Move files to their correct locations
    file_mapping = {
        "backup_rst/installation.md": "getting_started/installation.md",
        "backup_rst/configuration.md": "getting_started/configuration.md",
        "backup_rst/usage.md": "getting_started/usage.md",
        "backup_rst/examples.md": "getting_started/examples.md",
        "backup_rst/contributing.md": "development/contributing.md",
        "backup_rst/changelog.md": "changelog.md",
        "backup_rst/enhancement_plans/index.md": "enhancement_plans/index.md"
    }
    
    for src, dest in file_mapping.items():
        src_path = backup_dir / src
        dest_path = docs_dir / dest
        if src_path.exists():
            shutil.move(str(src_path), str(dest_path))
            print(f"Moved {src} to {dest}")
    
    # Move API files
    api_files = list(backup_dir.glob("api/**/*.md"))
    for api_file in api_files:
        relative_path = api_file.relative_to(backup_dir)
        dest_path = docs_dir / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(api_file), str(dest_path))
        print(f"Moved {relative_path} to {dest_path}")

if __name__ == "__main__":
    organize_docs()
