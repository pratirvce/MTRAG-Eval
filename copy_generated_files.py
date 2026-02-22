#!/usr/bin/env python3
"""
Copy generated LaTeX paper and figures to a separate folder with timestamped filename.
"""

import shutil
import pathlib
from datetime import datetime

# Source files and folders
SOURCE_FILES = [
    "ieee_acl_paper.tex",
]

SOURCE_DIRS = [
    "report_figures",
]

# Target folder
TARGET_FOLDER = pathlib.Path("submission")
TARGET_FOLDER.mkdir(exist_ok=True)

# Create subdirectories
FIGURES_TARGET = TARGET_FOLDER / "figures"
FIGURES_TARGET.mkdir(exist_ok=True)

def copy_files():
    """Copy generated files to the submission folder with timestamped LaTeX filename."""
    print(f"Copying files to {TARGET_FOLDER}/")
    print("-" * 60)
    
    # Copy LaTeX file and update figure paths with timestamp
    copied_count = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for source_file in SOURCE_FILES:
        source_path = pathlib.Path(source_file)
        if source_path.exists():
            # Create timestamped filename
            new_name = f"report_{timestamp}.tex"
            target_path = TARGET_FOLDER / new_name
            
            # Read and update paths in LaTeX file
            with open(source_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Update figure paths to point to figures/ subdirectory
            content = content.replace('report_figures/', 'figures/')
            # Write to target
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Copied and updated paths: {source_file} -> {TARGET_FOLDER.name}/{new_name}")
            copied_count += 1
        else:
            print(f"⚠️  Not found: {source_file}")
    
    # Copy JPEG figures from report_figures
    for source_dir in SOURCE_DIRS:
        source_path = pathlib.Path(source_dir)
        if source_path.exists() and source_path.is_dir():
            jpeg_files = list(source_path.glob("*.jpeg"))
            jpeg_files.extend(list(source_path.glob("*.jpg")))
            
            if jpeg_files:
                for jpeg_file in jpeg_files:
                    target_path = FIGURES_TARGET / jpeg_file.name
                    shutil.copy2(jpeg_file, target_path)
                    print(f"✅ Copied: {jpeg_file.name} -> {TARGET_FOLDER.name}/figures/{jpeg_file.name}")
                    copied_count += 1
            else:
                print(f"⚠️  No JPEG files found in {source_dir}/")
    
    print("-" * 60)
    print(f"✅ Total files copied: {copied_count}")
    print(f"\nFiles are now in: {TARGET_FOLDER.absolute()}/")
    print(f"  - LaTeX paper: {TARGET_FOLDER.name}/report_{timestamp}.tex")
    print(f"  - Figures: {TARGET_FOLDER.name}/figures/")

if __name__ == "__main__":
    copy_files()
