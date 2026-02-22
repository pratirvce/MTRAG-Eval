#!/usr/bin/env python3
"""
Rename ieee_acl_submission to submission and add timestamp to LaTeX file.
"""

import shutil
import pathlib
from datetime import datetime

# Current folder
OLD_FOLDER = pathlib.Path("ieee_acl_submission")
NEW_FOLDER = pathlib.Path("submission")

# Rename folder
if OLD_FOLDER.exists():
    if NEW_FOLDER.exists():
        print(f"⚠️  {NEW_FOLDER} already exists. Removing it first...")
        shutil.rmtree(NEW_FOLDER)
    shutil.move(str(OLD_FOLDER), str(NEW_FOLDER))
    print(f"✅ Renamed folder: {OLD_FOLDER.name} -> {NEW_FOLDER.name}")
else:
    print(f"⚠️  Folder {OLD_FOLDER} does not exist. Creating {NEW_FOLDER}...")
    NEW_FOLDER.mkdir(exist_ok=True)

# Rename LaTeX file with timestamp
latex_file = NEW_FOLDER / "ieee_acl_paper.tex"
if latex_file.exists():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_latex_name = f"report_{timestamp}.tex"
    new_latex_path = NEW_FOLDER / new_latex_name
    shutil.move(str(latex_file), str(new_latex_path))
    print(f"✅ Renamed LaTeX file: ieee_acl_paper.tex -> {new_latex_name}")
    print(f"   Full path: {new_latex_path.absolute()}")
else:
    print(f"⚠️  LaTeX file {latex_file} not found")

# List contents
print(f"\n📁 Contents of {NEW_FOLDER.name}/:")
for item in sorted(NEW_FOLDER.iterdir()):
    if item.is_file():
        print(f"   📄 {item.name}")
    elif item.is_dir():
        file_count = len(list(item.iterdir()))
        print(f"   📁 {item.name}/ ({file_count} files)")

print(f"\n✅ Done! Folder is now: {NEW_FOLDER.absolute()}/")

