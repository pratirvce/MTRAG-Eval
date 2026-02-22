#!/usr/bin/env python3
"""Execute rename operations."""
import shutil
import pathlib
from datetime import datetime

OLD_FOLDER = pathlib.Path("ieee_acl_submission")
NEW_FOLDER = pathlib.Path("submission")

if OLD_FOLDER.exists():
    if NEW_FOLDER.exists():
        print(f"Removing existing {NEW_FOLDER}...")
        shutil.rmtree(NEW_FOLDER)
    shutil.move(str(OLD_FOLDER), str(NEW_FOLDER))
    print(f"✅ Renamed folder: {OLD_FOLDER.name} -> {NEW_FOLDER.name}")
    
    latex_file = NEW_FOLDER / "ieee_acl_paper.tex"
    if latex_file.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"report_{timestamp}.tex"
        new_path = NEW_FOLDER / new_name
        shutil.move(str(latex_file), str(new_path))
        print(f"✅ Renamed LaTeX file: ieee_acl_paper.tex -> {new_name}")
        print(f"   Full path: {new_path.absolute()}")
    else:
        print(f"⚠️  LaTeX file not found")
    
    print(f"\n📁 Contents of {NEW_FOLDER.name}/:")
    for item in sorted(NEW_FOLDER.iterdir()):
        if item.is_file():
            print(f"   📄 {item.name}")
        elif item.is_dir():
            count = len(list(item.iterdir()))
            print(f"   📁 {item.name}/ ({count} files)")
else:
    print(f"⚠️  {OLD_FOLDER} does not exist")

print(f"\n✅ Done! Folder: {NEW_FOLDER.absolute()}/")

