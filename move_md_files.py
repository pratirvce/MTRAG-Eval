#!/usr/bin/env python3
"""Move all .md files except COMPREHENSIVE_EXPERIMENT_REPORT.md to md_files folder."""

import os
import shutil
from pathlib import Path

# Get the script directory (mt-rag-benchmark)
script_dir = Path(__file__).parent
md_files_dir = script_dir / "md_files"
exclude_file = "COMPREHENSIVE_EXPERIMENT_REPORT.md"

# Create md_files directory if it doesn't exist
md_files_dir.mkdir(exist_ok=True)

# Find all .md files in the root directory
md_files = list(script_dir.glob("*.md"))

# Filter out the excluded file
files_to_move = [f for f in md_files if f.name != exclude_file]

print(f"Found {len(md_files)} .md files total")
print(f"Moving {len(files_to_move)} files to md_files/")
print(f"Excluding: {exclude_file}")
print()

# Move each file
moved_count = 0
for md_file in files_to_move:
    try:
        dest = md_files_dir / md_file.name
        if dest.exists():
            print(f"⚠️  {md_file.name} already exists in md_files/, skipping...")
            continue
        shutil.move(str(md_file), str(dest))
        print(f"✅ Moved: {md_file.name}")
        moved_count += 1
    except Exception as e:
        print(f"❌ Error moving {md_file.name}: {e}")

print()
print(f"✅ Successfully moved {moved_count} files to md_files/")

