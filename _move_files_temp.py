#!/usr/bin/env python3
import os
import sys
import shutil
from pathlib import Path

os.chdir('/home/prevanka/prati/su-mt-rag/mt-rag-benchmark')
script_dir = Path('.')
md_files_dir = script_dir / "md_files"
exclude_file = "COMPREHENSIVE_EXPERIMENT_REPORT.md"

md_files_dir.mkdir(exist_ok=True)

md_files = list(script_dir.glob("*.md"))
files_to_move = [f for f in md_files if f.name != exclude_file]

print(f"Found {len(md_files)} .md files total")
print(f"Moving {len(files_to_move)} files to md_files/")
print(f"Excluding: {exclude_file}\n")

moved_count = 0
errors = []
for md_file in sorted(files_to_move):
    try:
        dest = md_files_dir / md_file.name
        if dest.exists():
            print(f"⚠️  {md_file.name} already exists, skipping...")
            continue
        shutil.move(str(md_file), str(dest))
        moved_count += 1
        if moved_count % 20 == 0:
            print(f"✅ Moved {moved_count} files so far...")
    except Exception as e:
        error_msg = f"❌ Error moving {md_file.name}: {e}"
        errors.append(error_msg)
        print(error_msg)

print(f"\n✅ Successfully moved {moved_count} files to md_files/")
if errors:
    print(f"\n⚠️  {len(errors)} errors occurred:")
    for err in errors[:10]:
        print(f"  {err}")
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more errors")

