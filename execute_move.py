import os
import shutil
from pathlib import Path

script_dir = Path(__file__).parent
md_files_dir = script_dir / "md_files"
exclude_file = "COMPREHENSIVE_EXPERIMENT_REPORT.md"

md_files_dir.mkdir(exist_ok=True)

md_files = list(script_dir.glob("*.md"))
files_to_move = [f for f in md_files if f.name != exclude_file]

print(f"Found {len(md_files)} .md files total")
print(f"Moving {len(files_to_move)} files to md_files/")
print(f"Excluding: {exclude_file}\n")

moved_count = 0
for md_file in files_to_move:
    try:
        dest = md_files_dir / md_file.name
        if dest.exists():
            print(f"⚠️  {md_file.name} already exists, skipping...")
            continue
        shutil.move(str(md_file), str(dest))
        moved_count += 1
        if moved_count % 10 == 0:
            print(f"Moved {moved_count} files...")
    except Exception as e:
        print(f"❌ Error moving {md_file.name}: {e}")

print(f"\n✅ Successfully moved {moved_count} files to md_files/")

