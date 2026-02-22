#!/usr/bin/env python3
"""
Script to update old experiment names in files across the codebase.
This replaces old names (phase1_, tier1_, best_paper_, task_a_) with new names.
"""

import json
import pathlib
import re
import sys

# Load mapping
with open('experiment_name_mapping.json') as f:
    mapping_data = json.load(f)
mapping = mapping_data.get('mapping', {})

def replace_in_file(filepath, replacements, dry_run=False):
    """Replace old names in a file."""
    filepath = pathlib.Path(filepath)
    if not filepath.exists():
        print(f"  ⚠️  File not found: {filepath}")
        return False
    
    content = filepath.read_text()
    original_content = content
    changes = []
    
    for old_name, new_name in replacements.items():
        # Handle both regular and LaTeX-escaped underscores
        old_pattern = old_name.replace('_', r'\\_')
        new_pattern = new_name.replace('_', r'\\_')
        
        # Replace in regular form
        if old_name in content:
            content = content.replace(old_name, new_name)
            changes.append((old_name, new_name))
        
        # Replace in LaTeX-escaped form
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)
            if (old_name, new_name) not in changes:
                changes.append((old_name, new_name))
    
    if content != original_content:
        if not dry_run:
            filepath.write_text(content)
        print(f"  ✅ {filepath.name}: {len(changes)} replacements")
        if changes:
            for old, new in changes[:3]:
                print(f"      {old} → {new}")
            if len(changes) > 3:
                print(f"      ... and {len(changes) - 3} more")
        return True
    return False

def update_latex_file(dry_run=False):
    """Update LaTeX file with new experiment names."""
    print("\n1. Updating LaTeX paper...")
    tex_file = pathlib.Path('experiments_paper_overleaf.tex')
    return replace_in_file(tex_file, mapping, dry_run)

def update_auto_start_script(dry_run=False):
    """Update auto_start_fixed_experiments.py."""
    print("\n2. Updating auto_start_fixed_experiments.py...")
    script_file = pathlib.Path('auto_start_fixed_experiments.py')
    return replace_in_file(script_file, mapping, dry_run)

def update_resume_file(dry_run=False):
    """Update command templates in resume file."""
    print("\n3. Updating resume file command templates...")
    resume_file = pathlib.Path('stopped_experiments_resume_info.json')
    if not resume_file.exists():
        print("  ⚠️  Resume file not found")
        return False
    
    with open(resume_file) as f:
        resume_data = json.load(f)
    
    updated = False
    for exp in resume_data.get('stopped_experiments', []):
        if 'command_template' in exp and exp['command_template']:
            old_cmd = exp['command_template']
            new_cmd = old_cmd
            for old_name, new_name in mapping.items():
                new_cmd = new_cmd.replace(old_name, new_name)
            if new_cmd != old_cmd:
                exp['command_template'] = new_cmd
                updated = True
    
    if updated and not dry_run:
        with open(resume_file, 'w') as f:
            json.dump(resume_data, f, indent=2)
        print(f"  ✅ Updated command templates in resume file")
        return True
    elif updated:
        print(f"  ✅ Would update command templates in resume file")
        return True
    else:
        print(f"  ℹ️  No command templates to update")
        return False

def update_config_files(dry_run=False):
    """Update model paths in config.json files."""
    print("\n4. Updating config.json files...")
    config_files = list(pathlib.Path('experiments/retrieval').rglob('config.json'))
    updated_count = 0
    
    for config_file in config_files:
        try:
            with open(config_file) as f:
                config = json.load(f)
            
            config_str = json.dumps(config)
            new_config_str = config_str
            
            # Replace old model paths
            for old_name, new_name in mapping.items():
                # Replace in paths like ./models/phase2_augmentation or /models/phase2_augmentation
                patterns = [
                    f'./models/{old_name}',
                    f'/models/{old_name}',
                    f'"models/{old_name}"',
                    f"'models/{old_name}'",
                ]
                replacements = [
                    f'./models/{new_name}',
                    f'/models/{new_name}',
                    f'"models/{new_name}"',
                    f"'models/{new_name}'",
                ]
                
                for old_pattern, new_pattern in zip(patterns, replacements):
                    new_config_str = new_config_str.replace(old_pattern, new_pattern)
            
            if new_config_str != config_str:
                if not dry_run:
                    new_config = json.loads(new_config_str)
                    with open(config_file, 'w') as f:
                        json.dump(new_config, f, indent=2)
                updated_count += 1
        except Exception as e:
            print(f"  ⚠️  Error processing {config_file}: {e}")
    
    if updated_count > 0:
        print(f"  ✅ Updated {updated_count} config files")
        return True
    else:
        print(f"  ℹ️  No config files need updating")
        return False

def main():
    dry_run = '--dry-run' in sys.argv
    
    if dry_run:
        print("=" * 80)
        print("DRY RUN - No files will be modified")
        print("=" * 80)
    
    print(f"\nUpdating old experiment names in files...")
    print(f"Mapping contains {len(mapping)} experiment name mappings\n")
    
    results = []
    results.append(update_latex_file(dry_run))
    results.append(update_auto_start_script(dry_run))
    results.append(update_resume_file(dry_run))
    results.append(update_config_files(dry_run))
    
    print("\n" + "=" * 80)
    if dry_run:
        print("DRY RUN COMPLETE - No files were modified")
    else:
        print("UPDATE COMPLETE")
        print(f"  Updated {sum(results)} file categories")
    print("=" * 80)

if __name__ == '__main__':
    main()

