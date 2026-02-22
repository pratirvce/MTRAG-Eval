#!/usr/bin/env python3
"""
Generate IEEE ACL paper and figures, then copy to submission folder.
"""

import sys
import pathlib

# Import and run generation script functions directly
def main():
    """Run generation script then copy files."""
    print("=" * 70)
    print("Step 1: Generating IEEE ACL Paper and Figures")
    print("=" * 70)
    
    # Import and run generation
    try:
        import generate_ieee_acl_paper
        generate_ieee_acl_paper.main()
    except Exception as e:
        print(f"❌ Error running generation script: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("Step 2: Copying Generated Files to Submission Folder")
    print("=" * 70)
    
    # Import and run copy script
    try:
        import copy_generated_files
        copy_generated_files.copy_files()
    except Exception as e:
        print(f"❌ Error copying files: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

