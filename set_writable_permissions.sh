#!/bin/bash
#
# Script to restore normal write permissions (for when you need to modify files)
# Files: 644 (rw-r--r--) - owner can read/write, others can read
# Directories: 755 (rwxr-xr-x) - owner can read/write/traverse, others can read/traverse
#
# Usage:
#   ./set_writable_permissions.sh [directory]
#   If no directory specified, uses current directory (.)
#

TARGET_DIR="${1:-.}"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist"
    exit 1
fi

echo "Restoring writable permissions on: $TARGET_DIR"
echo "=============================================="
echo ""
echo "This will:"
echo "  - Set files to 644 (owner can read/write, others read-only)"
echo "  - Set directories to 755 (owner can read/write/traverse, others read/traverse)"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Setting permissions..."

# Restore normal permissions
find "$TARGET_DIR" -type f -exec chmod 644 {} \;
find "$TARGET_DIR" -type d -exec chmod 755 {} \;

# Count files and directories
FILE_COUNT=$(find "$TARGET_DIR" -type f | wc -l)
DIR_COUNT=$(find "$TARGET_DIR" -type d | wc -l)

echo ""
echo "✅ Permissions restored!"
echo "   - Files processed: $FILE_COUNT"
echo "   - Directories processed: $DIR_COUNT"
echo ""
echo "Permission summary:"
echo "  Files:     644 (rw-r--r--) = owner can read/write, others read-only"
echo "  Directories: 755 (rwxr-xr-x) = owner full access, others read/traverse"

