#!/bin/bash
#
# Script to set read-only permissions for files and directories
# Files: 444 (r--r--r--) - readable by all, no write/modify
# Directories: 555 (r-xr-xr-x) - readable and traversable, no write/modify
#
# Usage:
#   ./set_readonly_permissions.sh [directory]
#   If no directory specified, uses current directory (.)
#

TARGET_DIR="${1:-.}"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist"
    exit 1
fi

echo "Setting read-only permissions on: $TARGET_DIR"
echo "=============================================="
echo ""
echo "This will:"
echo "  - Set files to 444 (read-only for everyone)"
echo "  - Set directories to 555 (read and traverse, no write)"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Setting permissions..."

# First, remove write permissions from all files and directories recursively
# This ensures no one can modify anything
find "$TARGET_DIR" -type f -exec chmod 444 {} \;
find "$TARGET_DIR" -type d -exec chmod 555 {} \;

# Count files and directories
FILE_COUNT=$(find "$TARGET_DIR" -type f | wc -l)
DIR_COUNT=$(find "$TARGET_DIR" -type d | wc -l)

echo ""
echo "✅ Permissions set successfully!"
echo "   - Files processed: $FILE_COUNT"
echo "   - Directories processed: $DIR_COUNT"
echo ""
echo "Permission summary:"
echo "  Files:     444 (r--r--r--) = read-only for all"
echo "  Directories: 555 (r-xr-xr-x) = read and traverse, no write"
echo ""
echo "Users can now:"
echo "  ✅ View/copy files"
echo "  ✅ List directory contents"
echo "  ✅ Navigate into subdirectories"
echo ""
echo "Users cannot:"
echo "  ❌ Modify any files"
echo "  ❌ Create new files"
echo "  ❌ Delete files"
echo "  ❌ Rename files"

