#!/bin/bash
#
# Script to set read-only permissions for /home/prevanka/prati/su-mt-rag
# Files: 444 (r--r--r--) - readable by all, no write/modify
# Directories: 555 (r-xr-xr-x) - readable and traversable, no write/modify
#

TARGET_DIR="/home/prevanka/prati/su-mt-rag"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist"
    exit 1
fi

echo "Setting read-only permissions on: $TARGET_DIR"
echo "=============================================="
echo ""
echo "⚠️  WARNING: This will make ALL files and directories in"
echo "   $TARGET_DIR"
echo "   read-only for everyone (including subdirectories)"
echo ""
echo "This will:"
echo "  - Set ALL files to 444 (read-only for everyone)"
echo "  - Set ALL directories to 555 (read and traverse, no write)"
echo "  - Apply recursively to ALL subdirectories"
echo ""
read -p "Are you sure you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Setting permissions (this may take a while for large directories)..."

# Set permissions recursively
# Files: 444 (read-only)
find "$TARGET_DIR" -type f -exec chmod 444 {} \;

# Directories: 555 (read and traverse, no write)
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
echo "  ✅ View/copy files from $TARGET_DIR"
echo "  ✅ List directory contents"
echo "  ✅ Navigate into subdirectories"
echo ""
echo "Users cannot:"
echo "  ❌ Modify any files"
echo "  ❌ Create new files"
echo "  ❌ Delete files"
echo "  ❌ Rename files"
echo ""
echo "Note: To restore write permissions later, run:"
echo "  ./set_writable_su_mt_rag.sh"

