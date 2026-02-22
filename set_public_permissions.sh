#!/bin/bash

# Script to set permissions for everyone to view and copy files
# Usage: ./set_public_permissions.sh [readonly]
#   - Without arguments: sets 644/755 (readable, you can still write)
#   - With "readonly": sets 444/555 (completely read-only for everyone)

TARGET_DIR="/home/prevanka/prati/su-mt-rag/mt-rag-benchmark"

if [ ! -d "$TARGET_DIR" ]; then
    echo "❌ Error: Directory $TARGET_DIR does not exist"
    exit 1
fi

if [ "$1" = "readonly" ]; then
    echo "Setting completely read-only permissions (444/555) for everyone..."
    FILE_PERM="444"
    DIR_PERM="555"
    MODE="read-only"
else
    echo "Setting readable permissions (644/755) - you can still write..."
    FILE_PERM="644"
    DIR_PERM="755"
    MODE="readable (you keep write access)"
fi

echo ""
echo "Target directory: $TARGET_DIR"
echo "Mode: $MODE"
echo ""

# Count files and directories first
FILE_COUNT=$(find "$TARGET_DIR" -type f | wc -l)
DIR_COUNT=$(find "$TARGET_DIR" -type d | wc -l)

echo "Found:"
echo "  - $FILE_COUNT files"
echo "  - $DIR_COUNT directories"
echo ""
echo "Setting permissions..."

# Set permissions for files
find "$TARGET_DIR" -type f -exec chmod $FILE_PERM {} \;
echo "✅ Set files to $FILE_PERM"

# Set permissions for directories
find "$TARGET_DIR" -type d -exec chmod $DIR_PERM {} \;
echo "✅ Set directories to $DIR_PERM"

echo ""
echo "✅ Done! Permissions set for all files and directories."
echo ""
echo "Current permissions summary:"
echo "  - Files: $FILE_PERM"
echo "  - Directories: $DIR_PERM"
echo ""
echo "Verification (sample):"
ls -ld "$TARGET_DIR"
ls -l "$TARGET_DIR" | head -5

