# Commands to Make Files Viewable and Copyable for Everyone

## For a Specific Directory

To make all files in `/home/prevanka/prati/su-mt-rag/mt-rag-benchmark` viewable and copyable by everyone:

```bash
# Set files to 644 (owner can read/write, everyone else can read/copy)
find /home/prevanka/prati/su-mt-rag/mt-rag-benchmark -type f -exec chmod 644 {} \;

# Set directories to 755 (owner full access, everyone else can read/traverse)
find /home/prevanka/prati/su-mt-rag/mt-rag-benchmark -type d -exec chmod 755 {} \;
```

## For Entire Directory Tree (/home/prevanka/prati/su-mt-rag)

```bash
# Set files to 644 (readable by everyone)
find /home/prevanka/prati/su-mt-rag -type f -exec chmod 644 {} \;

# Set directories to 755 (traversable by everyone)
find /home/prevanka/prati/su-mt-rag -type d -exec chmod 755 {} \;
```

## Completely Public (Read-Only for Everyone)

If you want **everyone** (including non-owners) to have read-only access:

```bash
# Files: 444 (read-only for everyone, including owner)
find /home/prevanka/prati/su-mt-rag/mt-rag-benchmark -type f -exec chmod 444 {} \;

# Directories: 555 (read and traverse for everyone, no write)
find /home/prevanka/prati/su-mt-rag/mt-rag-benchmark -type d -exec chmod 555 {} \;
```

## For a Single File

To make a specific file viewable/copyable:

```bash
# Make readable by everyone (644)
chmod 644 /home/prevanka/prati/su-mt-rag/mt-rag-benchmark/set_writable_su_mt_rag.sh

# Or completely public read-only (444)
chmod 444 /home/prevanka/prati/su-mt-rag/mt-rag-benchmark/set_writable_su_mt_rag.sh
```

## Permission Meanings

### 644 (Recommended for sharing)
- **Owner**: read + write (rw-)
- **Group**: read (r--)
- **Others**: read (r--)
- ✅ Others can view and copy
- ✅ You can still modify

### 444 (Completely read-only)
- **Owner**: read (r--)
- **Group**: read (r--)
- **Others**: read (r--)
- ✅ Everyone can view and copy
- ❌ No one can modify (even you, without changing permissions first)

### 755 for Directories
- **Owner**: read + write + execute (rwx)
- **Group**: read + execute (r-x)
- **Others**: read + execute (r-x)
- ✅ Others can list contents and enter directories
- ❌ Others cannot create/delete files

### 555 for Directories (Read-only)
- **Owner**: read + execute (r-x)
- **Group**: read + execute (r-x)
- **Others**: read + execute (r-x)
- ✅ Others can list and enter
- ❌ No one can create/delete files

## Verify Permissions

Check if permissions are set correctly:

```bash
# Check a file
ls -l /home/prevanka/prati/su-mt-rag/mt-rag-benchmark/set_writable_su_mt_rag.sh

# Check a directory
ls -ld /home/prevanka/prati/su-mt-rag/mt-rag-benchmark

# Check recursively
find /home/prevanka/prati/su-mt-rag/mt-rag-benchmark -exec ls -ld {} \; | head -20
```

## Example: Make submission folder public

```bash
# Make submission folder and all contents viewable/copyable
find /home/prevanka/prati/su-mt-rag/mt-rag-benchmark/submission -type f -exec chmod 644 {} \;
find /home/prevanka/prati/su-mt-rag/mt-rag-benchmark/submission -type d -exec chmod 755 {} \;
```

## What Users Can Do After Setting 644/755

✅ **Can:**
- View file contents (`cat`, `less`, `more`, `head`, `tail`)
- Copy files to their own locations (`cp file ~/copy`)
- List directory contents (`ls`)
- Navigate into directories (`cd`)
- Read files with any program

❌ **Cannot (with 644/755):**
- Modify files (unless they're the owner)
- Delete files (unless they're the owner)
- Create new files (unless they're the owner)

## Quick Reference

```bash
# Make files readable by everyone (you keep write access)
chmod 644 filename
find directory -type f -exec chmod 644 {} \;

# Make files completely read-only for everyone
chmod 444 filename
find directory -type f -exec chmod 444 {} \;

# Make directories traversable by everyone (you keep write access)
chmod 755 dirname
find directory -type d -exec chmod 755 {} \;

# Make directories read-only for everyone
chmod 555 dirname
find directory -type d -exec chmod 555 {} \;
```

