# Permissions for Read, Copy, and Execute (No Edit)

## Recommended Permissions

### For Files (Scripts/Executables)
```bash
# Set files to 555 = r-xr-xr-x (read + execute, no write)
find /home/prevanka/prati/su-mt-rag/mt-rag-benchmark -type f -exec chmod 555 {} \;
```

**555 Permission Breakdown:**
- **Owner**: `r-x` (read + execute, **no write**)
- **Group**: `r-x` (read + execute, **no write**)
- **Others**: `r-x` (read + execute, **no write**)

✅ **Everyone can:**
- Read/view file contents
- Copy files to their own locations
- Execute scripts/programs
- Run files

❌ **No one can:**
- Edit/modify files
- Delete files
- Create new files (in directories)

### For Regular Files (Non-Executable)
```bash
# Set files to 444 = r--r--r-- (read-only)
find /home/prevanka/prati/su-mt-rag/mt-rag-benchmark -type f -exec chmod 444 {} \;
```

**444 Permission Breakdown:**
- **Owner**: `r--` (read only, **no write/execute**)
- **Group**: `r--` (read only)
- **Others**: `r--` (read only)

### For Directories
```bash
# Set directories to 555 = r-xr-xr-x (read + execute, no write)
find /home/prevanka/prati/su-mt-rag/mt-rag-benchmark -type d -exec chmod 555 {} \;
```

**555 Directory Permission Breakdown:**
- **Owner**: `r-x` (read + execute, **no write**)
- **Group**: `r-x` (read + execute, **no write**)
- **Others**: `r-x` (read + execute, **no write**)

✅ **Everyone can:**
- List directory contents (`ls`)
- Navigate into directories (`cd`)
- Read files inside (if file permissions allow)
- Execute files inside

❌ **No one can:**
- Create new files/directories
- Delete files/directories
- Modify files (unless file permissions allow)

---

## Hybrid Approach (Recommended)

Since you have both executable scripts (`.sh`, `.py`) and regular files (`.md`, `.txt`, `.json`), use:

```bash
# First, make all files read-only (444)
find /home/prevanka/prati/su-mt-rag/mt-rag-benchmark -type f -exec chmod 444 {} \;

# Then, make script files executable (555)
find /home/prevanka/prati/su-mt-rag/mt-rag-benchmark -type f \( -name "*.sh" -o -name "*.py" \) -exec chmod 555 {} \;

# Make directories traversable but not writable (555)
find /home/prevanka/prati/su-mt-rag/mt-rag-benchmark -type d -exec chmod 555 {} \;
```

---

## Permission Number Reference

| Permission | Binary | Meaning |
|------------|--------|---------|
| **4** | `100` | Read (r) |
| **2** | `010` | Write (w) |
| **1** | `001` | Execute (x) |

**Common Combinations:**
- **444** = `r--r--r--` = Read-only for everyone
- **555** = `r-xr-xr-x` = Read + execute for everyone (no write)
- **644** = `rw-r--r--` = Owner read/write, others read-only
- **755** = `rwxr-xr-x` = Owner full, others read+execute

---

## What Users Can Do with 555/444 Permissions

### ✅ Allowed Actions

```bash
# Read files
cat file.txt
less file.py
head file.json

# Copy files
cp source.py ~/my_copy.py
scp user@host:/path/file.py ~/

# Execute scripts
./script.sh
python script.py

# List directories
ls -l directory/

# Navigate
cd directory/subdirectory/
```

### ❌ Prohibited Actions

```bash
# Edit files
nano file.txt      # ❌ Permission denied
vim file.py        # ❌ Permission denied
echo "test" >> file.txt  # ❌ Permission denied

# Delete files
rm file.txt        # ❌ Permission denied

# Create files
touch new_file.txt # ❌ Permission denied
mkdir new_dir      # ❌ Permission denied
```

---

## Complete Command Set

### Option 1: All Files Executable (555)

```bash
cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark

# Files: read + execute (555)
find . -type f -exec chmod 555 {} \;

# Directories: read + execute (555)
find . -type d -exec chmod 555 {} \;
```

### Option 2: Only Scripts Executable (444 + selective 555)

```bash
cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark

# First, make all files read-only
find . -type f -exec chmod 444 {} \;

# Then, make scripts executable
find . -type f \( -name "*.sh" -o -name "*.py" -o -name "*.pl" -o -name "*.rb" \) -exec chmod 555 {} \;

# Directories: read + execute
find . -type d -exec chmod 555 {} \;
```

### Option 3: Keep Owner Write Access (Recommended if you still need to edit)

If you want others to read/copy/execute but you still need to edit:

```bash
cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark

# Files: owner can write, others can only read/execute (755 for scripts, 644 for regular)
find . -type f -exec chmod 644 {} \;
find . -type f \( -name "*.sh" -o -name "*.py" \) -exec chmod 755 {} \;

# Directories: owner can write, others can read/traverse (755)
find . -type d -exec chmod 755 {} \;
```

This gives:
- **You (owner)**: Full read/write/execute
- **Others**: Read + execute (can view, copy, run, but not edit)

---

## Verify Permissions

After setting permissions, verify:

```bash
# Check a file
ls -l /home/prevanka/prati/su-mt-rag/mt-rag-benchmark/some_file.py

# Check a directory
ls -ld /home/prevanka/prati/su-mt-rag/mt-rag-benchmark/

# Check permissions distribution
find /home/prevanka/prati/su-mt-rag/mt-rag-benchmark -type f -exec stat -c "%a" {} \; | sort | uniq -c
find /home/prevanka/prati/su-mt-rag/mt-rag-benchmark -type d -exec stat -c "%a" {} \; | sort | uniq -c
```

---

## Quick Answer

**For read + copy + execute but NO edit:**

```bash
# Files: 555 (read + execute for everyone)
find /home/prevanka/prati/su-mt-rag/mt-rag-benchmark -type f -exec chmod 555 {} \;

# Directories: 555 (read + execute for everyone)
find /home/prevanka/prati/su-mt-rag/mt-rag-benchmark -type d -exec chmod 555 {} \;
```

**Or if you want to keep your own write access:**

```bash
# Files: 755 for scripts, 644 for regular (you can write, others can't)
find /home/prevanka/prati/su-mt-rag/mt-rag-benchmark -type f -exec chmod 644 {} \;
find /home/prevanka/prati/su-mt-rag/mt-rag-benchmark -type f \( -name "*.sh" -o -name "*.py" \) -exec chmod 755 {} \;

# Directories: 755 (you can write, others can't)
find /home/prevanka/prati/su-mt-rag/mt-rag-benchmark -type d -exec chmod 755 {} \;
```

