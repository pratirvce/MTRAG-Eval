# Move Markdown Files

To move all .md files (except COMPREHENSIVE_EXPERIMENT_REPORT.md) to the md_files folder, run:

```bash
cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark
python3 execute_move.py
```

Or use the shell script:

```bash
bash RUN_MOVE_SCRIPT.sh
```

The script will:
- Create the `md_files` directory if it doesn't exist
- Move all `.md` files from the root directory to `md_files/`
- Exclude `COMPREHENSIVE_EXPERIMENT_REPORT.md` (as requested)
- Skip files that already exist in `md_files/`

