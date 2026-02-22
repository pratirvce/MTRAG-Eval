# Safe Duplicate Process Stopper

## Overview

`stop_duplicate_process_safe.sh` is a comprehensive, safety-checked script to stop the duplicate `phase4_hard_negatives_cosine` process.

## Safety Features

### 8 Pre-Flight Checks:

1. **Process Existence** ✅
   - Verifies both original and parallel processes are running
   - Won't proceed if parallel process is missing

2. **Experiment Verification** ✅
   - Confirms both processes are running the same experiment
   - Prevents accidental stopping of wrong process

3. **Configuration Verification** ✅
   - Checks config file exists
   - Verifies processes use same configuration

4. **Checkpoint Status** ✅
   - Checks for existing checkpoints
   - Warns if progress will be lost
   - Requires confirmation if checkpoints exist

5. **Process Management Status** ✅
   - Checks if processes are part of managed system
   - Confirms original is safe to stop

6. **GPU Resource Analysis** ✅
   - Shows GPU utilization for both processes
   - Displays memory usage
   - Helps verify which GPU will be freed

7. **Runtime Comparison** ✅
   - Shows how long each process has been running
   - Helps identify which is older/newer

8. **Log File Analysis** ✅
   - Checks error logs for issues
   - Shows recent training log activity
   - Helps identify if process is stuck

## Usage

```bash
# Make executable (already done)
chmod +x stop_duplicate_process_safe.sh

# Run the safe script
./stop_duplicate_process_safe.sh
```

## What It Does

1. **Pre-Flight Checks**: Runs all 8 safety checks
2. **Summary Display**: Shows detailed comparison
3. **Backup Record**: Creates timestamped backup of process info
4. **Double Confirmation**: Requires typing 'STOP' to confirm
5. **Graceful Stop**: Sends SIGTERM first (30s wait)
6. **Force Stop**: Uses SIGKILL if needed
7. **Verification**: Confirms process stopped and parallel still running
8. **Final Report**: Shows GPU status after stop

## Safety Confirmations

The script requires **TWO confirmations**:

1. **Checkpoint Warning** (if checkpoints exist):
   ```
   Continue anyway? (yes/no):
   ```

2. **Final Confirmation** (always required):
   ```
   CONFIRM: Stop process 1311837? (type 'STOP' to confirm):
   ```

You must type exactly `STOP` (capital letters) to proceed.

## Backup Record

Before stopping, the script creates a backup record in:
```
experiments/retrieval/phase4_hard_negatives_cosine/process_analysis_YYYYMMDD_HHMMSS/stop_record.txt
```

This contains:
- Process details for both processes
- Timestamp of stop
- Checkpoint count
- Process information

## Exit Codes

- `0`: Success (process stopped)
- `1`: Error (checks failed or stop failed)
- `0`: Cancelled by user (safe exit)

## Example Output

```
═══════════════════════════════════════════════════════════════
     SAFE DUPLICATE PROCESS STOPPER - PRE-FLIGHT CHECKS
═══════════════════════════════════════════════════════════════

CHECK 1: Process Existence
✅ Original process (1311837) is running
✅ Parallel process (1850801) is running

CHECK 2: Experiment Verification
✅ Both processes are running phase4_hard_negatives_cosine

CHECK 3: Configuration Verification
✅ Config file exists

CHECK 4: Checkpoint Status
✅ No checkpoints found (still in mining phase)
   ⚠️  All mining progress will be lost (expected)

[... more checks ...]

⚠️  CONFIRM: Stop process 1311837? (type 'STOP' to confirm):
```

## Comparison with Simple Script

| Feature | Simple Script | Safe Script |
|---------|--------------|-------------|
| Process checks | Basic | Comprehensive (8 checks) |
| Checkpoint check | No | Yes (with warning) |
| Log analysis | No | Yes |
| GPU analysis | No | Yes |
| Backup record | No | Yes |
| Double confirmation | No | Yes (must type STOP) |
| Error handling | Basic | Extensive |
| Progress reporting | Basic | Detailed |

## When to Use

Use the **safe script** when:
- ✅ You want comprehensive safety checks
- ✅ You want to verify everything before stopping
- ✅ You want a backup record
- ✅ You want detailed analysis

Use the **simple script** when:
- ✅ You've already verified everything manually
- ✅ You need a quick stop
- ✅ You don't need all the checks

## Troubleshooting

### "Process not found"
- Process may have already stopped
- Check with: `ps -p 1311837`

### "Parallel process not running"
- Don't stop original if parallel is also stopped
- Check with: `ps -p 1850801`

### "Checkpoints found"
- Script will warn you
- Type `yes` to continue (progress after checkpoint will be lost)
- Type `no` to cancel

### Process won't stop
- Script will try graceful stop first (30s)
- Then force stop (SIGKILL)
- If still running, check permissions or process state

