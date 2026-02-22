# ✅ Email Notifications Configured

**Date**: 2025-12-16 23:35  
**Status**: ✅ **ACTIVE**

---

## ✅ Email Notifications Enabled

The experiment monitoring system has been enhanced with **email notifications** that will automatically send alerts to **pratirvce@gmail.com** when experiments fail.

---

## 📧 Configuration

- **Email Address**: `pratirvce@gmail.com`
- **Notification Method**: 
  - Primary: System `mail` command
  - Fallback: `sendmail` command
  - Alternative: SMTP (if configured via environment variables)
- **Duplicate Prevention**: ✅ Enabled (tracks sent notifications)

---

## 🔔 What Triggers Email

You will receive an email when:

1. ✅ **Experiment process stops unexpectedly**
2. ✅ **No results file created** (experiment failed)
3. ✅ **Errors detected in log files** (Error, Traceback, Exception, etc.)
4. ✅ **Non-zero exit codes**

---

## 📧 Email Content

Each email includes:

- **Subject**: `🚨 X Experiment(s) Failed - MTRAG Benchmark`
- **Body**: Detailed failure report with:
  - Experiment name and type
  - PID and GPU ID
  - Status and exit code
  - Config file path
  - Log file path
  - Error details (first error found)
  - Instructions to fix and restart
  - Timestamp

---

## 🚀 Current Status

### **Monitor Running**: ✅ **YES**

- **Process**: Running in background
- **Check Interval**: 5 minutes (300 seconds)
- **Log File**: `monitor.log`
- **Status File**: `experiment_status.json`
- **Failures File**: `experiment_failures.json`

### **Monitor Commands**:

```bash
# Check monitor status
ps aux | grep monitor_experiments | grep -v grep

# View monitor log
tail -f monitor.log

# Run one-time check
python monitor_experiments.py --once

# Start/restart monitor
./start_monitor_with_email.sh
```

---

## 📋 Email Notification Features

### **1. Automatic Detection**
- Monitors all running experiments every 5 minutes
- Detects failures in real-time
- Checks log files for error patterns

### **2. Duplicate Prevention**
- Tracks which failures have been notified
- Avoids sending duplicate emails
- Saves sent notifications to `experiment_failures_sent.json`

### **3. Multiple Delivery Methods**
- Tries `mail` command first
- Falls back to `sendmail` if available
- Can use SMTP if configured

### **4. Detailed Error Information**
- Shows first error pattern found
- Includes error line context
- Provides log file path for investigation

---

## 🛠️ SMTP Configuration (Optional)

If you want to use SMTP (e.g., Gmail), set these environment variables:

```bash
export SMTP_SERVER='smtp.gmail.com'
export SMTP_PORT='587'
export SMTP_USER='your-email@gmail.com'
export SMTP_PASSWORD='your-app-password'
export EMAIL_FROM='your-email@gmail.com'
```

**For Gmail**:
1. Enable 2-factor authentication
2. Generate an "App Password" (not your regular password)
3. Use the app password in `SMTP_PASSWORD`

---

## 📝 Example Email

```
Subject: 🚨 1 Experiment(s) Failed - MTRAG Benchmark

⚠️  EXPERIMENT FAILURES DETECTED

1 experiment(s) failed and need attention:

1. phase6_llm_query_expansion_gpt4_multi (llm_expansion)
   PID: 12345
   GPU: 2
   Status: stopped
   Config: experiments/retrieval/phase6_llm_query_expansion_gpt4_multi/config.json
   Log: experiments/retrieval/phase6_llm_query_expansion_gpt4_multi/training.log
   Errors Found: 1
   First Error: IndexError
   Error Line: IndexError: index 1 is out of bounds for dimension 0 with size 1...

📝 To fix and restart:
   1. Check the log files for detailed error messages
   2. Fix the issues in the code/config
   3. Restart with: python <script> --config <config> --gpu_id <gpu> --resume

📄 Full failure report saved to: experiment_failures.json

Timestamp: 2025-12-16 23:35:00
```

---

## ✅ Summary

**Email Notifications**: ✅ **ACTIVE**  
**Email Address**: `pratirvce@gmail.com`  
**Monitor Status**: ✅ **Running**  
**Check Interval**: 5 minutes

**You will receive email notifications automatically when experiments fail!** 📧

---

## 🔍 Testing

To test email notifications:

```bash
# Run one-time check (will email if failures found)
python monitor_experiments.py --once

# Check if email was sent
tail -f monitor.log | grep -i email
```

---

## 📚 Related Files

- `monitor_experiments.py` - Main monitoring script with email support
- `EMAIL_NOTIFICATIONS_SETUP.md` - Detailed setup guide
- `start_monitor_with_email.sh` - Script to start monitor with email
- `setup_email_notifications.sh` - Email setup helper script

---

**All set! You'll be notified by email when experiments fail.** 🎯

