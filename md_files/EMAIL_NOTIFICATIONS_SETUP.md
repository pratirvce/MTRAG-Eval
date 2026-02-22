# Email Notifications Setup for Experiment Failures

**Email**: pratirvce@gmail.com  
**Status**: ✅ **Configured**

---

## ✅ Email Notifications Enabled

The monitoring script (`monitor_experiments.py`) has been enhanced to send email notifications when experiments fail.

---

## 📧 How It Works

### **Automatic Email Notifications**:

1. **Failure Detection**: Monitor detects experiment failures
2. **Email Generation**: Creates detailed failure report
3. **Email Sending**: Sends to `pratirvce@gmail.com`
4. **Duplicate Prevention**: Avoids sending duplicate notifications for same failure

### **What Triggers Email**:

- ✅ Experiment process stopped unexpectedly
- ✅ No results file created (experiment failed)
- ✅ Errors detected in log files
- ✅ Non-zero exit codes

---

## 🔧 Email Configuration

### **Default Settings**:
- **To**: `pratirvce@gmail.com`
- **From**: `experiments@<hostname>`
- **Method**: Uses system `mail` or `sendmail` command (if available)

### **SMTP Configuration (Optional)**:

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

## 🚀 Usage

### **Start Monitoring with Email Notifications**:

```bash
# Continuous monitoring (checks every 60 seconds)
python monitor_experiments.py --check-interval 60

# Or in background
nohup python monitor_experiments.py --check-interval 60 > monitor.log 2>&1 &
```

### **One-Time Check with Email**:

```bash
python monitor_experiments.py --once
```

---

## 📧 Email Content

When an experiment fails, you'll receive an email with:

- **Subject**: `🚨 X Experiment(s) Failed - MTRAG Benchmark`
- **Body**: Detailed failure report including:
  - Experiment name and type
  - PID and GPU ID
  - Status and exit code
  - Config file path
  - Log file path
  - Error details (first error found)
  - Instructions to fix and restart

---

## 🔍 Testing Email Notifications

### **Test 1: Check if mail command works**:
```bash
echo "Test email from MTRAG monitoring" | mail -s "Test" pratirvce@gmail.com
```

### **Test 2: Test monitoring script**:
```bash
# Run one-time check
python monitor_experiments.py --once

# If there are failures, email will be sent automatically
```

### **Test 3: Simulate failure**:
```bash
# Create a test failure
echo "Error: Test failure" >> experiments/retrieval/phase7_test/training.log

# Run monitor (it should detect and email)
python monitor_experiments.py --once
```

---

## 📋 Email Notification Features

### **Duplicate Prevention**:
- Tracks which failures have been notified
- Avoids sending duplicate emails for same failure
- Saves sent notifications to `experiment_failures_sent.json`

### **Error Details**:
- Shows first error pattern found
- Includes error line context
- Provides log file path for detailed investigation

### **Actionable Information**:
- Config file path
- GPU ID used
- Restart instructions
- Full failure report location

---

## 🛠️ Troubleshooting

### **Email Not Sending**:

1. **Check if mail/sendmail is installed**:
   ```bash
   which mail
   which sendmail
   ```

2. **Install mailutils (Ubuntu/Debian)**:
   ```bash
   sudo apt-get install mailutils
   ```

3. **Install sendmail (CentOS/RHEL)**:
   ```bash
   sudo yum install sendmail
   ```

4. **Use SMTP instead**:
   ```bash
   export SMTP_SERVER='smtp.gmail.com'
   export SMTP_PORT='587'
   export SMTP_USER='your-email@gmail.com'
   export SMTP_PASSWORD='your-app-password'
   ```

### **Check Email Logs**:

```bash
# Check monitor log
tail -f monitor.log

# Check system mail log
tail -f /var/log/mail.log  # Ubuntu/Debian
tail -f /var/log/maillog   # CentOS/RHEL
```

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

**Email Notifications**: ✅ **Enabled**  
**Email Address**: `pratirvce@gmail.com`  
**Status**: Ready to send notifications on experiment failures

**To Start Monitoring**:
```bash
python monitor_experiments.py --check-interval 60
```

**You will receive email notifications automatically when experiments fail!** 📧

