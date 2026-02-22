#!/bin/bash

# Setup email notifications for experiment monitoring
# This script helps configure email notifications

echo "=== Email Notification Setup for Experiment Monitoring ==="
echo ""

EMAIL_TO="pratirvce@gmail.com"

# Check if mail command is available
if command -v mail &> /dev/null; then
    echo "✅ 'mail' command found"
    echo ""
    echo "To test email notifications, run:"
    echo "  echo 'Test email' | mail -s 'Test' $EMAIL_TO"
    echo ""
elif command -v sendmail &> /dev/null; then
    echo "✅ 'sendmail' command found"
    echo ""
    echo "To test email notifications, run:"
    echo "  echo 'Test email' | sendmail $EMAIL_TO"
    echo ""
else
    echo "⚠️  No mail/sendmail command found"
    echo ""
    echo "Options:"
    echo "1. Install mailutils (Ubuntu/Debian):"
    echo "   sudo apt-get install mailutils"
    echo ""
    echo "2. Install sendmail (CentOS/RHEL):"
    echo "   sudo yum install sendmail"
    echo ""
    echo "3. Use SMTP (configure via environment variables):"
    echo "   export SMTP_SERVER='smtp.gmail.com'"
    echo "   export SMTP_PORT='587'"
    echo "   export SMTP_USER='your-email@gmail.com'"
    echo "   export SMTP_PASSWORD='your-app-password'"
    echo "   export EMAIL_FROM='your-email@gmail.com'"
    echo ""
fi

echo "Email will be sent to: $EMAIL_TO"
echo ""
echo "To start monitoring with email notifications:"
echo "  python monitor_experiments.py --check-interval 60"
echo ""
echo "Or run in background:"
echo "  nohup python monitor_experiments.py --check-interval 60 > monitor.log 2>&1 &"

