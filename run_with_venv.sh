#!/bin/bash
# Script to run the generation and copy script with virtual environment

cd "$(dirname "$0")"

# Try to find and activate virtual environment
if [ -d "venv" ]; then
    echo "Activating venv in current directory..."
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo "Activating venv in parent directory..."
    source ../venv/bin/activate
else
    echo "No venv found, using system python3"
fi

echo "Running generate_and_copy.py..."
python3 generate_and_copy.py

echo ""
echo "Done! Check the output above for any errors."

