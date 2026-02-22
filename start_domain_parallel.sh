#!/bin/bash
# Start domain-specific experiments in parallel on separate GPUs

cd "$(dirname "$0")"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=== Starting Domain-Specific Parallel Experiments ==="
echo ""
echo "Each domain will run on a separate GPU:"
echo "  - clapnq → GPU 1"
echo "  - fiqa → GPU 2"
echo "  - govt → GPU 3"
echo "  - cloud → GPU 4"
echo ""

python run_domain_specific_parallel.py

echo ""
echo "✅ Domain-specific experiments completed!"


