#!/bin/bash
# Monitor GPU usage in real-time

echo "Monitoring GPU usage (press Ctrl+C to stop)"
echo ""

watch -n 2 nvidia-smi
