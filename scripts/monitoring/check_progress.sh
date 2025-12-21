#!/bin/bash
# Check training progress

PROGRESS_FILE="training_progress.txt"

if [ ! -f "$PROGRESS_FILE" ]; then
    echo ""
    echo "No training progress found yet."
    echo "Start training with: ./start_training.sh"
    echo ""
    exit 0
fi

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║         TRAINING PROGRESS STATUS               ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# Display the progress file content
cat "$PROGRESS_FILE"

echo ""
echo "=============================================="
echo ""

# Count completed models
completed=$(grep -c "^COMPLETED:" "$PROGRESS_FILE" 2>/dev/null || echo "0")
total=5

echo "Progress: $completed / $total models completed"
echo ""

# Show which models are done
MODELS=("resnet50" "efficientnet" "densenet" "vit" "swin")

for model in "${MODELS[@]}"; do
    if grep -q "^COMPLETED: $model" "$PROGRESS_FILE" 2>/dev/null; then
        echo "  ✓ $model - COMPLETED"
    else
        echo "  ⧖ $model - PENDING"
    fi
done

echo ""
echo "=============================================="
echo ""
