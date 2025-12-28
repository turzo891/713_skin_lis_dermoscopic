#!/bin/bash
#
# Check 10-Fold Cross-Validation Progress
#
# Shows which folds have been completed for each model
#

echo "=============================================="
echo "10-FOLD CV TRAINING PROGRESS"
echo "=============================================="
echo ""

MODELS=("swin" "densenet" "resnet50")
FOLDS=(0 1 2 3 4 5 6 7 8 9)

for model in "${MODELS[@]}"; do
    echo "=== $model ==="

    completed=0
    missing=""

    for fold in "${FOLDS[@]}"; do
        if ls models/${model}_fold${fold}_* 1> /dev/null 2>&1; then
            # Check if training completed (has final_results.json or best_model.pth)
            dir=$(ls -d models/${model}_fold${fold}_* 2>/dev/null | head -1)
            if [ -f "$dir/best_model.pth" ] || [ -f "$dir/final_results.json" ]; then
                echo "  Fold $fold: ✓ Complete"
                completed=$((completed + 1))
            else
                echo "  Fold $fold: ⏳ In Progress"
            fi
        else
            echo "  Fold $fold: ✗ Not Started"
            missing="$missing $fold"
        fi
    done

    echo ""
    echo "  Summary: $completed/10 folds complete"
    if [ -n "$missing" ]; then
        echo "  Missing folds:$missing"
    fi
    echo ""
done

# Overall progress
echo "=============================================="
echo "OVERALL PROGRESS"
echo "=============================================="

total_complete=$(ls -d models/*_fold*/ 2>/dev/null | wc -l)
total_expected=30  # 3 models x 10 folds

echo "Total: $total_complete / $total_expected models trained"

# Estimate remaining time
remaining=$((total_expected - total_complete))
if [ $remaining -gt 0 ]; then
    # Average ~1.5 hours per model
    hours=$((remaining * 3 / 2))
    echo "Estimated remaining time: ~${hours} hours"
fi

echo ""
echo "=============================================="
