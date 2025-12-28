#!/bin/bash
#
# 10-Fold Cross-Validation Training Script
# Trains all models on all folds (or specific model if specified)
#
# Usage:
#   ./scripts/run_10fold_cv.sh                    # Train all models
#   ./scripts/run_10fold_cv.sh --model swin       # Train only Swin
#   ./scripts/run_10fold_cv.sh --model densenet   # Train only DenseNet
#   ./scripts/run_10fold_cv.sh --model resnet50   # Train only ResNet50
#

set -e

# Configuration
EPOCHS=50
OUTPUT_DIR="models"
LOG_DIR="logs/cv_training"
METADATA_PATH="data/combined/master_metadata.csv"
IMAGES_ROOT="data/"
CLASS_WEIGHTS_PATH="data/combined/class_weights.csv"

# Parse arguments
MODEL_FILTER=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_FILTER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create log directory
mkdir -p "$LOG_DIR"

# Define models and their configurations
declare -A MODEL_CONFIG
MODEL_CONFIG["swin"]="--batch_size 16 --image_size 384"
MODEL_CONFIG["densenet"]="--batch_size 32 --image_size 224"
MODEL_CONFIG["resnet50"]="--batch_size 32 --image_size 224"

# Folds to train (excluding fold 5 which is already done)
FOLDS=(0 1 2 3 4 6 7 8 9)

# Function to check if fold is already trained
check_fold_exists() {
    local model=$1
    local fold=$2
    if ls "$OUTPUT_DIR"/${model}_fold${fold}_* 1> /dev/null 2>&1; then
        return 0  # exists
    else
        return 1  # doesn't exist
    fi
}

# Function to train a single fold
train_fold() {
    local model=$1
    local fold=$2
    local config=${MODEL_CONFIG[$model]}

    echo "=============================================="
    echo "Training: $model - Fold $fold"
    echo "Time: $(date)"
    echo "=============================================="

    # Check if already trained
    if check_fold_exists "$model" "$fold"; then
        echo "Fold $fold for $model already exists. Skipping..."
        return 0
    fi

    # Run training
    python3 train_combined_optimized.py \
        --metadata_path "$METADATA_PATH" \
        --images_root "$IMAGES_ROOT" \
        --class_weights_path "$CLASS_WEIGHTS_PATH" \
        --model "$model" \
        --fold "$fold" \
        --epochs "$EPOCHS" \
        $config \
        --lr 1e-4 \
        --use_amp \
        --use_focal_loss \
        --use_skin_tone_sampling \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee "$LOG_DIR/${model}_fold${fold}.log"

    echo "Completed: $model - Fold $fold at $(date)"
    echo ""
}

# Main training loop
echo "=============================================="
echo "10-FOLD CROSS-VALIDATION TRAINING"
echo "Started at: $(date)"
echo "=============================================="

# Determine which models to train
if [ -n "$MODEL_FILTER" ]; then
    MODELS=("$MODEL_FILTER")
else
    MODELS=("resnet50" "densenet" "swin")  # Fastest first
fi

# Count total folds
TOTAL_FOLDS=$((${#MODELS[@]} * ${#FOLDS[@]}))
COMPLETED=0

# Train each model
for model in "${MODELS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Starting training for: $model"
    echo "=============================================="

    for fold in "${FOLDS[@]}"; do
        train_fold "$model" "$fold"
        COMPLETED=$((COMPLETED + 1))
        echo "Progress: $COMPLETED / $TOTAL_FOLDS folds"
    done
done

echo ""
echo "=============================================="
echo "10-FOLD CV TRAINING COMPLETE"
echo "Finished at: $(date)"
echo "=============================================="

# Run aggregation
echo "Running results aggregation..."
python3 scripts/aggregate_cv_results.py --output_dir results/cv_results

echo "Done! Check results/cv_results/ for aggregated results."
