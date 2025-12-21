#!/bin/bash
# Sequential model training script with progress tracking

# Define all models to train
MODELS=("resnet50" "efficientnet" "densenet" "vit" "swin")
DATA_PATH="data/ISIC2019/ISIC_2019_Training_Input"
CSV_PATH="data/ISIC2019/ISIC_2019_Training_GroundTruth.csv"
EPOCHS=50
BATCH_SIZE=256
LR=1e-4
IMAGE_SIZE=224

# Progress file to track completed models
PROGRESS_FILE="training_progress.txt"

# Create progress file if it doesn't exist
if [ ! -f "$PROGRESS_FILE" ]; then
    echo "# Training Progress - Started $(date)" > "$PROGRESS_FILE"
    echo "# Models: ${MODELS[@]}" >> "$PROGRESS_FILE"
    echo "" >> "$PROGRESS_FILE"
fi

# Function to check if a model is already trained
is_model_trained() {
    local model=$1
    grep -q "^COMPLETED: $model" "$PROGRESS_FILE" 2>/dev/null
    return $?
}

# Function to mark model as completed
mark_completed() {
    local model=$1
    echo "COMPLETED: $model - $(date)" >> "$PROGRESS_FILE"
}

# Function to display progress
show_progress() {
    echo ""
    echo "=============================================="
    echo "        TRAINING PROGRESS STATUS"
    echo "=============================================="
    echo ""

    local completed=0
    local pending=0

    for model in "${MODELS[@]}"; do
        if is_model_trained "$model"; then
            echo "✓ $model - COMPLETED"
            ((completed++))
        else
            echo "⧖ $model - PENDING"
            ((pending++))
        fi
    done

    echo ""
    echo "Total: ${#MODELS[@]} | Completed: $completed | Remaining: $pending"
    echo "=============================================="
    echo ""
}

# Main training loop
echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║   Sequential Model Training - ISIC2019        ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# Show initial progress
show_progress

# Train each model sequentially
for model in "${MODELS[@]}"; do

    # Check if already trained
    if is_model_trained "$model"; then
        echo "⏭  Skipping $model (already trained)"
        echo ""
        continue
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  NOW TRAINING: $model"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Start time: $(date)"
    echo ""

    # Run training
    python3 train_with_logging.py \
        --model "$model" \
        --data_path "$DATA_PATH" \
        --csv_path "$CSV_PATH" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --image_size $IMAGE_SIZE

    # Check if training succeeded
    if [ $? -eq 0 ]; then
        mark_completed "$model"
        echo ""
        echo "✅ $model training completed successfully!"
        echo "  End time: $(date)"
        echo ""
        show_progress
    else
        echo ""
        echo "❌ $model training failed!"
        echo "  Failed at: $(date)"
        echo ""
        echo "You can resume training by running this script again."
        echo "Already completed models will be skipped."
        exit 1
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Small pause between models
    sleep 2
done

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║         ALL MODELS TRAINED SUCCESSFULLY!       ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "Training completed at: $(date)"
echo ""
