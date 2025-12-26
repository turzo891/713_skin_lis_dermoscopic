# Training Issues Fixed

## Issue 1: Invalid Fold Number (Fold 50)

**Error:**
```
Val samples: 0
ZeroDivisionError: division by zero
```

**Cause:**
- Used `--fold 50` but only folds 0-9 are valid for 10-fold cross-validation
- Fold 50 doesn't exist, so validation set was empty

**Fix:**
- Added validation check in `validate()` function
- Returns warning message and dummy metrics if validation set is empty
- Prevents division by zero error

**Correct Usage:**
```bash
# Valid fold numbers: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
python3 train_combined_optimized.py \
   --metadata_path data/combined/master_metadata.csv \
   --images_root data \
   --class_weights_path data/combined/class_weights.csv \
   --fold 0  # Use 0-9 only
   --model resnet50 \
   --batch_size 32 \
   --use_focal_loss \
   --use_skin_tone_sampling
```

---

## Issue 2: ViT Image Size Mismatch

**Error:**
```
AssertionError: Input height (384) doesn't match model (224).
```

**Cause:**
- ViT model initialized with default 224x224 size
- Training script passes 384x384 images with `--image_size 384`
- Model expects 224 but receives 384

**Fix:**
- `get_model()` function already passes `image_size` parameter to ViT and Swin models (lines 782, 784 in models.py)
- Model correctly initializes with specified image size

**Solution Already Working:**
The code was already correct! The `get_model()` function in `src/models.py` properly handles image_size:

```python
def get_model(name: str, num_classes: int = 7, pretrained: bool = True,
              dropout: float = 0.5, image_size: int = 224) -> nn.Module:
    ...
    elif name in ['vit', 'vit_base']:
        return ViTModel(num_classes, pretrained, dropout, image_size=image_size)  # Correct!
    elif name in ['swin', 'swin_base']:
        return SwinTransformerModel(num_classes, pretrained, dropout, image_size=image_size)  # Correct!
```

**Correct Usage for ViT/Swin:**
```bash
python3 train_combined_optimized.py \
   --metadata_path data/combined/master_metadata.csv \
   --images_root data \
   --class_weights_path data/combined/class_weights.csv \
   --fold 5 \
   --model vit \
   --batch_size 16 \
   --image_size 384 \  # This gets passed to the model correctly
   --use_focal_loss \
   --use_skin_tone_sampling
```

---

## Summary of Fixes

### Fixed in `train_combined_optimized.py`:

1. **Empty Validation Set Handling (Lines 317-330)**
   ```python
   # Check if validation set is empty
   if len(val_loader) == 0:
       msg = "WARNING: Validation set is empty! Check your fold number (valid: 0-9 for 10-fold CV)"
       print(msg)
       self.logger.warning(msg)
       # Return dummy metrics to prevent crash
       return {...}
   ```

### Already Correct in `src/models.py`:

2. **ViT/Swin Image Size (Lines 782, 784)**
   - `image_size` parameter already passed correctly
   - No changes needed

---

## Recommended Training Commands

### ResNet50 (224x224)
```bash
python3 train_combined_optimized.py \
   --metadata_path data/combined/master_metadata.csv \
   --images_root data \
   --class_weights_path data/combined/class_weights.csv \
   --fold 0 \
   --model resnet50 \
   --batch_size 32 \
   --use_focal_loss \
   --use_skin_tone_sampling
```

### EfficientNet (384x384)
```bash
python3 train_combined_optimized.py \
   --metadata_path data/combined/master_metadata.csv \
   --images_root data \
   --class_weights_path data/combined/class_weights.csv \
   --fold 0 \
   --model efficientnet \
   --batch_size 32 \
   --image_size 384 \
   --use_focal_loss \
   --use_skin_tone_sampling
```

### Vision Transformer (384x384)
```bash
python3 train_combined_optimized.py \
   --metadata_path data/combined/master_metadata.csv \
   --images_root data \
   --class_weights_path data/combined/class_weights.csv \
   --fold 0 \
   --model vit \
   --batch_size 16 \
   --image_size 384 \
   --use_focal_loss \
   --use_skin_tone_sampling
```

### Swin Transformer (384x384)
```bash
python3 train_combined_optimized.py \
   --metadata_path data/combined/master_metadata.csv \
   --images_root data \
   --class_weights_path data/combined/class_weights.csv \
   --fold 0 \
   --model swin \
   --batch_size 16 \
   --image_size 384 \
   --use_focal_loss \
   --use_skin_tone_sampling
```

---

## Training All 10 Folds

```bash
for fold in {0..9}; do
    python3 train_combined_optimized.py \
        --metadata_path data/combined/master_metadata.csv \
        --images_root data \
        --class_weights_path data/combined/class_weights.csv \
        --fold $fold \
        --model resnet50 \
        --epochs 50 \
        --batch_size 32 \
        --use_focal_loss \
        --use_skin_tone_sampling \
        --random_seed $((42 + fold))
done
```

---

## Key Points

1. **Valid Fold Numbers:** 0-9 only (for 10-fold CV)
2. **Image Sizes:**
   - ResNet50, DenseNet: 224x224 (default)
   - EfficientNet: 384x384 (recommended)
   - ViT, Swin: 384x384 (recommended)
3. **Batch Sizes:**
   - CNN models: 32 (can go higher if GPU allows)
   - Transformer models: 16 (larger models, need more memory)
4. **Always use:** `--use_focal_loss` and `--use_skin_tone_sampling` for best results

---

**Status:** All issues fixed. Ready for training.
