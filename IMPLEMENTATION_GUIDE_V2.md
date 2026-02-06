# SBERT V2 Implementation Guide
## Step-by-Step Instructions to Train Improved Model

**Estimated Time:** ~2 hours
**Date:** 2026-02-05

---

## What's New in V2?

### Improvements Applied:
✅ **Better Model:** Upgraded from `all-MiniLM-L6-v2` (22M) → `all-mpnet-base-v2` (110M params)
✅ **More Training:** Increased from 2 → 5 epochs
✅ **Larger Batches:** Increased from 16 → 32 batch size
✅ **Higher Learning Rate:** 2e-5 → 5e-5
✅ **Validation Split:** Added 70/15/15 train/val/test split
✅ **Validation Evaluator:** Track performance during training
✅ **Best Model Saving:** Automatically saves best performing model
✅ **Better Logging:** Detailed logs saved to `training_v2.log`

### Expected Results:
- **Accuracy:** 10-20% improvement over v1
- **Better generalization:** Validation evaluator ensures no overfitting
- **Larger embeddings:** 768 dimensions (vs 384 in v1) = richer representations

---

## Prerequisites

### 1. Check Your Environment
```bash
# Activate virtual environment
venv\Scripts\activate

# Verify all packages installed
python -c "import sentence_transformers, faiss, torch; print('All packages OK')"
```

### 2. Backup V1 Model (Optional but Recommended)
```bash
# Rename your current v1 model to keep it
ren finetuned_sbert_model finetuned_sbert_model_v1_backup
ren disease_faiss.index disease_faiss_v1_backup.index
ren disease_mapping.json disease_mapping_v1_backup.json
```

---

## Step-by-Step Implementation

### Step 1: Review the Training Script
```bash
# Open and review the new training script
notepad train_sbert_v2.py
```

**Key Parameters to Note:**
```python
EPOCHS = 5                      # More epochs
BATCH_SIZE = 32                 # Larger batch
MODEL_NAME = "all-mpnet-base-v2"  # Better model
SAMPLES_PER_CLASS = 100         # Same as v1
```

### Step 2: Start Training
```bash
# Run the improved training script
venv\Scripts\python train_sbert_v2.py
```

**What Happens:**
1. ✅ Loads dataset (27 categories, 7 crops)
2. ✅ Splits data 70/15/15 (train/val/test)
3. ✅ Generates ~2,700 training pairs
4. ✅ Generates ~810 validation pairs
5. ✅ Downloads all-mpnet-base-v2 model (first time only)
6. ✅ Trains for 5 epochs with validation
7. ✅ Saves best model to `finetuned_sbert_model_v2/`
8. ✅ Builds FAISS index
9. ✅ Saves training config

**Expected Console Output:**
```
============================================================
SBERT V2 Training with Improved Hyperparameters
============================================================
Model: sentence-transformers/all-mpnet-base-v2
Epochs: 5
Batch Size: 32
Learning Rate: 5e-05
Samples per Class: 100
Estimated Time: ~2 hours
============================================================
Loading dataset from PlantVillage_train_prompts - Copy.json...
Loaded 38 categories.
Splitting data 70/15/15 (train/val/test) for selected crops...
...
```

### Step 3: Monitor Training Progress
Watch for these key indicators:

**Epoch Progress:**
```
Epoch: 1/5
  0%|          | 1/85 [00:20<28:20, 20.25s/it]
 12%|█▏        | 10/85 [03:25<25:15, 20.20s/it]
```

**Validation Scores (Every 100 steps):**
```
Evaluation on disease-validation dataset:
Cosine-Similarity: Pearson: 0.8234 Spearman: 0.8156
```

**Look For:**
- ✅ Validation scores should improve over time
- ✅ Training loss should decrease
- ✅ No errors or warnings

### Step 4: Check Training Logs
```bash
# View detailed logs
notepad training_v2.log

# Or check last 50 lines
tail -n 50 training_v2.log
```

### Step 5: Verify Outputs Created
```bash
# Check that all files were created
dir finetuned_sbert_model_v2
dir disease_faiss_v2.index
dir disease_mapping_v2.json
dir training_config_v2.json
```

**Expected Files:**
```
finetuned_sbert_model_v2/
  ├── config.json
  ├── model.safetensors
  ├── modules.json
  ├── sentence_bert_config.json
  └── tokenizer files...

disease_faiss_v2.index          (~3-4 MB)
disease_mapping_v2.json         (~100 KB)
training_config_v2.json         (~1 KB)
training_v2.log                 (variable)
```

### Step 6: Test the V2 Model
```bash
# Run inference test with new model
venv\Scripts\python inference_sbert_v2.py
```

**Expected Output:**
```
Loading SBERT V2 model...
Loading FAISS index...
Loading disease mapping...

==================================================
RUNNING INFERENCE TEST - V2 MODEL
==================================================

Query: Apple leaves have large black spots and are curling
{
  "query": "Apple leaves have large black spots and are curling",
  "top_predictions": [
    {
      "rank": 1,
      "disease": "Apple Scab",
      "confidence": 0.78    <-- Higher than v1!
    },
    ...
  ],
  "key_symptoms_identified": ["spots", "black", "curling"]
}
```

### Step 7: Compare V1 vs V2 Performance

**Quick Comparison Test:**
```bash
# Test same query on both models
echo "Testing V1..."
venv\Scripts\python inference_sbert.py

echo "Testing V2..."
venv\Scripts\python inference_sbert_v2.py
```

**Expected Improvements:**
- ✅ V2 confidence scores 5-15% higher
- ✅ Better ranking of correct disease
- ✅ More accurate top predictions

---

## Troubleshooting

### Issue 1: Out of Memory Error
**Symptoms:**
```
RuntimeError: CUDA out of memory
```
**Solution:**
```python
# Reduce batch size in train_sbert_v2.py
BATCH_SIZE = 16  # Instead of 32
```

### Issue 2: Training Too Slow
**Symptoms:** Taking more than 3 hours

**Solution:**
```python
# Reduce epochs in train_sbert_v2.py
EPOCHS = 3  # Instead of 5
```

### Issue 3: Model Download Fails
**Symptoms:**
```
ConnectionError: Failed to download model
```
**Solution:**
```bash
# Pre-download the model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"
```

### Issue 4: Validation Scores Not Improving
**Symptoms:** Validation score stays flat or decreases

**Possible Causes:**
- Learning rate too high
- Overfitting
- Data quality issues

**Solution:**
```python
# Lower learning rate
LEARNING_RATE = 2e-5  # Instead of 5e-5
```

---

## Performance Benchmarks

### V1 Model (Baseline):
```
Model: all-MiniLM-L6-v2
Epochs: 2
Training Time: 19 minutes
Embedding Dim: 384
Training Loss: 0.5186
```

### V2 Model (Expected):
```
Model: all-mpnet-base-v2
Epochs: 5
Training Time: ~2 hours
Embedding Dim: 768
Training Loss: 0.35-0.45 (lower is better)
Validation Score: 0.75-0.85 (higher is better)
```

**Expected Improvements:**
- Apple Scab: 74% → **82-85%** confidence
- Corn Gray Leaf Spot: 86% → **90-93%** confidence
- Overall accuracy: **+10-15%**

---

## Post-Training Validation

### Test with New Queries

**Create a test file:**
```python
# test_v2_accuracy.py
test_cases = [
    {
        "query": "Round brown spots with yellow halos on tomato leaves",
        "expected": "Tomato Early Blight"
    },
    {
        "query": "White powdery coating on grape leaf surface",
        "expected": "Grape Esca (Black Measles)"  # or similar
    },
    {
        "query": "Long grayish lesions on corn leaves",
        "expected": "Corn Gray Leaf Spot"
    }
]

from inference_sbert_v2 import DiseaseDetectorV2

detector = DiseaseDetectorV2()
correct = 0

for test in test_cases:
    preds = detector.predict(test["query"])
    if preds and test["expected"] in preds[0]["disease"]:
        correct += 1
        print(f"✓ {test['query'][:40]}... -> {preds[0]['disease']}")
    else:
        print(f"✗ {test['query'][:40]}... -> {preds[0]['disease'] if preds else 'None'}")

print(f"\nAccuracy: {correct}/{len(test_cases)} = {100*correct/len(test_cases):.1f}%")
```

---

## Next Steps After V2 Training

### 1. Deploy V2 Model
```bash
# Update your inference script to use v2 by default
cp inference_sbert_v2.py inference_sbert.py

# Update model paths
MODEL_PATH = r"finetuned_sbert_model_v2"
FAISS_INDEX_PATH = r"disease_faiss_v2.index"
DISEASE_MAPPING_PATH = r"disease_mapping_v2.json"
```

### 2. Compare Models Side-by-Side
```bash
# Keep both models and test
python compare_models.py  # (create this script if needed)
```

### 3. Document Performance Gains
```bash
# Save your results
notepad v2_performance_results.txt
```

### 4. Read Future Improvements Guide
```bash
# Check the improvements document for next steps
notepad FUTURE_IMPROVEMENTS.md
```

---

## Quick Reference Commands

```bash
# Activate environment
venv\Scripts\activate

# Train V2 model (~2 hours)
venv\Scripts\python train_sbert_v2.py

# Test V2 model
venv\Scripts\python inference_sbert_v2.py

# View logs
notepad training_v2.log

# Check config
notepad training_config_v2.json
```

---

## Success Checklist

- [ ] Virtual environment activated
- [ ] V1 model backed up (optional)
- [ ] Started training (`train_sbert_v2.py`)
- [ ] Training completed without errors
- [ ] All output files created
- [ ] Inference test passed
- [ ] Performance improved over V1
- [ ] Documented results
- [ ] Ready for next improvements

---

## Support & Questions

**If training fails:**
1. Check `training_v2.log` for errors
2. Verify dataset path is correct
3. Ensure enough disk space (~5 GB free)
4. Check internet connection (for model download)

**If results are worse than expected:**
1. Verify data quality in dataset
2. Check validation scores in logs
3. Try reducing learning rate
4. Consider data augmentation (see FUTURE_IMPROVEMENTS.md)

---

**Good luck with your improved SBERT training!** 🚀
