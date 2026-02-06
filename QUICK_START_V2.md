# Quick Start Guide - SBERT V2 Training

## TL;DR - Just Run This

```bash
# 1. Activate environment
venv\Scripts\activate

# 2. Start training (~2 hours)
venv\Scripts\python train_sbert_v2.py

# 3. Test when done
venv\Scripts\python inference_sbert_v2.py
```

---

## What You're Getting

### V2 Improvements:
- ✅ **Better Model:** all-mpnet-base-v2 (5x larger, more accurate)
- ✅ **More Training:** 5 epochs (was 2)
- ✅ **Larger Batches:** 32 (was 16)
- ✅ **Validation:** Tracks performance during training
- ✅ **Best Model:** Auto-saves best performing checkpoint

### Expected Results:
- **Training Time:** ~2 hours
- **Accuracy Gain:** +10-15% over V1
- **Higher Confidence:** More reliable predictions
- **Better Model:** 768-dim embeddings (vs 384)

---

## Files Created

1. **train_sbert_v2.py** - Improved training script
2. **inference_sbert_v2.py** - Inference with V2 model
3. **IMPLEMENTATION_GUIDE_V2.md** - Detailed step-by-step guide
4. **FUTURE_IMPROVEMENTS.md** - Advanced optimizations for later
5. **QUICK_START_V2.md** - This file!

---

## Training Progress

Watch for these milestones:

```
[  5%] Model downloaded (if first time)
[ 10%] Data loaded and split
[ 15%] Training pairs generated
[ 20%] Epoch 1 started
[ 40%] Epoch 2 started (validation improving)
[ 60%] Epoch 3 started
[ 80%] Epoch 4 started (best model saved)
[100%] Epoch 5 complete → Building FAISS index → Done!
```

---

## Output Files

After training completes:

```
finetuned_sbert_model_v2/    ← Your trained model
disease_faiss_v2.index        ← Vector index
disease_mapping_v2.json       ← Disease labels
training_config_v2.json       ← Training parameters
training_v2.log               ← Detailed logs
```

---

## Quick Test

After training:

```python
from inference_sbert_v2 import DiseaseDetectorV2

detector = DiseaseDetectorV2()
result = detector.predict("Apple leaves have black spots")
print(result)
# Expected: Apple Scab with 80%+ confidence
```

---

## Need Help?

1. **Training fails?** Check `training_v2.log`
2. **Out of memory?** Reduce `BATCH_SIZE` to 16
3. **Too slow?** Reduce `EPOCHS` to 3
4. **More details?** Read `IMPLEMENTATION_GUIDE_V2.md`
5. **What's next?** Read `FUTURE_IMPROVEMENTS.md`

---

## Ready? Let's Go!

```bash
venv\Scripts\python train_sbert_v2.py
```

**Grab some coffee ☕ - See you in ~2 hours!**
