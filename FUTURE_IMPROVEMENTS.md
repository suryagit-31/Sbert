# Future Improvements for SBERT Plant Disease Detection
## Advanced Optimizations Beyond V2

**Last Updated:** 2026-02-05
**Current Version:** V2 (all-mpnet-base-v2, 5 epochs)

This document outlines additional improvements you can make to further boost accuracy and performance.

---

## Quick Wins ✅ (Already Implemented in V2)

- [x] Increased epochs (2 → 5)
- [x] Larger batch size (16 → 32)
- [x] Better base model (MiniLM → MPNet)
- [x] Validation split and evaluator
- [x] Best model saving

---

## Medium Effort Improvements (1-3 Days)

### 1. Data Augmentation Pipeline
**Time:** 4-6 hours | **Impact:** +5-10% accuracy

Generate more diverse training descriptions using paraphrasing.

**Implementation:**

```python
# augment_data.py
import json
from transformers import pipeline

# Use a paraphrase model
paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

def augment_description(text):
    """Generate 3-5 paraphrases of a disease description"""
    paraphrases = []
    for i in range(3):
        result = paraphraser(f"paraphrase: {text}",
                            max_length=100,
                            num_return_sequences=1)
        paraphrases.append(result[0]['generated_text'])
    return paraphrases

def augment_dataset(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    augmented = {}
    for disease, descriptions in data.items():
        augmented[disease] = descriptions.copy()

        # Augment each description
        for desc in descriptions[:20]:  # Augment first 20
            new_descs = augment_description(desc)
            augmented[disease].extend(new_descs)

    with open(output_path, 'w') as f:
        json.dump(augmented, f, indent=2)

    print(f"Original: {sum(len(v) for v in data.values())} descriptions")
    print(f"Augmented: {sum(len(v) for v in augmented.values())} descriptions")

# Usage:
augment_dataset(
    "PlantVillage_train_prompts - Copy.json",
    "PlantVillage_train_prompts_augmented.json"
)
```

**Expected Results:**
- 3-4x more training data
- Better generalization
- 5-10% accuracy improvement

---

### 2. Hard Negative Mining
**Time:** 3-4 hours | **Impact:** +3-7% accuracy

Train on commonly confused disease pairs to improve discrimination.

**Implementation:**

```python
# hard_negatives.py
import json
from sentence_transformers import InputExample

# Define commonly confused pairs
HARD_NEGATIVE_PAIRS = [
    ("Tomato Early Blight", "Tomato Late Blight"),
    ("Apple Scab", "Apple Black Rot"),
    ("Corn Common Rust", "Corn Northern Leaf Blight"),
    ("Grape Esca (Black Measles)", "Grape Black Rot"),
    ("Potato Early Blight", "Potato Late Blight"),
]

def generate_hard_negative_pairs(data, hard_pairs):
    """Generate training pairs that distinguish similar diseases"""
    hard_examples = []

    for disease_a, disease_b in hard_pairs:
        if disease_a in data and disease_b in data:
            texts_a = data[disease_a]
            texts_b = data[disease_b]

            # Create pairs: (same disease) = 1.0, (different disease) = 0.0
            for i, text_a in enumerate(texts_a[:10]):
                # Positive pair (same disease)
                if i+1 < len(texts_a):
                    hard_examples.append(
                        InputExample(texts=[text_a, texts_a[i+1]], label=1.0)
                    )

                # Hard negative pair (similar but different disease)
                for text_b in texts_b[:5]:
                    hard_examples.append(
                        InputExample(texts=[text_a, text_b], label=0.0)
                    )

    return hard_examples

# Add to train_sbert_v2.py:
# hard_negatives = generate_hard_negative_pairs(train_data, HARD_NEGATIVE_PAIRS)
# train_examples.extend(hard_negatives)
```

**Expected Results:**
- Better distinction between similar diseases
- Fewer false positives
- 3-7% accuracy improvement

---

### 3. Ensemble Model System
**Time:** 2-3 hours | **Impact:** +5-8% accuracy

Combine predictions from multiple models for robust results.

**Implementation:**

```python
# ensemble_detector.py
from sentence_transformers import SentenceTransformer
import numpy as np

class EnsembleDetector:
    def __init__(self):
        # Load multiple models
        self.models = [
            SentenceTransformer("finetuned_sbert_model_v1"),
            SentenceTransformer("finetuned_sbert_model_v2"),
            # Could add more variants
        ]

        # Load FAISS indexes for each
        # (implement similarly to DiseaseDetector)

    def predict(self, query):
        """Average predictions from all models"""
        all_predictions = []

        for model in self.models:
            preds = self._predict_single(model, query)
            all_predictions.append(preds)

        # Aggregate by averaging confidence scores
        combined = {}
        for preds in all_predictions:
            for p in preds:
                disease = p['disease']
                if disease not in combined:
                    combined[disease] = []
                combined[disease].append(p['confidence'])

        # Average and sort
        results = [
            {
                'disease': disease,
                'confidence': np.mean(scores),
                'std': np.std(scores)  # Uncertainty measure
            }
            for disease, scores in combined.items()
        ]

        return sorted(results, key=lambda x: x['confidence'], reverse=True)
```

**Expected Results:**
- More stable predictions
- Lower variance
- 5-8% accuracy improvement

---

### 4. Different Loss Functions
**Time:** 1-2 hours | **Impact:** +2-5% accuracy

Experiment with alternative loss functions.

**Implementation:**

```python
# In train_sbert_v2.py, replace MultipleNegativesRankingLoss with:

# Option 1: CoSENT Loss (better for similarity)
from sentence_transformers.losses import CoSENTLoss
train_loss = CoSENTLoss(model)

# Option 2: Contrastive Loss
from sentence_transformers.losses import ContrastiveLoss
train_loss = ContrastiveLoss(model)

# Option 3: Triplet Loss (requires triplets)
from sentence_transformers.losses import TripletLoss
train_loss = TripletLoss(model)
```

**Expected Results:**
- Better embedding quality
- Improved similarity scoring
- 2-5% accuracy improvement

---

### 5. Balanced Class Sampling
**Time:** 2-3 hours | **Impact:** +3-5% accuracy

Ensure all diseases get equal representation during training.

**Implementation:**

```python
# balanced_sampling.py
from torch.utils.data import WeightedRandomSampler

def create_balanced_sampler(train_examples, labels):
    """Create sampler that balances rare and common diseases"""

    # Count samples per disease
    disease_counts = {}
    for example, label in zip(train_examples, labels):
        if label not in disease_counts:
            disease_counts[label] = 0
        disease_counts[label] += 1

    # Calculate weights (inverse frequency)
    weights = []
    for label in labels:
        weight = 1.0 / disease_counts[label]
        weights.append(weight)

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    return sampler

# Use in DataLoader:
# sampler = create_balanced_sampler(train_examples, labels)
# train_dataloader = DataLoader(train_examples, batch_size=32, sampler=sampler)
```

**Expected Results:**
- Better performance on rare diseases
- More balanced predictions
- 3-5% accuracy improvement

---

## Advanced Improvements (3-7 Days)

### 6. Multi-Task Learning
**Time:** 1-2 days | **Impact:** +8-12% accuracy

Train on multiple related tasks simultaneously.

**Approach:**
- Task 1: Disease classification
- Task 2: Symptom extraction
- Task 3: Severity prediction
- Task 4: Crop identification

**Benefits:**
- Richer representations
- Better feature learning
- Improved generalization

---

### 7. Cross-Encoder Re-ranking
**Time:** 1 day | **Impact:** +5-10% accuracy

Use a cross-encoder to re-rank top predictions.

**Pipeline:**
1. Bi-encoder (SBERT) retrieves top 10 candidates (fast)
2. Cross-encoder re-ranks top 10 (accurate but slow)

**Implementation:**
```python
from sentence_transformers import CrossEncoder

# Train a cross-encoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Re-rank top predictions
def rerank_predictions(query, candidates):
    pairs = [[query, c['disease']] for c in candidates]
    scores = cross_encoder.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate['rerank_score'] = score

    return sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
```

---

### 8. Domain-Specific Pre-training
**Time:** 2-3 days | **Impact:** +10-15% accuracy

Pre-train on general plant science text before fine-tuning.

**Data Sources:**
- Plant pathology textbooks
- Agricultural research papers
- Plant disease databases
- Gardening forums

**Approach:**
1. Collect domain text corpus (100K+ sentences)
2. Pre-train with MLM (Masked Language Modeling)
3. Fine-tune on disease detection task

---

### 9. Active Learning Loop
**Time:** 3-5 days | **Impact:** Continuous improvement

Iteratively improve model with user feedback.

**Pipeline:**
```
1. Model makes prediction
2. User confirms/corrects
3. Add to training set
4. Periodically retrain
5. Deploy updated model
```

**Implementation:**
```python
# active_learning.py
class ActiveLearningPipeline:
    def __init__(self):
        self.uncertain_samples = []
        self.feedback_db = []

    def predict_with_uncertainty(self, query):
        preds = self.model.predict(query)

        # Flag uncertain predictions
        if preds[0]['confidence'] < 0.7:
            self.uncertain_samples.append({
                'query': query,
                'predictions': preds,
                'timestamp': datetime.now()
            })

        return preds

    def collect_feedback(self, query, predicted, actual):
        """Store user corrections"""
        self.feedback_db.append({
            'query': query,
            'predicted': predicted,
            'actual': actual,
            'timestamp': datetime.now()
        })

    def retrain_with_feedback(self):
        """Retrain model on collected feedback"""
        if len(self.feedback_db) > 100:
            # Add feedback to training data
            # Retrain model
            # Deploy new version
            pass
```

---

### 10. Image + Text Multi-Modal
**Time:** 5-7 days | **Impact:** +15-25% accuracy

Combine text descriptions with plant images.

**Architecture:**
```
Image Branch: ResNet/EfficientNet → Image Embeddings
Text Branch: SBERT → Text Embeddings
Fusion: Concatenate or Cross-Attention
Output: Combined Disease Prediction
```

**Benefits:**
- Visual + textual information
- More robust predictions
- Better accuracy

---

## Evaluation & Monitoring

### 1. Create Comprehensive Test Suite
```python
# test_suite.py
import json

class ModelEvaluator:
    def __init__(self, test_data_path):
        with open(test_data_path) as f:
            self.test_cases = json.load(f)

    def evaluate_model(self, detector):
        metrics = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'per_disease': {}
        }

        correct = 0
        total = 0

        for disease, descriptions in self.test_cases.items():
            disease_correct = 0

            for desc in descriptions:
                preds = detector.predict(desc)

                if preds and preds[0]['disease'] == disease:
                    disease_correct += 1
                    correct += 1

                total += 1

            metrics['per_disease'][disease] = {
                'accuracy': disease_correct / len(descriptions),
                'samples': len(descriptions)
            }

        metrics['accuracy'] = correct / total
        return metrics

# Usage:
evaluator = ModelEvaluator('test_data.json')
results_v1 = evaluator.evaluate_model(detector_v1)
results_v2 = evaluator.evaluate_model(detector_v2)

print(f"V1 Accuracy: {results_v1['accuracy']:.2%}")
print(f"V2 Accuracy: {results_v2['accuracy']:.2%}")
print(f"Improvement: {(results_v2['accuracy'] - results_v1['accuracy']):.2%}")
```

---

### 2. A/B Testing Framework
```python
# ab_testing.py
import random

class ABTestingFramework:
    def __init__(self, model_a, model_b):
        self.model_a = model_a
        self.model_b = model_b
        self.results = {'a': [], 'b': []}

    def predict_with_tracking(self, query):
        # Randomly assign to A or B
        variant = 'a' if random.random() < 0.5 else 'b'

        model = self.model_a if variant == 'a' else self.model_b
        prediction = model.predict(query)

        self.results[variant].append({
            'query': query,
            'prediction': prediction,
            'timestamp': datetime.now()
        })

        return prediction

    def analyze_results(self):
        """Compare performance of A vs B"""
        # Implement statistical significance testing
        pass
```

---

## Infrastructure Improvements

### 1. Model Versioning
```bash
# Use Git LFS for model versioning
git lfs install
git lfs track "*.safetensors"
git lfs track "*.index"

# Track model versions
git commit -am "v1.0: Initial model"
git tag v1.0

git commit -am "v2.0: MPNet with 5 epochs"
git tag v2.0
```

### 2. Model Registry
```python
# model_registry.py
import json
from pathlib import Path

class ModelRegistry:
    def __init__(self, registry_path="model_registry.json"):
        self.registry_path = registry_path
        self.models = self._load_registry()

    def register_model(self, version, path, metrics):
        """Register a new model version"""
        self.models[version] = {
            'path': path,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'status': 'deployed'
        }
        self._save_registry()

    def get_best_model(self, metric='accuracy'):
        """Get model with best performance"""
        best = max(
            self.models.items(),
            key=lambda x: x[1]['metrics'].get(metric, 0)
        )
        return best

    def _load_registry(self):
        if Path(self.registry_path).exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        with open(self.registry_path, 'w') as f:
            json.dump(self.models, f, indent=2)
```

---

## Implementation Priority

### Phase 1 (Next Week):
1. ✅ Data Augmentation
2. ✅ Hard Negative Mining
3. ✅ Comprehensive Test Suite

### Phase 2 (Next Month):
4. ✅ Ensemble System
5. ✅ Balanced Sampling
6. ✅ Alternative Loss Functions

### Phase 3 (Next Quarter):
7. ✅ Cross-Encoder Re-ranking
8. ✅ Active Learning Loop
9. ✅ Multi-Task Learning

### Phase 4 (Long-term):
10. ✅ Domain Pre-training
11. ✅ Multi-Modal (Image + Text)
12. ✅ Production Infrastructure

---

## Resources & References

### Papers:
- **SBERT:** "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- **Hard Negatives:** "In Defense of Hard Negative Mining for Dense Retrieval"
- **Multi-Modal:** "CLIP: Learning Transferable Visual Models From Natural Language Supervision"

### Datasets:
- PlantVillage Dataset
- PlantDoc Dataset
- Plant Pathology 2020/2021 Challenge

### Tools:
- Hugging Face Transformers
- Sentence-Transformers
- FAISS (Facebook AI Similarity Search)
- MLflow (experiment tracking)
- Weights & Biases (monitoring)

---

## Next Steps Checklist

After completing V2 training:

- [ ] Evaluate V2 performance
- [ ] Compare with V1 baseline
- [ ] Document accuracy improvements
- [ ] Choose next improvement from Phase 1
- [ ] Set up experiment tracking
- [ ] Create test dataset
- [ ] Implement selected improvement
- [ ] A/B test results
- [ ] Deploy if better
- [ ] Repeat!

---

**Remember:** Incremental improvements compound over time. Each 3-5% gain adds up to significant overall performance boost!

**Good luck with your improvements!** 🚀
