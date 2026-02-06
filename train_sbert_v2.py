import json
import random
import os
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import logging

# ==============================
# LOGGING SETUP
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================
# CONFIGURATION - IMPROVED HYPERPARAMETERS
# ==============================
# Quick Wins Applied - FAST Training (Option 3):
# 1. Increased EPOCHS from 2 to 4 for better convergence
# 2. Increased BATCH_SIZE from 16 to 32 for efficiency
# 3. Increased LEARNING_RATE for faster convergence
# 4. Added validation split and evaluator
# 5. Keep same model for speed (all-MiniLM-L6-v2)
# Estimated training time: ~45 minutes

EPOCHS = 4                      # ↑ Increased from 2
BATCH_SIZE = 32                 # ↑ Increased from 16
LEARNING_RATE = 5e-5            # ↑ Increased from 2e-5
WARMUP_STEPS_RATIO = 0.1
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Same as v1 (for speed)
SAMPLES_PER_CLASS = 100         # Same as v1
EVALUATION_STEPS = 100          # Evaluate every 100 steps

dataset_path = r"PlantVillage_train_prompts - Copy.json"
model_save_path = r"finetuned_sbert_model_v2"
faiss_index_path = r"disease_faiss_v2.index"
disease_mapping_path = r"disease_mapping_v2.json"

# ==============================
# LOAD DATASET
# ==============================
def load_dataset(path):
    logger.info(f"Loading dataset from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} categories.")
    return data

# ==============================
# PREPARE DATA WITH VALIDATION SPLIT
# ==============================
def prepare_data_with_validation(data):
    """
    Improved data split: 70% train, 15% validation, 15% test
    Ensures balanced representation across all splits
    """
    train_data = {}
    val_data = {}
    test_data = {}

    # Define selected crops - 7 major crops
    SELECTED_CROPS = ['tomato', 'apple', 'corn', 'maize', 'potato', 'grape', 'soybean', 'orange']

    logger.info("Splitting data 70/15/15 (train/val/test) for selected crops...")

    stats = {"total_categories": 0, "total_samples": 0}

    for category, prompts in data.items():
        # Filter: only include selected crops
        if not any(crop in category.lower() for crop in SELECTED_CROPS):
            continue

        prompts = [p.strip() for p in prompts if len(p.strip()) > 0]

        # Remove duplicates within category
        prompts = list(set(prompts))

        stats["total_categories"] += 1
        stats["total_samples"] += len(prompts)

        if len(prompts) < 7:
            logger.warning(f"Category '{category}' has very few samples ({len(prompts)}). Using all for training.")
            train_data[category] = prompts
            val_data[category] = []
            test_data[category] = []
            continue

        # Shuffle for random split
        random.shuffle(prompts)

        # 70/15/15 split
        n = len(prompts)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        train_data[category] = prompts[:train_end]
        val_data[category] = prompts[train_end:val_end]
        test_data[category] = prompts[val_end:]

    logger.info(f"Data split complete: {stats['total_categories']} categories, {stats['total_samples']} total samples")
    logger.info(f"Train samples: {sum(len(v) for v in train_data.values())}")
    logger.info(f"Val samples: {sum(len(v) for v in val_data.values())}")
    logger.info(f"Test samples: {sum(len(v) for v in test_data.values())}")

    return train_data, val_data, test_data

# ==============================
# GENERATE TRAINING PAIRS
# ==============================
def generate_training_pairs(dataset, samples_per_class=100):
    pairs = []
    logger.info(f"Generating training pairs (target: {samples_per_class} per class)...")

    for category in tqdm(dataset, desc="Generating pairs"):
        texts = dataset[category]
        if len(texts) < 2:
            continue

        # Generate positive pairs (contrastive learning)
        # Map different descriptions of same disease close together

        possible_pairs = len(texts) * (len(texts) - 1) // 2
        num_pairs = min(samples_per_class, possible_pairs)

        count = 0
        attempts = 0
        max_attempts = num_pairs * 5
        seen_pairs = set()

        while count < num_pairs and attempts < max_attempts:
            attempts += 1
            a, b = random.sample(texts, 2)

            # Sort to handle symmetry
            pair_key = tuple(sorted((a, b)))

            if pair_key in seen_pairs:
                continue

            seen_pairs.add(pair_key)
            pairs.append(InputExample(texts=[a, b], label=1.0))
            count += 1

    logger.info(f"Total training pairs generated: {len(pairs)}")
    return pairs

# ==============================
# GENERATE VALIDATION PAIRS
# ==============================
def generate_validation_pairs(val_data, samples_per_class=30):
    """Generate validation pairs for evaluator"""
    val_pairs = []
    logger.info(f"Generating validation pairs (target: {samples_per_class} per class)...")

    for category in tqdm(val_data, desc="Validation pairs"):
        texts = val_data[category]
        if len(texts) < 2:
            continue

        num_pairs = min(samples_per_class, len(texts) * (len(texts) - 1) // 2)

        count = 0
        attempts = 0
        max_attempts = num_pairs * 3
        seen_pairs = set()

        while count < num_pairs and attempts < max_attempts:
            attempts += 1
            a, b = random.sample(texts, 2)
            pair_key = tuple(sorted((a, b)))

            if pair_key in seen_pairs:
                continue

            seen_pairs.add(pair_key)
            # Score of 1.0 for same disease
            val_pairs.append(InputExample(texts=[a, b], label=1.0))
            count += 1

    logger.info(f"Total validation pairs generated: {len(val_pairs)}")
    return val_pairs

# ==============================
# MAIN TRAINING
# ==============================
def main():
    logger.info("="*60)
    logger.info("SBERT V2 Training with Improved Hyperparameters")
    logger.info("="*60)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info(f"Samples per Class: {SAMPLES_PER_CLASS}")
    logger.info(f"Estimated Time: ~45 minutes (FAST mode)")
    logger.info("="*60)

    # 1. Load Data
    data = load_dataset(dataset_path)
    train_data, val_data, test_data = prepare_data_with_validation(data)

    # 2. Generate Training Pairs
    train_examples = generate_training_pairs(train_data, samples_per_class=SAMPLES_PER_CLASS)

    # 3. Generate Validation Pairs
    val_examples = generate_validation_pairs(val_data, samples_per_class=30)

    # 4. Setup Model
    logger.info(f"Initializing model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # 5. Setup Evaluator
    logger.info("Setting up validation evaluator...")
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples,
        name='disease-validation',
        write_csv=True
    )

    # 6. Training Loop
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    total_steps = len(train_dataloader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_STEPS_RATIO)

    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info("Starting training with validation...")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': LEARNING_RATE},
        evaluator=evaluator,                    # ← Validation evaluator
        evaluation_steps=EVALUATION_STEPS,       # ← Evaluate every N steps
        output_path=model_save_path,
        save_best_model=True,                    # ← Save only best model
        show_progress_bar=True
    )

    # 7. Save Final Model
    logger.info(f"Saving final model to {model_save_path}...")
    model.save(model_save_path)

    # 8. Build FAISS Index
    logger.info("Building FAISS index...")
    corpus_sentences = []
    corpus_labels = []

    # Index all data (train + val + test) for maximum coverage
    index_data = {**train_data, **val_data, **test_data}

    for category, texts in index_data.items():
        for text in texts:
            corpus_sentences.append(text)
            corpus_labels.append(category)

    logger.info(f"Encoding {len(corpus_sentences)} sentences for index...")
    embeddings = model.encode(
        corpus_sentences,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE
    )

    # Initialize FAISS (Inner Product for cosine similarity)
    dimension = embeddings.shape[1]
    logger.info(f"FAISS index dimension: {dimension}")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))

    logger.info(f"Saving FAISS index to {faiss_index_path}...")
    faiss.write_index(index, faiss_index_path)

    logger.info(f"Saving mapping to {disease_mapping_path}...")
    with open(disease_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(corpus_labels, f)

    # 9. Training Summary
    logger.info("="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Model saved: {model_save_path}")
    logger.info(f"FAISS index: {faiss_index_path}")
    logger.info(f"Disease mapping: {disease_mapping_path}")
    logger.info(f"Indexed sentences: {len(corpus_sentences)}")
    logger.info(f"Unique categories: {len(set(corpus_labels))}")
    logger.info("="*60)

    # 10. Save Training Config
    config = {
        "model_name": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "samples_per_class": SAMPLES_PER_CLASS,
        "total_training_pairs": len(train_examples),
        "total_validation_pairs": len(val_examples),
        "indexed_sentences": len(corpus_sentences),
        "num_categories": len(set(corpus_labels))
    }

    with open('training_config_v2.json', 'w') as f:
        json.dump(config, f, indent=2)

    logger.info("Training configuration saved to training_config_v2.json")

if __name__ == "__main__":
    main()
