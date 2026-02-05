import json
import random
import os
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# ==============================
# CONFIGURATION
# ==============================
EPOCHS = 2
BATCH_SIZE = 16
LEARNING_RATE = 2e-5  # Slightly adjusted for fine-tuning
WARMUP_STEPS_RATIO = 0.1
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SAMPLES_PER_CLASS = 100  # Increased from 50 for better robustness

dataset_path = r"d:\APPLY\plant village\PlantVillage_train_prompts - Copy.json"
model_save_path = r"d:\APPLY\plant village\finetuned_sbert_model"
faiss_index_path = r"d:\APPLY\plant village\disease_faiss.index"
disease_mapping_path = r"d:\APPLY\plant village\disease_mapping.json"

# ==============================
# LOAD DATASET
# ==============================
def load_dataset(path):
    print(f"Loading dataset from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} categories.")
    return data

# ==============================
# PREPARE DATA
# ==============================
def prepare_data(data):
    train_data = {}
    test_data = {}
    
    print("Splitting data 70/30...")
    for category, prompts in data.items():
        # No filtering - use ALL categories
        prompts = [p.strip() for p in prompts if len(p.strip()) > 0]
        
        # Remove duplicates within category
        prompts = list(set(prompts))
        
        if len(prompts) < 5:
            print(f"Warning: Category '{category}' has very few samples ({len(prompts)}). using all for training.")
            train_data[category] = prompts
            test_data[category] = []
            continue
            
        split_idx = int(0.7 * len(prompts))
        train_data[category] = prompts[:split_idx]
        test_data[category] = prompts[split_idx:]
        
    return train_data, test_data

# ==============================
# GENERATE PAIRS
# ==============================
def generate_training_pairs(dataset, samples_per_class=100):
    pairs = []
    print(f"Generating training pairs (target: {samples_per_class} per class)...")
    
    for category in tqdm(dataset, desc="Categories"):
        texts = dataset[category]
        if len(texts) < 2:
            continue
            
        # Generate positive pairs (SimCLR style contrastive learning constraint)
        # We want the model to map different descriptions of the same disease close together
        
        # Strategy: Randomly sample pairs up to limit
        possible_pairs = len(texts) * (len(texts) - 1) // 2
        num_pairs = min(samples_per_class, possible_pairs)
        
        # If we have many texts, we randomly sample
        # If we have few, we iterate carefully to avoid infinite loops in naive sampling, 
        # but for simplicity and since datasets are small enough (~100 items), random sampling is fine.
        
        count = 0
        attempts = 0
        max_attempts = num_pairs * 5 # Prevent infinite loop
        
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
            
    print(f"Total training pairs generated: {len(pairs)}")
    return pairs

# ==============================
# MAIN TRAINING
# ==============================
def main():
    # 1. Load Data
    data = load_dataset(dataset_path)
    train_data, test_data = prepare_data(data)
    
    # 2. Generate Pairs
    train_examples = generate_training_pairs(train_data, samples_per_class=SAMPLES_PER_CLASS)
    
    # 3. Setup Model
    print(f"Initializing model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    # 4. Training Loop
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    total_steps = len(train_dataloader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_STEPS_RATIO)
    
    print("Starting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': LEARNING_RATE},
        show_progress_bar=True
    )
    
    # 5. Save Model
    print(f"Saving model to {model_save_path}...")
    model.save(model_save_path)
    
    # 6. Build FAISS Index
    print("Building FAISS index...")
    corpus_sentences = []
    corpus_labels = []
    
    # We index the TRAINING data as the "knowledge base"
    # Ideally we'd index everything or a representative set. Using train_data is standard for retrieval-augmented tasks on the same domain.
    # Actually, for reference, we should index ALL valid descriptions we have to be able to retrieve them.
    # Let's index the entire dataset (train + test) so the retriever has maximum coverage of "ground truth" descriptions.
    
    index_data = {**train_data, **test_data} # Merge
    
    for category, texts in index_data.items():
        for text in texts:
            corpus_sentences.append(text)
            corpus_labels.append(category)
            
    print(f"Encoding {len(corpus_sentences)} sentences for index...")
    embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    
    # Initialize FAISS (Inner Product for cosine similarity since normalized)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    
    print(f"Saving FAISS index to {faiss_index_path}...")
    faiss.write_index(index, faiss_index_path)
    
    print(f"Saving mapping to {disease_mapping_path}...")
    with open(disease_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(corpus_labels, f)
        
    print("Training and Indexing Complete!")

if __name__ == "__main__":
    main()
