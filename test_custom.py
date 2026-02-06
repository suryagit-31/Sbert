import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model and index
print("Loading V2 model...")
model = SentenceTransformer("finetuned_sbert_model_v2")
index = faiss.read_index("disease_faiss_v2.index")

with open("disease_mapping_v2.json", 'r', encoding='utf-8') as f:
    labels = json.load(f)

print(f"\nModel loaded! Index contains {index.ntotal} embeddings\n")
print("="*60)

# Test queries
test_queries = [
    "Small yellow spots on tomato leaves",
    "Brown patches spreading on potato plant",
    "Orange rust pustules on corn leaves",
    "Leaves have white mold or powder",
    "Black spots with yellow halo on apple",
    "Healthy green corn plant",
    "Wilting leaves with brown edges",
    "Circular spots with concentric rings",
    "Early blight symptoms on tomato",
    "Downy mildew on grape leaves"
]

print("TESTING V2 MODEL WITH CUSTOM QUERIES")
print("="*60)

for query in test_queries:
    # Encode query
    query_embedding = model.encode([query], normalize_embeddings=True)

    # Search
    distances, indices = index.search(query_embedding.astype('float32'), k=3)

    print(f"\n>> Query: {query}")
    print("-" * 60)

    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
        confidence = float(dist)
        disease = labels[idx]

        # Color coding based on confidence
        if confidence >= 0.7:
            status = "[HIGH]"
        elif confidence >= 0.5:
            status = "[MEDIUM]"
        else:
            status = "[LOW]"

        print(f"  {rank}. {disease}")
        print(f"     Confidence: {confidence:.2f} {status}")

print("\n" + "="*60)
print("Testing Complete!")
