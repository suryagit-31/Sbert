import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from lime.lime_text import LimeTextExplainer

# ==============================
# CONFIGURATION - V2 PATHS
# ==============================
MODEL_PATH = r"finetuned_sbert_model_v2"
FAISS_INDEX_PATH = r"disease_faiss_v2.index"
DISEASE_MAPPING_PATH = r"disease_mapping_v2.json"
THRESHOLD = 0.4  # Slightly lower threshold to allow more flexible matches
TOP_K = 3

# Stopwords to clean up LIME explanations
STOPWORDS = {
    "and", "or", "are", "is", "on", "in", "of", "the", "a", "an", "it", "this", "that",
    "plant", "plants", "leaf", "leaves", "foliage", "fruit", "fruits", "vegetable",

    # Crop names - filter these out to focus on symptoms
    "tomato", "apple", "corn", "maize", "potato", "grape", "soybean", "orange",
    "cedar",  # for Cedar Apple Rust

    # Generic disease words
    "very", "turn", "become", "appear", "look", "show", "symptom", "symptoms", "condition",
    "infected", "affected", "disease", "severe", "cases", "early", "late", "stage",
    "upper", "lower", "surface", "side", "under", "over", "small", "large"
}

# ==============================
# CLASS: DISEASE DETECTOR V2
# ==============================
class DiseaseDetectorV2:
    def __init__(self):
        print("Loading SBERT V2 model...")
        self.model = SentenceTransformer(MODEL_PATH)

        print("Loading FAISS index...")
        self.index = faiss.read_index(FAISS_INDEX_PATH)

        print("Loading disease mapping...")
        with open(DISEASE_MAPPING_PATH, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)

    def predict(self, query):
        emb = self.model.encode(
            [query.lower()],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Search the index
        D, I = self.index.search(emb.astype("float32"), k=50)

        # Aggregate scores by disease category
        scores = {}
        for d, i in zip(D[0], I[0]):
            if i < len(self.labels):
                disease = self.labels[i]
                if disease not in scores:
                    scores[disease] = float(d)
                else:
                    scores[disease] = max(scores[disease], float(d))

        # Filter and Sort
        results = [
            {"disease": k, "confidence": v}
            for k, v in scores.items()
            if v >= THRESHOLD
        ]

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:TOP_K]

# ==============================
# LIME EXPLAINABILITY
# ==============================
class DiseaseExplainer:
    def __init__(self, detector):
        self.detector = detector
        self.explainer = LimeTextExplainer(class_names=["not_disease", "disease"])

    def _lime_predict_proba(self, texts):
        outputs = []
        for t in texts:
            predictions = self.detector.predict(t)

            if predictions:
                conf = predictions[0]["confidence"]
                conf = max(0.0, min(1.0, conf))
            else:
                conf = 0.0

            outputs.append([1 - conf, conf])

        return np.array(outputs)

    def explain(self, query):
        print(f"Generating explanation for: '{query}'...")
        explanation = self.explainer.explain_instance(
            query,
            self._lime_predict_proba,
            num_features=10,
            labels=[1]
        )

        # Extract meaningful words
        cleaned_symptoms = []
        for word, weight in explanation.as_list(label=1):
            w = str(word).lower().strip()
            if weight > 0 and w not in STOPWORDS and len(w) > 2:
                cleaned_symptoms.append(w)

        # Deduplicate preserving order
        cleaned_symptoms = list(dict.fromkeys(cleaned_symptoms))
        return cleaned_symptoms[:5]

# ==============================
# PIPELINE
# ==============================
def full_pipeline(query, detector=None, explainer=None):
    if detector is None:
        detector = DiseaseDetectorV2()
    if explainer is None:
        explainer = DiseaseExplainer(detector)

    predictions = detector.predict(query)

    if not predictions:
        return {
            "query": query,
            "error": "No disease detected (low confidence)."
        }

    # Get explanation (symptoms)
    key_symptoms = explainer.explain(query)

    formatted_preds = []
    for i, p in enumerate(predictions, 1):
        formatted_preds.append({
            "rank": i,
            "disease": p["disease"],
            "confidence": round(p["confidence"], 2)
        })

    return {
        "query": query,
        "top_predictions": formatted_preds,
        "key_symptoms_identified": key_symptoms
    }

# ==============================
# TEST RUN
# ==============================
if __name__ == "__main__":
    # Example queries covering different crops
    queries = [
        "Apple leaves have large black spots and are curling",
        "Corn leaves have gray lesions that are rectangular",
        "White powdery substance on the surface of grape leaves",
        "Tomato leaves turning yellow with dark brown spots",
        "Potato leaves have dark lesions and wilting"
    ]

    detector = DiseaseDetectorV2()
    explainer = DiseaseExplainer(detector)

    print("\n" + "="*50)
    print("RUNNING INFERENCE TEST - V2 MODEL")
    print("="*50)

    for q in queries:
        print(f"\nQuery: {q}")
        result = full_pipeline(q, detector, explainer)
        print(json.dumps(result, indent=2))
