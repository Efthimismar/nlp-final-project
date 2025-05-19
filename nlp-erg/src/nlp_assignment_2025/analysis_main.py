import os
import pandas as pd
from nlp_assignment_2025.analysis.embeddings import EmbeddingLoader
from nlp_assignment_2025.analysis.similarity import compute_cosine_similarity
from nlp_assignment_2025.analysis.visualization import plot_pca

def get_project_root():
    # Πηγαίνει 3 φακέλους πάνω για να φτάσει στο root (nlp-erg/)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_texts_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        texts = f.read().split("---")
    return [t.strip() for t in texts if t.strip()]

def save_similarities_to_csv(similarity_dict, output_path):
    rows = []
    for pipeline, scores in similarity_dict.items():
        for i, score in enumerate(scores):
            rows.append({
                "Text": f"Text {i+1}",
                "Pipeline": pipeline,
                "CosineSimilarity": round(score, 4)
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"📁 Cosine Similarities saved to {output_path}")

def main():
    print("🔍 Φόρτωση embeddings...")
    embedder = EmbeddingLoader()

    base_path = get_project_root()
    data_dir = os.path.join(base_path, "data")
    output_dir = os.path.join(data_dir, "output_texts")

    # Load original
    original_path = os.path.join(data_dir, "input_texts.txt")
    originals = load_texts_from_file(original_path)

    # Load reconstructions
    custom_texts = load_texts_from_file(os.path.join(output_dir, "custom_pipeline.txt"))
    spacy_texts = load_texts_from_file(os.path.join(output_dir, "spacy_pipeline.txt"))
    transformers_texts = load_texts_from_file(os.path.join(output_dir, "transformers_pipeline.txt"))

    pipelines = {
        "custom": custom_texts,
        "spacy": spacy_texts,
        "transformers": transformers_texts
    }

    # Υπολογισμός και εμφάνιση cosine similarities
    similarity_scores = {}

    for name, texts in pipelines.items():
        print(f"\n🔹 Cosine Similarity για pipeline: {name}")
        scores = []
        for i in range(len(originals)):
            vec_orig = embedder.get_text_vector(originals[i])
            vec_recon = embedder.get_text_vector(texts[i])
            sim = compute_cosine_similarity(vec_orig, vec_recon)
            print(f"  Κείμενο {i+1} ➤ Cosine Similarity: {sim:.4f}")
            scores.append(sim)
        similarity_scores[name] = scores

    # Αποθήκευση σε .csv
    save_similarities_to_csv(
        similarity_scores,
        os.path.join(data_dir, "cosine_similarities.csv")
    )

    # PCA για Κείμενο 1
    print("\n🧠 PCA Visual για Κείμενο 1")
    vectors1 = [
        embedder.get_text_vector(originals[0]),
        embedder.get_text_vector(custom_texts[0]),
        embedder.get_text_vector(spacy_texts[0]),
        embedder.get_text_vector(transformers_texts[0]),
    ]
    labels1 = ["original", "custom", "spacy", "transformers"]
    plot_pca(vectors1, labels1, title="PCA - Semantic Shift (Text 1)")

    # PCA για Κείμενο 2
    print("\n🧠 PCA Visual για Κείμενο 2")
    vectors2 = [
        embedder.get_text_vector(originals[1]),
        embedder.get_text_vector(custom_texts[1]),
        embedder.get_text_vector(spacy_texts[1]),
        embedder.get_text_vector(transformers_texts[1]),
    ]
    labels2 = ["original", "custom", "spacy", "transformers"]
    plot_pca(vectors2, labels2, title="PCA - Semantic Shift (Text 2)")

if __name__ == "__main__":
    main()