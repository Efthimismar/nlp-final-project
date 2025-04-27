from preprocessing.tokenizer import tokenize_text
from pipelines.custom_reconstructor import custom_reconstruct_sentence
from pipelines.spacy_reconstructor import SpacyReconstructor
from pipelines.transformers_reconstructor import TransformersReconstructor

import os

def load_texts(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        texts = f.read().split('---')
    return [t.strip() for t in texts if t.strip()]

def get_data_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "input_texts.txt")
    return data_path

def get_output_dir():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(base_dir, "data", "output_texts")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_text(texts, filename):
    output_dir = get_output_dir()
    save_path = os.path.join(output_dir, filename)
    with open(save_path, "w", encoding="utf-8") as f:
        for idx, text in enumerate(texts):
            f.write(f"--- Reconstructed Text {idx+1} ---\n")
            f.write(text)
            f.write("\n\n")
    print(f"✅ Saved {filename}")

def main():
    print("🔍 Φόρτωση κειμένων...")
    data_path = get_data_path()
    texts = load_texts(data_path)

    # Pipelines
    spacy_pipeline = SpacyReconstructor()
    transformers_pipeline = TransformersReconstructor()

    # Collect reconstructed texts
    custom_outputs = []
    spacy_outputs = []
    transformers_outputs = []

    for idx, text in enumerate(texts):
        print(f"\n======= Κείμενο {idx+1} =======")

        print("\n🔵 Custom Pipeline:")
        custom = custom_reconstruct_sentence(text)
        print(custom)
        custom_outputs.append(custom)

        print("\n🟢 SpaCy Pipeline:")
        spacy = spacy_pipeline.reconstruct(text)
        print(spacy)
        spacy_outputs.append(spacy)

        print("\n🟣 Transformers Pipeline:")
        transformer = transformers_pipeline.reconstruct(text)
        print(transformer)
        transformers_outputs.append(transformer)

    # Save outputs
    save_text(custom_outputs, "custom_pipeline.txt")
    save_text(spacy_outputs, "spacy_pipeline.txt")
    save_text(transformers_outputs, "transformers_pipeline.txt")

if __name__ == "__main__":
    main()