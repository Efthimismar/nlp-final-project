"""
SpaCy-based text reconstructor.
Uses spaCy NLP pipeline for text parsing and reconstruction.
"""
import spacy
from typing import Optional
from ..config import config

class SpacyReconstructor:
    """SpaCy-based text reconstructor."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize with spaCy model."""
        model_name = model_name or config.SPACY_MODEL
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"⚠️ Warning: spaCy model '{model_name}' not found. Using 'en_core_web_sm' as fallback.")
            self.nlp = spacy.load("en_core_web_sm")

    def reconstruct(self, text: str) -> str:
        """Reconstruct text using spaCy parsing."""
        doc = self.nlp(text)
        reconstructed = []

        for sent in doc.sents:
            # Extract tokens and join them with proper spacing
            tokens = [token.text for token in sent if not token.is_space]
            sentence = " ".join(tokens)
            reconstructed.append(sentence)

        return " ".join(reconstructed)