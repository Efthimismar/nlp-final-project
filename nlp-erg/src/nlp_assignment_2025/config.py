"""
Configuration management for NLP Assignment 2025.
Handles environment variables and project settings.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Central configuration class for the NLP assignment."""
    
    # Project structure
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = DATA_DIR / "output_texts"
    
    # Input files
    INPUT_TEXTS_FILE = DATA_DIR / "input_texts.txt"
    SIMILARITIES_CSV = DATA_DIR / "cosine_similarities.csv"
    
    # Model configurations from environment variables
    T5_MODEL_NAME = os.getenv("T5_MODEL_NAME", "t5-small")
    SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
    BERT_MODEL_NAME = os.getenv("BERT_MODEL_NAME", "bert-base-uncased")
    GREEK_BERT_MODEL = os.getenv("GREEK_BERT_MODEL", "nlpaueb/bert-base-greek-uncased-v1")
    SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
    
    # Processing parameters
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
    TOP_K_PREDICTIONS = int(os.getenv("TOP_K_PREDICTIONS", "5"))
    RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
    
    # Visualization settings
    FIGURE_SIZE = tuple(map(int, os.getenv("FIGURE_SIZE", "12,8").split(",")))
    DPI = int(os.getenv("DPI", "300"))
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_output_path(cls, filename: str) -> Path:
        """Get full path for output file."""
        cls.ensure_directories()
        return cls.OUTPUT_DIR / filename
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required files and settings exist."""
        if not cls.INPUT_TEXTS_FILE.exists():
            print(f"⚠️ Warning: Input texts file not found at {cls.INPUT_TEXTS_FILE}")
            return False
        return True

# Global config instance
config = Config()
