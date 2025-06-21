"""
Utility functions for the NLP assignment.
Centralized functions for file I/O, text processing, and common operations.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .config import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_texts_from_file(file_path: Path) -> List[str]:
    """Load and split texts from file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        texts = content.split("---")
        texts = [text.strip() for text in texts if text.strip()]
        logger.info(f"Loaded {len(texts)} texts from {file_path}")
        return texts
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading texts from {file_path}: {e}")
        return []

def save_texts_to_file(texts: List[str], file_path: Path, prefix: str = "Text") -> None:
    """Save texts to file with consistent formatting."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for idx, text in enumerate(texts):
                f.write(f"--- Reconstructed {prefix} {idx+1} ---\n")
                f.write(text)
                f.write("\n\n")
        logger.info(f"Saved {len(texts)} texts to {file_path}")
    except Exception as e:
        logger.error(f"Error saving texts to {file_path}: {e}")

def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """Save data to JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")

def load_json(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON data from {file_path}")
        return data
    except FileNotFoundError:
        logger.warning(f"JSON file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None

def ensure_output_directory() -> None:
    """Ensure output directory exists."""
    config.ensure_directories()

class TextProcessor:
    """Simple text processing utilities."""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Basic text normalization."""
        return text.strip().replace('\n', ' ').replace('\r', '')
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Basic sentence splitting."""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text for processing."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
