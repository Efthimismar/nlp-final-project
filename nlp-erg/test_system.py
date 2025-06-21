#!/usr/bin/env python3
"""
Test script for the NLP assignment codebase.
Validates all core functionality.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_functionality():
    """Test basic system functionality."""
    print("🧪 Testing NLP Assignment Functionality")
    print("=" * 45)
    
    try:
        # Test configuration
        from nlp_assignment_2025.config import config
        print(" Configuration loaded successfully")
        
        # Test utilities
        from nlp_assignment_2025.utils import load_texts_from_file
        texts = load_texts_from_file(config.INPUT_TEXTS_FILE)
        print(f"Text loading works - {len(texts)} texts loaded")
        
        # Test reconstruction
        from nlp_assignment_2025.pipelines.custom_reconstructor import CustomReconstructor
        reconstructor = CustomReconstructor()
        result = reconstructor.reconstruct("I got this message")
        print("Text reconstruction functional")
        
        # Test analysis
        from nlp_assignment_2025.analysis.embeddings import EmbeddingLoader
        loader = EmbeddingLoader()
        embeddings = loader.get_sentence_embeddings(["Test sentence"])
        print("Embedding analysis operational")
        
        # Test main modules
        from nlp_assignment_2025 import main, enhanced_analysis_main, bonus_masked_lm
        print("All main modules importable")
        
        print("=" * 45)
        print("All tests passed! System is operational.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
