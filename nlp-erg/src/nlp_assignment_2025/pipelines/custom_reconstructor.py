
"""
Custom rule-based text reconstructor.
Implements grammar corrections and linguistic improvements using predefined rules.
"""
from typing import Dict
from ..config import config

class CustomReconstructor:
    """Custom rule-based reconstructor for text improvement."""
    
    def __init__(self):
        """Initialize with predefined replacement rules."""
        self.replacements: Dict[str, str] = {
            "got": "received",
            "very appreciated": "greatly appreciate", 
            "Hope you too, to enjoy": "I hope you enjoy",
            "bit delay": "a slight delay",
            "less communication": "reduced communication",
            "very appreciate": "greatly appreciate",
            "Thank your message": "Thank you for your message",
            "all safe and great": "all safety and greatness",
        }
    
    def reconstruct(self, text: str) -> str:
        """Apply custom reconstruction rules to the text."""
        result = text
        for wrong, correct in self.replacements.items():
            result = result.replace(wrong, correct)
        return result

# Backward compatibility function
def custom_reconstruct_sentence(sentence: str) -> str:
    """Legacy function for backward compatibility."""
    reconstructor = CustomReconstructor()
    return reconstructor.reconstruct(sentence)