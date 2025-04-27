import re
from typing import List

def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes a text into words using simple regex.
    """
    pattern = r"\b\w+\b"
    tokens = re.findall(pattern, text)
    return tokens