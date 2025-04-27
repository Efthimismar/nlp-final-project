def custom_reconstruct_sentence(sentence: str) -> str:
    """
     simple manual rule-based reconstruction: basic grammar corrections.
    """

    replacements = {
        "got": "received",
        "very appreciated": "greatly appreciate",
        "Hope you too, to enjoy": "I hope you enjoy",
        "bit delay": "a slight delay",
        "less communication": "reduced communication"
    }

    for wrong, correct in replacements.items():
        sentence = sentence.replace(wrong, correct)
    
    return sentence