import spacy

class SpacyReconstructor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def reconstruct(self, text: str) -> str:
        """
         basic cleaning using spaCy parsing (token reordering if necessary).
        """
        doc = self.nlp(text)
        reconstructed = []

        for sent in doc.sents:
            tokens = [token.text for token in sent]
            sentence = " ".join(tokens)
            reconstructed.append(sentence)

        return " ".join(reconstructed)