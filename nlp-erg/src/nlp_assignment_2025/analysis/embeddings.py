import spacy

class EmbeddingLoader:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")  # μεσαίου μεγέθους μοντέλο με έτοιμα word vectors

    def get_text_vector(self, text: str):
        """
        Παίρνει ένα κείμενο και επιστρέφει το μέσο όρο των word vectors του.
        """
        doc = self.nlp(text)
        if len(doc) == 0:
            return None
        return doc.vector