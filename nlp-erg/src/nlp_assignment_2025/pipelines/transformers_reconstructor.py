from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class TransformersReconstructor:
    def __init__(self, model_name="t5-small"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def reconstruct(self, text: str, max_length=512) -> str:
        """
        Reconstruct text using T5 paraphrasing model.
        """
        input_text = f"paraphrase: {text} </s>"
        encoding = self.tokenizer.encode_plus(
            input_text, padding="longest", return_tensors="pt", truncation=True
        )
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )

        paraphrased = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return paraphrased