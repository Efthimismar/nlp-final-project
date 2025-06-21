"""
Transformers-based text reconstructor using T5 model.
Provides paraphrasing and grammar correction capabilities.
"""
import torch
from typing import Optional
from ..config import config

class TransformersReconstructor:
    """T5-based text reconstructor."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize T5 model and tokenizer."""
        model_name = model_name or config.T5_MODEL_NAME
        
        try:
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model_available = True
        except ImportError as e:
            print(f" Could not load T5 model ({e}). Using fallback reconstruction.")
            self.tokenizer = None
            self.model = None
            self.model_available = False

    def reconstruct(self, text: str, task: str = "paraphrase") -> str:
        """Reconstruct text using T5 with task-specific prompts."""
        if not self.model_available:
            return f"[T5 unavailable] {text}"
        
        # Task-specific prompting
        task_prompts = {
            "grammar": f"grammar: {text}",
            "clarity": f"improve clarity: {text}",
            "paraphrase": f"paraphrase: {text}"
        }
        
        input_text = task_prompts.get(task, f"paraphrase: {text}")
        
        # Encode input
        encoding = self.tokenizer.encode_plus(
            input_text, 
            padding="longest", 
            return_tensors="pt", 
            truncation=True, 
            max_length=config.MAX_LENGTH
        )
        
        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                max_length=config.MAX_LENGTH,
                num_return_sequences=1,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True,
                do_sample=True,
                temperature=0.7
            )

        # Decode result
        reconstructed = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return reconstructed

    def grammar_correction(self, text: str) -> str:
        """Perform grammar correction using T5."""
        return self.reconstruct(text, task="grammar")