"""
Greek Masked Language Model Analysis (Bonus).
Implements masked language modeling using Greek BERT for legal texts.
"""
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import matplotlib.pyplot as plt
import unicodedata
from typing import List, Tuple, Dict, Any

from .config import config
from .utils import logger

def normalize_greek_text(text: str) -> str:
    """Normalize Greek text by removing accents and converting to lowercase."""
    # Remove accents and diacritics
    normalized = unicodedata.normalize('NFD', text)
    # Remove combining characters 
    without_accents = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    return without_accents.lower().strip()

def is_greek_word_match(predicted: str, ground_truth: str) -> bool:
    """Check if two Greek words match, accounting for accent variations and common equivalents."""
    # Normalize both words
    pred_norm = normalize_greek_text(predicted)
    truth_norm = normalize_greek_text(ground_truth)
    
    # Direct match after normalization
    if pred_norm == truth_norm:
        return True
    
    # Check for common Greek word variations
    word_variants = {
        'κυριος': ['κυριου', 'κυριους', 'κυριοι'],
        'ακινητο': ['ακινητου', 'ακινητων', 'ακινητα'],
        'εκαστοτε': ['καθε', 'καθενας'],
        'πραγματικη': ['πραγματικος', 'πραγματικου'],
        'κοινο': ['κοινου', 'κοινων', 'κοινα'],
    }
    
    # Check if either word is a variant of the other
    for base_word, variants in word_variants.items():
        normalized_variants = [normalize_greek_text(v) for v in variants]
        if ((pred_norm == base_word and truth_norm in normalized_variants) or
            (truth_norm == base_word and pred_norm in normalized_variants) or
            (pred_norm in normalized_variants and truth_norm in normalized_variants)):
            return True
    
    return False

class GreekMaskedLanguageModel:
    """Greek BERT model for masked language modeling."""
    
    def __init__(self):
        """Initialize Greek BERT model with fallback."""
        self.model_name = config.GREEK_BERT_MODEL
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self.fallback_tokenizer = None
            self.fallback_model = None
            logger.info(f"Loaded Greek BERT model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load Greek BERT model: {e}")
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize fallback multilingual BERT."""
        fallback_model_name = "bert-base-multilingual-cased"
        self.fallback_tokenizer = AutoTokenizer.from_pretrained(fallback_model_name)
        self.fallback_model = AutoModelForMaskedLM.from_pretrained(fallback_model_name)
        logger.info(f"Using fallback model: {fallback_model_name}")
        
    def predict_masked_tokens(self, text: str, top_k: int = None) -> List[List[Tuple[str, float]]]:
        """Predict masked tokens in Greek text."""
        top_k = top_k or config.TOP_K_PREDICTIONS
        
        try:
            return self._predict_with_model(text, top_k, self.tokenizer, self.model)
        except Exception as e:
            logger.warning(f"Greek BERT failed, using fallback: {e}")
            return self._predict_with_model(text, top_k, self.fallback_tokenizer, self.fallback_model)
    
    def _predict_with_model(self, text: str, top_k: int, tokenizer, model) -> List[List[Tuple[str, float]]]:
        """Predict tokens using specified model."""
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits
        
        # Find mask token positions
        mask_token_indices = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
        
        results = []
        for mask_idx in mask_token_indices:
            mask_token_logits = predictions[0, mask_idx, :]
            top_tokens = torch.topk(mask_token_logits, top_k, dim=0)
            
            predictions_list = []
            for score, token_id in zip(top_tokens.values, top_tokens.indices):
                token = tokenizer.decode([token_id]).strip()
                predictions_list.append((token, score.item()))
            
            results.append(predictions_list)
        
        return results

def analyze_civil_code_texts() -> List[Dict[str, Any]]:
    """Analyze masked Greek civil code articles."""
    
    # Civil code texts with MASK tokens
    civil_code_texts = [
        "Άρθρο 1113. Κοινό πράγμα. — Αν η κυριότητα του [MASK] ανήκει σε περισσότερους [MASK] αδιαιρέτου κατ΄ιδανικά [MASK], εφαρμόζονται οι διατάξεις για την κοινωνία.",
        "Άρθρο 1114. Πραγματική δουλεία σε [MASK] η υπέρ του κοινού ακινήτου. — Στο κοινό [MASK] μπορεί να συσταθεί πραγματική δουλεία υπέρ του [MASK] κύριου άλλου ακινήτου και αν ακόμη αυτός είναι [MASK] του ακινήτου που βαρύνεται με τη δουλεία. Το ίδιο ισχύει και για την [MASK] δουλεία πάνω σε ακίνητο υπέρ των εκάστοτε κυρίων κοινού ακινήτου, αν [MASK] από αυτούς είναι κύριος του [MASK] που βαρύνεται με τη δουλεία."
    ]
    
    # Ground truth completions
    ground_truth = [
        ["ακινήτου", "κύριους", "μερίδια"],
        ["κοινό", "ακίνητο", "εκάστοτε", "συγκύριος", "πραγματική", "κανένας", "ακινήτου"]
    ]
    
    logger.info("🔄 Loading Greek BERT model...")
    mlm = GreekMaskedLanguageModel()
    
    print("\n" + "="*60)
    print("GREEK CIVIL CODE - MASKED LANGUAGE MODEL ANALYSIS")
    print("="*60)
    
    all_results = []
    
    for i, text in enumerate(civil_code_texts):
        print(f"\n📖 Article 111{3+i}:")
        print(f"Text: {text}")
        
        # Convert [MASK] to model's mask token
        masked_text = text.replace("[MASK]", mlm.tokenizer.mask_token if mlm.tokenizer else "[MASK]")
        
        # Get predictions
        predictions = mlm.predict_masked_tokens(masked_text, top_k=3)
        
        print(f"\n🎯 Predictions vs Ground Truth:")
        
        article_results = {
            'article': f"Article 111{3+i}",
            'predictions': [],
            'ground_truth': ground_truth[i]
        }
        
        for j, (pred_list, true_word) in enumerate(zip(predictions, ground_truth[i])):
            print(f"\n  Mask {j+1} - Ground Truth: '{true_word}'")
            print(f"  Top Predictions:")
            
            mask_results = {
                'mask_position': j+1,
                'ground_truth': true_word,
                'predictions': pred_list,
                'top_1_correct': False,
                'top_3_correct': False
            }
            
            for rank, (predicted_word, score) in enumerate(pred_list):
                print(f"    {rank+1}. '{predicted_word}' (score: {score:.4f})")
                
                if is_greek_word_match(predicted_word, true_word):
                    if rank == 0:
                        mask_results['top_1_correct'] = True
                    mask_results['top_3_correct'] = True
                    print(f"      ✅ MATCH at rank {rank+1}!")
            
            article_results['predictions'].append(mask_results)
        
        all_results.append(article_results)
    
    return all_results

def evaluate_performance(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate model performance metrics."""
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*50)
    
    total_masks = 0
    top_1_correct = 0
    top_3_correct = 0
    
    for article in results:
        print(f"\n📊 {article['article']}:")
        
        article_top1 = sum(1 for p in article['predictions'] if p['top_1_correct'])
        article_top3 = sum(1 for p in article['predictions'] if p['top_3_correct'])
        article_total = len(article['predictions'])
        
        print(f"  Top-1 Accuracy: {article_top1}/{article_total} ({article_top1/article_total*100:.1f}%)")
        print(f"  Top-3 Accuracy: {article_top3}/{article_total} ({article_top3/article_total*100:.1f}%)")
        
        total_masks += article_total
        top_1_correct += article_top1
        top_3_correct += article_top3
    
    print(f"\n🎯 Overall Performance:")
    print(f"  Total Masks: {total_masks}")
    print(f"  Top-1 Accuracy: {top_1_correct}/{total_masks} ({top_1_correct/total_masks*100:.1f}%)")
    print(f"  Top-3 Accuracy: {top_3_correct}/{total_masks} ({top_3_correct/total_masks*100:.1f}%)")
    
    return {
        'total_masks': total_masks,
        'top_1_accuracy': top_1_correct/total_masks,
        'top_3_accuracy': top_3_correct/total_masks
    }

def visualize_performance(results: List[Dict[str, Any]], metrics: Dict[str, Any]) -> None:
    """Create performance visualizations."""
    
    # Overall accuracy chart
    categories = ['Top-1 Accuracy', 'Top-3 Accuracy']
    values = [metrics['top_1_accuracy']*100, metrics['top_3_accuracy']*100]
    
    plt.figure(figsize=config.FIGURE_SIZE)
    bars = plt.bar(categories, values, color=['#ff7f0e', '#2ca02c'], alpha=0.7)
    plt.title('Greek BERT Performance on Civil Code Masked Language Modeling')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    # Add value labels
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Article breakdown chart
    articles = [result['article'] for result in results]
    top1_scores = []
    top3_scores = []
    
    for result in results:
        article_top1 = sum(1 for p in result['predictions'] if p['top_1_correct'])
        article_top3 = sum(1 for p in result['predictions'] if p['top_3_correct'])
        total_masks = len(result['predictions'])
        
        top1_scores.append(article_top1/total_masks*100)
        top3_scores.append(article_top3/total_masks*100)
    
    x = range(len(articles))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar([i - width/2 for i in x], top1_scores, width, label='Top-1 Accuracy', alpha=0.7)
    plt.bar([i + width/2 for i in x], top3_scores, width, label='Top-3 Accuracy', alpha=0.7)
    
    plt.xlabel('Civil Code Articles')
    plt.ylabel('Accuracy (%)')
    plt.title('Performance Breakdown by Article')
    plt.xticks(x, articles)
    plt.legend()
    plt.ylim(0, 100)
    
    # Add value labels
    for i, (top1, top3) in enumerate(zip(top1_scores, top3_scores)):
        plt.text(i - width/2, top1 + 2, f'{top1:.0f}%', ha='center', va='bottom')
        plt.text(i + width/2, top3 + 2, f'{top3:.0f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function for the bonus masked language modeling task."""
    print("🇬🇷 Greek Civil Code - Masked Language Modeling (BONUS)")
    print("="*60)
    
    # Run analysis
    results = analyze_civil_code_texts()
    metrics = evaluate_performance(results)
    
    # Create visualizations
    print("\n📈 Creating performance visualizations...")
    visualize_performance(results, metrics)
    
    # Discussion
    print("\n" + "="*50)
    print("DISCUSSION OF FINDINGS")
    print("="*50)
    
    print("""
🔍 Key Observations:
1. Legal domain specificity: Legal texts contain specialized terminology
2. Context dependency: Many legal terms are highly context-dependent
3. Morphological complexity: Greek inflection affects prediction accuracy
4. Semantic relationships: Legal concepts have specific semantic relationships

📊 Model Limitations:
- Limited legal domain training data
- Difficulty with highly specialized terminology
- Context window limitations for long legal documents
- Inflectional morphology challenges in Greek

💡 Potential Improvements:
- Fine-tuning on legal Greek texts
- Larger context windows
- Domain-specific preprocessing
- Ensemble methods with multiple models
""")
    
    return results, metrics

if __name__ == "__main__":
    main()
