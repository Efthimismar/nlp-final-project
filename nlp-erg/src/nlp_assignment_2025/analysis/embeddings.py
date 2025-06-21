"""
Embedding and similarity analysis for NLP assignment.
Provides multiple embedding types and visualization capabilities.
"""
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from ..config import config

class EmbeddingLoader:
    """Handles loading and managing different embedding models."""
    
    def __init__(self):
        self.models = {
            'sentence_transformers': None,
            'bert': None,
        }
        self.tokenizer = None
        
    def load_sentence_transformer(self, model_name: Optional[str] = None) -> Any:
        """Load sentence transformer model."""
        model_name = model_name or config.SENTENCE_TRANSFORMER_MODEL
        if self.models['sentence_transformers'] is None:
            from sentence_transformers import SentenceTransformer
            self.models['sentence_transformers'] = SentenceTransformer(model_name)
        return self.models['sentence_transformers']
        
    def get_sentence_embeddings(self, texts: List[str], model_type: str = 'sentence_transformers') -> np.ndarray:
        """Get embeddings for sentences/texts."""
        if model_type == 'sentence_transformers':
            model = self.load_sentence_transformer()
            return model.encode(texts)
        elif model_type == 'bert':
            return self._get_bert_embeddings(texts)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _get_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get BERT embeddings."""
        if self.models['bert'] is None:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
            self.models['bert'] = AutoModel.from_pretrained(config.BERT_MODEL_NAME)
        
        import torch
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, 
                                  truncation=True, max_length=config.MAX_LENGTH)
            with torch.no_grad():
                outputs = self.models['bert'](**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)
    
    def get_word2vec_embeddings(self, texts: List[str]) -> tuple[np.ndarray, Any]:
        """Train Word2Vec and get document embeddings."""
        import nltk
        try:
            from gensim.models import Word2Vec
        except ImportError:
            print("Warning: gensim Word2Vec not available. Using random embeddings.")
            return np.random.random((len(texts), 100)), None
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        # Tokenize texts
        tokenized_texts = [nltk.word_tokenize(text.lower()) for text in texts]
        
        # Train Word2Vec
        model = Word2Vec(sentences=tokenized_texts, vector_size=100, 
                        window=5, min_count=1, workers=4, seed=config.RANDOM_SEED)
        
        # Get document embeddings by averaging word vectors
        doc_embeddings = []
        for tokens in tokenized_texts:
            word_vecs = [model.wv[word] for word in tokens if word in model.wv]
            if word_vecs:
                doc_embedding = np.mean(word_vecs, axis=0)
            else:
                doc_embedding = np.zeros(100)
            doc_embeddings.append(doc_embedding)
        
        return np.array(doc_embeddings), model
    
    def get_text_vector(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a single text (backward compatibility)."""
        try:
            embeddings = self.get_sentence_embeddings([text])
            return embeddings[0] if len(embeddings) > 0 else None
        except Exception:
            return None

class SimilarityAnalyzer:
    """Analyzes similarity between texts using embeddings."""
    
    def __init__(self):
        self.embedding_loader = EmbeddingLoader()
    
    def calculate_cosine_similarity(self, original_texts: List[str], 
                                  reconstructed_texts: List[str], 
                                  embedding_type: str = 'sentence_transformers') -> List[float]:
        """Calculate cosine similarity between original and reconstructed texts."""
        original_embeddings = self.embedding_loader.get_sentence_embeddings(original_texts, embedding_type)
        reconstructed_embeddings = self.embedding_loader.get_sentence_embeddings(reconstructed_texts, embedding_type)
        
        similarities = []
        for i in range(len(original_texts)):
            sim = cosine_similarity([original_embeddings[i]], [reconstructed_embeddings[i]])[0][0]
            similarities.append(sim)
        
        return similarities
    
    def compare_methods(self, original_texts: List[str], 
                       reconstructed_dict: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        """Compare different reconstruction methods."""
        results = {}
        
        for method_name, reconstructed_texts in reconstructed_dict.items():
            similarities = self.calculate_cosine_similarity(original_texts, reconstructed_texts)
            results[method_name] = {
                'similarities': similarities,
                'mean_similarity': np.mean(similarities),
                'std_similarity': np.std(similarities)
            }
        
        return results
    
    def visualize_embeddings(self, texts: List[str], labels: List[str], 
                           method: str = 'pca', title: str = "Embedding Visualization") -> np.ndarray:
        """Visualize embeddings using PCA or t-SNE."""
        embeddings = self.embedding_loader.get_sentence_embeddings(texts)
        
        # dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=config.RANDOM_SEED)
        else:  # t-SNE
            reducer = TSNE(n_components=2, random_state=config.RANDOM_SEED, perplexity=min(30, len(texts)-1))
        
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        # create the visualization
        plt.figure(figsize=config.FIGURE_SIZE)
        colors = plt.cm.Set3(np.linspace(0, 1, len(set(labels))))
        
        for i, label in enumerate(set(labels)):
            mask = np.array(labels) == label
            plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], 
                       c=[colors[i]], label=label, alpha=0.7, s=100)
        
        plt.title(title)
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return reduced_embeddings