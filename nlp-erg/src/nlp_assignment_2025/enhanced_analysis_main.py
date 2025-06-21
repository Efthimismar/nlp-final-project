"""
Enhanced computational analysis module (Deliverable 2).
Comprehensive similarity analysis and visualization of reconstruction methods.
"""
import numpy as np
import matplotlib.pyplot as plt

from .config import config
from .utils import load_texts_from_file, save_json, logger
from .analysis.embeddings import SimilarityAnalyzer
from .pipelines.custom_reconstructor import CustomReconstructor
from .pipelines.spacy_reconstructor import SpacyReconstructor
from .pipelines.transformers_reconstructor import TransformersReconstructor

def run_all_reconstructions(texts):
    """Run all reconstruction methods and return results."""
    logger.info("🔄 Running all reconstruction methods...")
    
    # Initialize reconstructors
    custom_reconstructor = CustomReconstructor()
    spacy_reconstructor = SpacyReconstructor()
    transformers_reconstructor = TransformersReconstructor()
    
    results = {
        'original': texts,
        'custom': [],
        'spacy': [],
        'transformers_paraphrase': [],
        'transformers_grammar': []
    }
    
    for i, text in enumerate(texts):
        logger.info(f"Processing text {i+1}/{len(texts)}...")
        
        # Apply all reconstruction methods
        results['custom'].append(custom_reconstructor.reconstruct(text))
        results['spacy'].append(spacy_reconstructor.reconstruct(text))
        results['transformers_paraphrase'].append(
            transformers_reconstructor.reconstruct(text, task="paraphrase")
        )
        results['transformers_grammar'].append(
            transformers_reconstructor.reconstruct(text, task="grammar")
        )
    
    return results

def analyze_similarities(results):
    """Analyze similarities between original and reconstructed texts."""
    logger.info("📊 Analyzing similarities...")
    analyzer = SimilarityAnalyzer()
    
    # Prepare reconstructed texts dictionary
    reconstructed_dict = {
        'Custom': results['custom'],
        'SpaCy': results['spacy'], 
        'T5-Paraphrase': results['transformers_paraphrase'],
        'T5-Grammar': results['transformers_grammar']
    }
    
    return analyzer.compare_methods(results['original'], reconstructed_dict)

def visualize_results(results, similarity_results):
    """Create comprehensive visualizations."""
    logger.info("📈 Creating visualizations...")
    
    # 1. Similarity comparison bar chart
    methods = list(similarity_results.keys())
    mean_similarities = [similarity_results[method]['mean_similarity'] for method in methods]
    std_similarities = [similarity_results[method]['std_similarity'] for method in methods]
    
    plt.figure(figsize=config.FIGURE_SIZE)
    bars = plt.bar(methods, mean_similarities, yerr=std_similarities, capsize=5, alpha=0.7)
    plt.title('Mean Cosine Similarity by Reconstruction Method')
    plt.ylabel('Cosine Similarity')
    plt.xlabel('Method')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, mean_sim in zip(bars, mean_similarities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{mean_sim:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Embedding visualizations
    analyzer = SimilarityAnalyzer()
    
    # Prepare all texts for visualization
    all_texts = []
    all_labels = []
    
    for i, original in enumerate(results['original']):
        all_texts.extend([
            original,
            results['custom'][i],
            results['spacy'][i], 
            results['transformers_paraphrase'][i]
        ])
        all_labels.extend([
            f'Original_{i+1}',
            f'Custom_{i+1}',
            f'SpaCy_{i+1}',
            f'T5-Para_{i+1}'
        ])
    
    # Create PCA and t-SNE visualizations
    analyzer.visualize_embeddings(all_texts, all_labels, method='pca', 
                                title='PCA Visualization of Text Embeddings')
    analyzer.visualize_embeddings(all_texts, all_labels, method='tsne', 
                                title='t-SNE Visualization of Text Embeddings')

def main():
    """Main analysis function for computational analysis."""
    logger.info("🧠 Starting Computational Analysis (Deliverable 2)")
    
    # Load and validate texts
    if not config.validate_config():
        return
        
    texts = load_texts_from_file(config.INPUT_TEXTS_FILE)
    if not texts:
        logger.error("No texts loaded. Exiting.")
        return
        
    logger.info(f"📖 Loaded {len(texts)} texts")
    
    # Run reconstructions and analysis
    results = run_all_reconstructions(texts)
    similarity_results = analyze_similarities(results)
    
    # Display results
    print("\n" + "="*50)
    print("SIMILARITY ANALYSIS RESULTS")
    print("="*50)
    
    for method, result in similarity_results.items():
        print(f"\n{method}:")
        print(f"  Mean Similarity: {result['mean_similarity']:.4f}")
        print(f"  Std Deviation: {result['std_similarity']:.4f}")
        print(f"  Individual Similarities: {[f'{s:.4f}' for s in result['similarities']]}")
    
    # Create visualizations
    visualize_results(results, similarity_results)
    
    # Save results
    config.ensure_directories()
    save_json(results, config.get_output_path("reconstruction_results.json"))
    save_json(similarity_results, config.get_output_path("similarity_analysis.json"))
    
    logger.info("✅ Analysis complete!")
    return results, similarity_results

if __name__ == "__main__":
    main()
