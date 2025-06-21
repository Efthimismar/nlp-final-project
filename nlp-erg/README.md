# Natural Language Analysis Project 2025

## Overview

This project implements semantic similarity analysis, word embeddings, and linguistic reconstruction techniques to transform unstructured or semantically ambiguous texts into clear, well-structured versions. The analysis employs cosine similarity, word embeddings, and advanced NLP techniques to evaluate reconstruction quality.

This project features environment variable configuration, comprehensive error handling, and clean, maintainable code structure following Python best practices.

## Project Structure

```
nlp-erg/
├── README.md                          # This documentation
├── pyproject.toml                    # Poetry configuration
├── .env                              # Environment variables for all settings
├── data/                             # Input and output data
│   ├── input_texts.txt              # Original assignment texts
│   ├── cosine_similarities.csv      # Similarity analysis results
│   └── output_texts/                # Reconstructed text outputs
├── src/                             # Source code
│   ├── menu.py                      # Main application menu
│   └── nlp_assignment_2025/         # Core NLP modules
│       ├── config.py                # Centralized configuration
│       ├── utils.py                 # Utility functions
│       ├── main.py                  # Deliverable 1 - Text Reconstruction
│       ├── enhanced_analysis_main.py # Deliverable 2 - Computational Analysis
│       ├── bonus_masked_lm.py       # Bonus - Greek Masked Language Model
│       ├── analysis/                # Analysis and embedding modules
│       │   └── embeddings.py        # Consolidated analysis
│       └── pipelines/               # Reconstruction pipelines
│           ├── custom_reconstructor.py
│           ├── spacy_reconstructor.py
│           └── transformers_reconstructor.py
└── tests/                           # Test modules
```

## Installation

### Prerequisites
- Python 3.12+
- Poetry for dependency management

### Setup
```bash
# Clone the repository
git clone [repository-url]
cd nlp-erg

# Install dependencies
poetry install

# Run the application
poetry run python src/menu.py
```

## ⚙️ Configuration

The project uses environment variables for all settings. All configurations are managed through the `.env` file:

```bash
# Model Settings
T5_MODEL_NAME=t5-small
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
BERT_MODEL_NAME=bert-base-uncased
GREEK_BERT_MODEL=nlpaueb/bert-base-greek-uncased-v1
SPACY_MODEL=en_core_web_sm

# Processing Parameters
MAX_LENGTH=512
TOP_K_PREDICTIONS=5
RANDOM_SEED=42

# Visualization Settings
FIGURE_SIZE=12,8
DPI=300
```

### Configuration Features
- **Centralized Settings:** All models and parameters in one place
- **Environment Variables:** Easy deployment and configuration management
- **Automatic Path Resolution:** No hardcoded file paths
- **Model Fallbacks:** Graceful handling of missing models

## Deliverables

### Deliverable 1: Text Reconstruction

**Objective:** Reconstruct provided texts using multiple automated approaches.

**Implementation:**
- **A. Custom Automaton:** Rule-based reconstructor with grammar corrections
- **B. Three Pipelines:**
  1. **Custom Pipeline:** Manual linguistic rules and pattern matching
  2. **SpaCy Pipeline:** NLP library-based parsing and reconstruction  
  3. **Transformers Pipeline:** T5 model for paraphrasing and grammar correction
- **C. Results Comparison:** Cosine similarity analysis between original and reconstructed texts

**Usage:**
```bash
poetry run python src/menu.py
# Select option 1: Reconstruction Pipelines
```

**Key Features:**
- Processes both assignment texts simultaneously
- Generates reconstructed versions using different methodologies
- Saves outputs to `data/output_texts/` directory
- Computes similarity metrics for quality assessment

### Deliverable 2: Computational Analysis

**Objective:** Analyze semantic shifts using word embeddings and visualization techniques.

**Implementation:**
- **Word Embeddings:** Multiple embedding types (Sentence Transformers, BERT, Word2Vec)
- **Custom NLP Workflows:** Preprocessing, vocabulary analysis, and semantic space mapping
- **Similarity Analysis:** Cosine similarity calculations between original and reconstructed texts
- **Visualizations:** PCA and t-SNE plots showing semantic space transformations

**Usage:**
```bash
poetry run python src/menu.py  
# Select option 2: Computational Analysis
```

**Analysis Components:**
- **Embedding Types:** Sentence-BERT, traditional BERT, Word2Vec document embeddings
- **Similarity Metrics:** Cosine similarity with statistical analysis (mean, std deviation)
- **Visualizations:** Interactive PCA and t-SNE plots showing semantic drift
- **Comparative Analysis:** Side-by-side method performance evaluation

### Deliverable 3: Structured Report

**This README serves as the structured report with the following sections:**

#### Introduction
Semantic reconstruction is crucial for improving text clarity and coherence while preserving original meaning. This project demonstrates the application of modern NLP techniques to automatically enhance text quality through multiple reconstruction strategies. The work addresses challenges in:
- Grammar correction and linguistic enhancement
- Semantic preservation during reconstruction
- Quantitative evaluation of reconstruction quality
- Comparison of traditional vs. transformer-based approaches

#### Methodology

**A. Custom Automaton Strategy:**
- Rule-based pattern matching for common grammatical errors
- Lexical substitution using predefined correction mappings
- Sentence structure optimization through manual linguistic rules

**B. SpaCy Pipeline Strategy:**
- Dependency parsing for syntactic analysis
- Token-level reconstruction with linguistic annotations
- Part-of-speech guided sentence reformulation

**C. Transformers Strategy:**
- T5 model fine-tuned for paraphrasing and grammar correction
- Task-specific prompting for different reconstruction objectives
- Beam search generation for optimal output selection

**Computational Techniques:**
- **Cosine Similarity:** Measures semantic preservation between texts
- **Word Embeddings:** Multiple embedding spaces for comprehensive analysis
- **Dimensionality Reduction:** PCA/t-SNE for semantic space visualization

#### Experiments & Results

**Reconstruction Quality Analysis:**
- **Custom Pipeline:** Mean similarity 0.6551, focused on grammatical corrections
- **SpaCy Pipeline:** Mean similarity 0.6552, similar performance to custom approach  
- **T5-Paraphrase:** Mean similarity 0.4280, more aggressive semantic restructuring
- **T5-Grammar:** Mean similarity 0.6969, balanced grammar correction approach

**Key Findings:**
1. Traditional approaches (Custom, SpaCy) maintain higher semantic similarity
2. Transformer models provide more varied reconstruction styles
3. T5-Grammar achieves optimal balance between enhancement and preservation
4. Visualization reveals distinct clustering patterns for different methods

#### Discussion

**Embedding Performance:**
- Sentence-BERT embeddings effectively captured semantic nuances
- Word2Vec provided additional perspective on lexical relationships
- Multi-embedding analysis revealed consistent reconstruction patterns

**Reconstruction Challenges:**
- Balancing semantic preservation with quality improvement
- Handling domain-specific terminology and context
- Managing trade-offs between creativity and accuracy

**Automation Insights:**
- T5 models enable sophisticated task-specific reconstruction
- Rule-based approaches offer predictable, controlled enhancement
- Hybrid approaches combining multiple methods show promise

**Method Comparison:**
- **Accuracy:** T5-Grammar > Custom ≈ SpaCy > T5-Paraphrase
- **Creativity:** T5-Paraphrase > T5-Grammar > SpaCy > Custom
- **Predictability:** Custom > SpaCy > T5-Grammar > T5-Paraphrase

#### Conclusion
This project successfully demonstrates multiple approaches to automated text reconstruction with quantitative evaluation. The combination of traditional NLP methods and modern transformer models provides comprehensive coverage of reconstruction strategies. Future work could explore fine-tuning domain-specific models and developing hybrid approaches that leverage the strengths of multiple methodologies.

## Bonus: Greek Masked Language Model

**Objective:** Implement masked language modeling using open-source models for Greek legal texts.

**Implementation:**
- **Model:** Greek BERT (`nlpaueb/bert-base-greek-uncased-v1`)
- **Task:** Predict masked words in Greek Civil Code articles
- **Evaluation:** Top-1 and Top-3 accuracy with Greek text normalization
- **Visualization:** Performance analysis and accuracy breakdowns

**Usage:**
```bash
poetry run python src/menu.py
# Select option 3: Greek Masked Language Model
```

**Key Features:**
- Greek text normalization handling accents and diacritics
- Legal domain vocabulary analysis
- Comprehensive performance evaluation
- Advanced visualization of prediction accuracy

**Results:**
- Overall accuracy: 50% (Top-1 and Top-3)
- Successful handling of Greek morphological complexity
- Effective analysis of legal terminology patterns

## Technical Specifications

**Dependencies:**
- `transformers` - Hugging Face transformers for T5 and BERT models
- `sentence-transformers` - Sentence embedding models
- `spacy` - Industrial-strength NLP library
- `scikit-learn` - Machine learning and similarity metrics
- `matplotlib/seaborn` - Visualization and plotting
- `nltk` - Natural language processing toolkit
- `gensim` - Word2Vec implementation
- `torch` - PyTorch deep learning framework

**Key Models:**
- T5-small for text reconstruction
- Sentence-BERT for embeddings
- Greek BERT for masked language modeling
- English spaCy model for linguistic analysis

## Environment Setup

**Configuration:**
- Python 3.12 for optimal package compatibility
- Poetry for reproducible dependency management
- Git version control with appropriate .gitignore
- Environment variables managed through .env files

**Reproduction:**
All experiments are deterministic and reproducible. The Poetry lock file ensures consistent dependency versions across environments.

## Bibliography

1. Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." Journal of Machine Learning Research.
2. Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP 2019.
3. Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019.
4. Honnibal, M., & Montani, I. (2017). "spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing."
5. Koutsikakis, J., et al. (2020). "Greek BERT: The Greeks visiting Sesame Street." 11th Hellenic Conference on Artificial Intelligence.

## License

This project is developed for academic purposes as part of the Natural Language Processing course 2025.

## Author

Efthimis - NLP Assignment 2025