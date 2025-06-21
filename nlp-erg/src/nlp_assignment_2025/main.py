"""
Main module for text reconstruction pipelines (Deliverable 1).
Implements three different reconstruction approaches and saves results.
"""
from .config import config
from .utils import load_texts_from_file, save_texts_to_file, logger
from .pipelines.custom_reconstructor import CustomReconstructor
from .pipelines.spacy_reconstructor import SpacyReconstructor
from .pipelines.transformers_reconstructor import TransformersReconstructor

def main():
    """Run all reconstruction pipelines and save results."""
    logger.info("🔍 Starting text reconstruction pipelines...")
    
    # Validate configuration and load texts
    if not config.validate_config():
        return
    
    texts = load_texts_from_file(config.INPUT_TEXTS_FILE)
    if not texts:
        logger.error("No texts loaded. Exiting.")
        return
    
    # Initialize pipelines
    logger.info("Initializing reconstruction pipelines...")
    custom_pipeline = CustomReconstructor()
    spacy_pipeline = SpacyReconstructor()
    transformers_pipeline = TransformersReconstructor()
    
    # Process texts with each pipeline
    custom_outputs = []
    spacy_outputs = []
    transformers_outputs = []

    for idx, text in enumerate(texts):
        logger.info(f"Processing text {idx+1}/{len(texts)}")

        # Custom reconstruction
        logger.info("🔵 Custom Pipeline")
        custom_result = custom_pipeline.reconstruct(text)
        custom_outputs.append(custom_result)

        # SpaCy reconstruction  
        logger.info("🟢 SpaCy Pipeline")
        spacy_result = spacy_pipeline.reconstruct(text)
        spacy_outputs.append(spacy_result)

        # Transformers reconstruction
        logger.info("🟣 Transformers Pipeline")
        transformers_result = transformers_pipeline.reconstruct(text)
        transformers_outputs.append(transformers_result)

    # Save all outputs
    logger.info("Saving reconstruction results...")
    save_texts_to_file(custom_outputs, config.get_output_path("custom_pipeline.txt"))
    save_texts_to_file(spacy_outputs, config.get_output_path("spacy_pipeline.txt"))
    save_texts_to_file(transformers_outputs, config.get_output_path("transformers_pipeline.txt"))
    
    logger.info("✅ Text reconstruction completed successfully!")

if __name__ == "__main__":
    main()