import os
import sys

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath('.'))

from topic_pipeline import FidelityModule

def main():
    print("Testing FidelityModule Initialization...")
    
    module = FidelityModule(scenario="test_scenario", enable_llm=True)
    
    from sklearn.datasets import fetch_20newsgroups
    print("Fetching 20 newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    
    import random

    # Build a pool of non-empty documents first
    non_empty_docs = [
        {"doc_id": str(i), "text_english": text}
        for i, text in enumerate(newsgroups.data)
        if text.strip()
    ]

    # Sample 1000 random documents
    dummy_docs = random.sample(non_empty_docs, 1000)
    
    try:
        print("Running resource building...")
        output_df = module.resource_building(dummy_docs, redo=True, threshold=0.85)
        print("Output Dataframe:")
        print(output_df)
        print("-" * 50)
        print("Pipeline executed successfully!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during execution: {e}")

if __name__ == "__main__":
    main()
