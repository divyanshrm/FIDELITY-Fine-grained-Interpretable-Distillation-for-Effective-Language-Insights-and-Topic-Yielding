# FIDELITY: Fine-grained Interpretable Distillation for Effective Language Insights and Topic Yielding

FIDELITY is a high-performance, modular topic modeling pipeline designed to bridge the gap between traditional unsupervised clustering and modern Large Language Models (LLMs). It processes large document corpora to generate semantic, human-readable topic labels that are more intuitive and contextually specific than those produced by standard LDA or BERTopic approaches.

## Research and Publication

This repository provides the reference implementation for the methodology detailed in our NAACL 2025 publication:

**[FIDELITY: Fine-grained Interpretable Distillation for Effective Language Insights and Topic Yielding](https://aclanthology.org/2025.findings-naacl.132/)**  
*Divyansh Singh, Brodie Mather, Demi Zhang, Patrick Lehman, Justin Ho, Bonnie J Dorr*

### Abstract
The rapid expansion of text data has increased the need for effective methods to distill meaningful information from large datasets. Traditional and state-of-the-art approaches have made significant strides in topic modeling, yet they fall short in generating contextually specific and semantically intuitive topics, particularly in dynamic environments and low-resource languages. FIDELITY introduces a hybrid method that combines topic modeling and text summarization to produce fine-grained, semantically rich output. It outperforms traditional models in topic diversity, similarity, and the ability to process new, unseen documents, while demonstrating robust multilingual capabilities.

## Technical Methodology

FIDELITY operates through a sophisticated Five-Stage Pipeline:

1.  **Keyword Extraction**: Utilizing KeyBERT and specialized preprocessing, the system identifies the most salient terms within each document.
2.  **Multimodal Embedding**: Documents and extracted keywords are transformed into high-dimensional vector spaces using OpenAI-compatible APIs or local Sentence-Transformer models.
3.  **Dimensionality Reduction**: UMAP (Uniform Manifold Approximation and Projection) is employed to project high-dimensional embeddings into a lower-dimensional space while preserving local and global structures.
4.  **Density-Based Clustering**: HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) identifies naturally occurring clusters of concepts within the reduced vector space.
5.  **LLM-Augmented Labeling**: Cluster centroids and representative keywords are passed to a Large Language Model (e.g., Llama 3.1) to synthesize concise, human-readable semantic descriptors (typically 5-6 words).

## Key Features

*   **Modular Architecture**: Each component (Extraction, Embedding, Reduction, Clustering, Labeling) is decoupled, allowing for easy experimentation and replacement of specific sub-modules.
*   **Vectorized Assignment Engine**: Employs PyTorch-accelerated matrix similarity calculations to assign documents to topics, achieving significant performance gains over iterative methods (processing time reduced from minutes to seconds).
*   **LLM Integration**: Leverages few-shot JSON prompting techniques to ensure consistent and high-quality topic generation from instructions-tuned models.
*   **Robust Monitoring**: Comprehensive logging and nested progress tracking via `tqdm` provide transparency into every stage of the pipeline.
*   **Hybrid Connectivity**: Specifically designed to offload compute-intensive tasks (Embedding and LLM inference) to remote high-performance computing (HPC) endpoints.
*   **Persistence and Caching**: Full state serialization (pickling) allows for pausing, resuming, and reloading pipeline resources without re-computation.

## Project Structure

```text
fidelity/
├── fidelity/                 # Core library implementation
│   ├── fidelity_module.py    # Main orchestrator managing state and data flow
│   ├── keyword_extractor.py  # Advanced NLP cleaning and KeyBERT extraction
│   ├── embedder.py           # Interface for remote and local embedding models
│   ├── dimension_reducer.py  # UMAP configuration and transformation logic
│   ├── clusterer.py          # HDBSCAN implementation for concept grouping
│   └── label_generator.py    # Prompt engineering for LLM-based labeling
├── tests/                    # Comprehensive unit test suite for all modules
├── resources/                # Storage for cached models, pkl files, and output CSVs
├── verify_pipeline.py        # End-to-end integration test using Newsgroups dataset
├── requirements.txt          # Python dependency specifications
└── .env                      # Environment configuration (API keys and URLs)
```

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd fidelity
```

### 2. Environment Setup
We recommend using a dedicated Conda or virtual environment (Python 3.10+):
```bash
conda create -p ./env python=3.10
conda activate ./env
pip install -r requirements.txt
```

### 3. API Configuration
Configure your connection to an OpenAI-compatible API provider by creating a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_secure_api_key
LLM_BASE_URL=https://api.your-provider.com/v1
LLM_MODEL=llama-3.1-70b-instruct
EMB_MODEL=text-embedding-3-small
```

## Usage and Examples

### Standard Pipeline Execution
The `FidelityModule` provides a unified interface for processing document collections.

```python
from fidelity import FidelityModule

# Prepare data: Format as a list of dictionaries with 'doc_id' and 'text_english'
docs_ids = [
    {"doc_id": "001", "text_english": "The satellite mission successfully entered lunar orbit..."},
    {"doc_id": "002", "text_english": "Advances in semiconductor manufacturing reduce qubit decoherence..."},
]

# Initialize module with a specific scenario name for resource caching
module = FidelityModule(scenario="aerospace_analysis", enable_llm=True)

# Run the full pipeline
# redo=True forces re-computation even if cached resources exist
output_df = module.resource_building(docs_ids, redo=True, threshold=0.85)

print(output_df[['Topics', 'Number of Documents']])
```

### Predicting Topics for New Documents
Once resources are built and cached, you can predict topics for unseen data without re-clustering:

```python
new_doc = {"doc_id": "999", "text_english": "New propulsion systems allow faster deep space travel."}
prediction = module.predict(new_doc)
print(prediction['Topics'].tolist())
```

### End-to-End Verification
To validate the installation and pipeline on a standard benchmark dataset:
```bash
python verify_pipeline.py
```

## Configuration Details

The pipeline behavior is controlled by several key parameters:
- **`scenario`**: A unique identifier used to create sub-directories in `resources/` for caching models.
- **`threshold`**: (Default: 0.85) Similarity threshold used during topic "collapsing" to merge highly similar semantic labels.
- **`enable_llm`**: Set to `False` if you only want to extract keywords and cluster without generating natural language labels.

## License
This project is licensed under the MIT License.
