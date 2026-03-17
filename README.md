# Fidelity: Topic Detection Pipeline

Fidelity is a high-performance, modular topic modeling pipeline that combines traditional unsupervised clustering with state-of-the-art Large Language Models (LLMs) to generate semantic, human-readable topic labels for large document corpora.

## 📖 Research & Publication

This repository is an implementation of the methodology described in our NAACL 2025 publication:

**[FIDELITY: Fine-grained Interpretable Distillation for Effective Language Insights and Topic Yielding](https://aclanthology.org/2025.findings-naacl.132/)**
*Divyansh Singh, Brodie Mather, Demi Zhang, Patrick Lehman, Justin Ho, Bonnie J Dorr*

> **Abstract:** The rapid expansion of text data has increased the need for effective methods to distill meaningful information from large datasets. Traditional and state-of-the-art approaches have made significant strides in topic modeling, yet they fall short in generating contextually specific and semantically intuitive topics, particularly in dynamic environments and low-resource languages. Additionally, multi-document summarization systems often struggle with issues like redundancy, scalability, and maintaining readability. We introduce FIDELITY (Fine-grained Interpretable Distillation for Effective Language Insights and Topic Yielding), a hybrid method that combines topic modeling and text summarization to produce fine-grained, semantically rich, and contextually relevant output. FIDELITY enhances dataset accessibility and interpretability, outperforming traditional models in topic diversity, similarity, and in the ability to process new, unseen documents. Additionally, it demonstrates robust multilingual capabilities, effectively handling low-resource languages like Tagalog. This makes FIDELITY a powerful tool for distilling and understanding complex textual data, providing detailed insights while maintaining the necessary granularity for practical applications.

## 🚀 Key Features

- **Modular Architecture**: Clean separation of concerns between keyword extraction, embedding, dimension reduction, clustering, and labeling.
- **LLM-Powered Labeling**: Uses `llama-3.1-70b-instruct` via the UF API to generate natural language topic sentences (5-6 words) using few-shot JSON prompting.
- **High-Performance Scaling**: Optimized document-to-topic assignment using vectorized matrix similarity calculations (PyTorch), reducing processing time from minutes to seconds.
- **Real-Time Monitoring**: Granular progress tracking across all pipeline stages via nested `tqdm` progress bars.
- **Remote API Integration**: Offloads heavy embedding and LLM inference to high-performance remote endpoints, keeping local resource usage low.
- **Human-Readable Output**: Automatically saves results to a structured `output_df.csv` for easy analysis and downstream use.

## 📂 Project Structure

```text
fidelity/
├── topic_pipeline/           # Core library
│   ├── fidelity_module.py    # Master orchestrator
│   ├── keyword_extractor.py  # Keyword extraction (KeyBERT)
│   ├── embedder.py           # API-based embeddings
│   ├── dimension_reducer.py  # UMAP/t-SNE reduction
│   ├── clusterer.py          # HDBSCAN/KMeans clustering
│   └── label_generator.py    # LLM-based semantic labeling
├── tests/                    # Unit test suite
├── resources/                # Cached models and output files
├── verify_pipeline.py        # End-to-end verification script
├── requirements.txt          # Core dependencies
└── .env                      # API configuration and keys
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fidelity
   ```

2. **Set up the environment**:
   It is recommended to use a Conda environment:
   ```bash
   conda create -p ./env python=3.10
   conda activate ./env
   pip install -r requirements.txt
   ```

3. **Configure API Keys**:
   Create a `.env` file in the root directory (based on `.env.example`) with your credentials. Fidelity supports any OpenAI-compatible API endpoint:
   ```env
   OPENAI_API_KEY=your_key_here
   LLM_BASE_URL=https://api.your-provider.com/v1
   LLM_MODEL=your-preferred-llm-model
   EMB_MODEL=your-preferred-embedding-model
   ```

## 💻 Usage

### Basic Execution
You can use the `FidelityModule` to process your documents:

```python
from topic_pipeline import FidelityModule

# Sample data
docs_ids = [
    {"doc_id": "1", "text_english": "The satellite was launched into geostationary orbit..."},
    {"doc_id": "2", "text_english": "New quantum computing processors utilize superconducting qubits..."},
]

# Initialize and run
module = FidelityModule(scenario="my_project", enable_llm=True)
output_df = module.resource_building(docs_ids, redo=True)

print(output_df[['Topics', 'Documents']])
```

### End-to-End Verification
To verify the entire pipeline on a subset of the 20 Newsgroups dataset:
```bash
python verify_pipeline.py
```

## 🧪 Testing
Run the unit test suite to ensure individual components are working correctly:
```bash
python -m unittest discover tests
```

## 📄 License
MIT
