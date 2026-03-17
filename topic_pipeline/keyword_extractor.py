from tqdm import tqdm
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags
from keybert import KeyBERT
from .embedder import Embedder

class CustomAPIBackend:
    """Wrapper to forward KeyBERT embeddings into the remote NLP API."""
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        
    def embed(self, documents: list[str], verbose: bool = False):
        return self.embedder.find_embeddings_using_transformers(documents)

class KeywordExtractor:
    """Class to manage data preprocessing and KeyBERT token topic extraction."""
    
    def __init__(self, embedder: Embedder | None = None):
        if embedder is None:
            embedder = Embedder()
        self.kw_model = KeyBERT(model=CustomAPIBackend(embedder))

    @staticmethod
    def preprocess_clean(document: str) -> list[str]:
        """Strips HTML tags and removes accents from an input string."""
        return simple_preprocess(strip_tags(document), deacc=True)

    def get_keywords(self, text: str, ngram_range: int = 1, top: int = 5) -> list[tuple[str, float]]:
        """Extracts significant keywords from text using max marginal relevance."""
        topic_entities = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, ngram_range),
            stop_words='english',
            top_n=top,
            use_mmr=True,
            diversity=0.4
        )
        return topic_entities

    def get_thresholded_keywords(self, text_list: list[str], threshold: float = 0.0) -> tuple[set[str], list[list[str]]]:
        """Gets keywords passing a dynamic scoring threshold per document."""
        all_keywords = set()
        keywords_per_doc = []

        for sample in tqdm(text_list, desc="Extracting Keywords", leave=False):
            topics = self.get_keywords(sample, ngram_range=3, top=10)
            if not topics:
                keywords_per_doc.append([])
                continue
                
            scores = [score for _, score in topics]
            dynamic_threshold = max(scores) / 3 if scores else threshold
            
            valid_topics = [kw for kw, score in topics if score > dynamic_threshold]
            
            all_keywords.update(valid_topics)
            keywords_per_doc.append(valid_topics)

        return all_keywords, keywords_per_doc
