import numpy as np
import umap
from sklearn.manifold import TSNE

class DimensionReducer:
    """Class to lower the dimensionality of dense embedding vectors for better clustering."""
    
    def __init__(self, method: str = "UMAP"):
        self.method = method
        self.reducer = None

    def dimension_reduce(self, word_values: np.ndarray) -> tuple[np.ndarray, object]:
        """
        Reduces the dimensions of embeddings using either UMAP or t-SNE algorithms.
        Returns: Tuple of (Reduced Output Array, Fitted Reducer Object)
        """
        if self.method == "TSNE":
            self.reducer = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3)
            out = self.reducer.fit_transform(word_values)
        elif self.method == "UMAP":
            self.reducer = umap.UMAP(n_neighbors=25, n_components=5, metric='cosine', min_dist=0.0)
            out = self.reducer.fit_transform(word_values)
        else:
            raise ValueError(f"Unknown reduction method: {self.method}")
            
        return out, self.reducer
