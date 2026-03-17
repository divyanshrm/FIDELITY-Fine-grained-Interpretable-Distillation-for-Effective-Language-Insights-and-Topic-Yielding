import numpy as np
import hdbscan
from sklearn.cluster import DBSCAN, KMeans

class Clusterer:
    """Class holding logic to cluster high-dimensional or reduced text/keyword embeddings."""
    
    def __init__(self, method: str = 'HDBSCAN'):
        self.method = method
        self.clusterer = None

    def clustering(self, embeddings: np.ndarray) -> tuple[object, np.ndarray]:
        """
        Fits a clustering algorithm on the provided embeddings and returns the labels.
        Returns: Tuple of (Fitted Clusterer Object, Array of Cluster Labels)
        """
        if self.method == 'DBSCAN':
            self.clusterer = DBSCAN(eps=0.2, min_samples=15, metric='cosine')
        elif self.method == 'HDBSCAN':
            self.clusterer = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        elif self.method == 'KMeans':
            self.clusterer = KMeans(n_clusters=10, n_init='auto')
        else:
             raise ValueError(f"Unknown clustering method: {self.method}")
            
        cluster_labels = self.clusterer.fit_predict(embeddings)
        return self.clusterer, cluster_labels
