import unittest
import numpy as np
from fidelity.clusterer import Clusterer

class TestClusterer(unittest.TestCase):
    
    def setUp(self):
        # Create dummy reduced data: 100 samples, 5 dimensions
        self.dummy_data = np.random.rand(100, 5)

    def test_hdbscan_clustering(self):
        clusterer = Clusterer(method="HDBSCAN")
        _, labels = clusterer.clustering(self.dummy_data)
        
        self.assertEqual(labels.shape, (100,))
        # HDBSCAN returns numpy array

    def test_dbscan_clustering(self):
        clusterer = Clusterer(method="DBSCAN")
        _, labels = clusterer.clustering(self.dummy_data)
        
        self.assertEqual(labels.shape, (100,))

    def test_kmeans_clustering(self):
        clusterer = Clusterer(method="KMeans")
        _, labels = clusterer.clustering(self.dummy_data)
        
        self.assertEqual(labels.shape, (100,))
        # KMeans finds 10 clusters as configured
        self.assertTrue(len(np.unique(labels)) <= 10)

    def test_invalid_method(self):
        clusterer = Clusterer(method="INVALID")
        with self.assertRaises(ValueError):
            clusterer.clustering(self.dummy_data)

if __name__ == '__main__':
    unittest.main()
