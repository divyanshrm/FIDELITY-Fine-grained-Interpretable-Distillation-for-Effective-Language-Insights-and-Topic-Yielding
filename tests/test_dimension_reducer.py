import unittest
import numpy as np
from fidelity.dimension_reducer import DimensionReducer

class TestDimensionReducer(unittest.TestCase):
    
    def setUp(self):
        # Create dummy data: 100 samples, 10 dimensions
        self.dummy_data = np.random.rand(100, 10)

    def test_umap_reduction(self):
        reducer = DimensionReducer(method="UMAP")
        reduced_data, _ = reducer.dimension_reduce(self.dummy_data)
        
        # UMAP is configured to reduce to 5 components in the actual file
        self.assertEqual(reduced_data.shape, (100, 5))

    def test_tsne_reduction(self):
        reducer = DimensionReducer(method="TSNE")
        reduced_data, _ = reducer.dimension_reduce(self.dummy_data)
        
        # TSNE is configured to reduce to 3 components
        self.assertEqual(reduced_data.shape, (100, 3))

    def test_invalid_method(self):
        reducer = DimensionReducer(method="INVALID")
        with self.assertRaises(ValueError):
            reducer.dimension_reduce(self.dummy_data)

if __name__ == '__main__':
    unittest.main()
