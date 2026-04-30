import unittest
from unittest.mock import patch, MagicMock
from fidelity.embedder import Embedder
import numpy as np
import torch

class TestEmbedder(unittest.TestCase):
    
    @patch('fidelity.embedder.OpenAI')
    @patch('fidelity.embedder.load_dotenv')
    def setUp(self, mock_load_dotenv, MockOpenAI):
        self.mock_openai_instance = MagicMock()
        MockOpenAI.return_value = self.mock_openai_instance
        self.embedder = Embedder(model_name="test-model")

    def test_find_embeddings_using_transformers_string(self):
        # Setup mock response
        mock_response = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = [0.1, 0.2, 0.3]
        mock_response.data = [mock_item]
        self.mock_openai_instance.embeddings.create.return_value = mock_response

        emb = self.embedder.find_embeddings_using_transformers("test string")
        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (3,))

    def test_find_embeddings_using_transformers_list(self):
        # Setup mock response for list
        mock_response = MagicMock()
        mock_item1 = MagicMock()
        mock_item1.embedding = [0.1, 0.2, 0.3]
        mock_item2 = MagicMock()
        mock_item2.embedding = [0.4, 0.5, 0.6]
        mock_response.data = [mock_item1, mock_item2]
        self.mock_openai_instance.embeddings.create.return_value = mock_response

        emb = self.embedder.find_embeddings_using_transformers(["test 1", "test 2"])
        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (2, 3))

    def test_empty_input(self):
        emb = self.embedder.find_embeddings_using_transformers([])
        self.assertEqual(len(emb), 0)

    def test_get_vector_similarity(self):
        v1 = torch.tensor([1.0, 0.0])
        v2 = torch.tensor([1.0, 0.0])
        score = self.embedder.get_vector_similarity(v1, v2)
        # Cosine similarity of identical vectors is 1
        self.assertAlmostEqual(score, 1.0, places=4)
        
        v3 = torch.tensor([-1.0, 0.0])
        score_opp = self.embedder.get_vector_similarity(v1, v3)
        self.assertAlmostEqual(score_opp, -1.0, places=4)

if __name__ == '__main__':
    unittest.main()
