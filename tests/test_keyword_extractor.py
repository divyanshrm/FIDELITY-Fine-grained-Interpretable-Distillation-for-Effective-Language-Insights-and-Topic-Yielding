import unittest
from unittest.mock import patch, MagicMock
from topic_pipeline.keyword_extractor import KeywordExtractor

class TestKeywordExtractor(unittest.TestCase):
    
    @patch('topic_pipeline.keyword_extractor.Embedder')
    @patch('topic_pipeline.keyword_extractor.KeyBERT')
    def setUp(self, MockKeyBERT, MockEmbedder):
        self.mock_keybert_instance = MagicMock()
        MockKeyBERT.return_value = self.mock_keybert_instance
        self.mock_embedder = MockEmbedder()
        self.extractor = KeywordExtractor(embedder=self.mock_embedder)

    def test_preprocess_clean(self):
        cleaned = self.extractor.preprocess_clean("Hello <p>World</p>! 123")
        self.assertIn("hello", cleaned)
        self.assertIn("world", cleaned)

    def test_get_keywords(self):
        self.mock_keybert_instance.extract_keywords.return_value = [("topic1", 0.9)]
        res = self.extractor.get_keywords("Sample text")
        self.assertEqual(res, [("topic1", 0.9)])

    def test_get_thresholded_keywords(self):
        self.mock_keybert_instance.extract_keywords.return_value = [("topic1", 0.9), ("topic2", 0.1)]
        all_kw, kw_per_doc = self.extractor.get_thresholded_keywords(["Sample 1"])
        
        # dynamic threshold will be 0.9 / 3 = 0.3
        self.assertIn("topic1", all_kw)
        self.assertNotIn("topic2", all_kw)
        self.assertIn("topic1", kw_per_doc[0])

    def test_empty_keyword_list(self):
        self.mock_keybert_instance.extract_keywords.return_value = []
        all_kw, kw_per_doc = self.extractor.get_thresholded_keywords(["Sample empty"])
        self.assertEqual(len(all_kw), 0)
        self.assertEqual(len(kw_per_doc[0]), 0)

if __name__ == '__main__':
    unittest.main()
