import unittest
import numpy as np
from topic_pipeline.label_generator import LabelGenerator

class TestLabelGenerator(unittest.TestCase):
    
    def setUp(self):
        # We allow it to use real API now as requested, so we ensure enable_llm=True
        # This assumes OPENAI_API_KEY is in .env
        self.label_gen = LabelGenerator(enable_llm=True)

    def test_get_response(self):
        # Test if we get a valid non-empty string response from the live API
        prompt = "Describe 'apple, banana, fruit' in 5 words."
        res = self.label_gen.get_response(prompt)
        self.assertIsInstance(res, str)
        self.assertNotEqual(res.strip(), "")

    def test_get_semantic_label(self):
        # Test the JSON parsing and semantic generation logic with live API
        keystring = "apple, banana, orange, fruit, healthy"
        res = self.label_gen.get_semantic_label(keystring)
        
        self.assertIsInstance(res, str)
        self.assertNotEqual(res, "Topic extraction failed")
        self.assertNotEqual(res.strip(), "")

    def test_get_topics_from_keywords(self):
        cluster_labels = np.array([0, 0, -1])
        all_keywords = ["galaxy", "stars", "outlier_kw"]
        
        labels_dict, keywords_dict = self.label_gen.get_topics_from_keywords(cluster_labels, all_keywords)
        
        self.assertEqual(labels_dict[-1], "Outlier")
        self.assertIsInstance(labels_dict[0], str)
        self.assertNotEqual(labels_dict[0].strip(), "")
        self.assertEqual(keywords_dict["galaxy"], labels_dict[0])
        self.assertEqual(keywords_dict["stars"], labels_dict[0])

if __name__ == '__main__':
    unittest.main()