import os
import json
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

class LabelGenerator:
    """Class to prompt LLM APIs to generate semantic topics from clustered keywords."""
    
    def __init__(self, enable_llm: bool = True):
        load_dotenv()
        self.model_name = os.getenv('LLM_MODEL', 'gpt-oss-120b')
        self.base_url = os.getenv('LLM_BASE_URL', 'https://api.ai.it.ufl.edu')
        self.client = None
        
        if enable_llm:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=os.getenv('OPENAI_API_KEY', 'dummy-key')
            )

    def get_response(self, text: str, json_mode: bool = False) -> str:
        """Gets generated string response from the LLM."""
        if not self.client:
            raise RuntimeError("LLM not initialized properly.")

        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0.0,
            "max_tokens": 100
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        return content.strip() if content else ""

    def get_semantic_label(self, keystring: str) -> str:
        """Constructs prompt and parses JSON response from LLM."""
        prompt = (
            "You are a topic modeling expert. Given a list of keywords, generate a single topic label of exactly 5 or 6 words.\n\n"
            "Instructions:\n"
            "- Infer only the most directly supported common theme.\n"
            "- Do not invent actions, intentions, or narrative context.\n"
            "- Keep the label specific, clear, and natural.\n"
            "- Avoid vague phrases like 'various aspects of' or overly creative wording.\n"
            "- Return only JSON in the format: {\"topic\": \"...\"}\n\n"
            
            "Examples:\n"
            "Keywords: galaxy, stars, planet, telescope, astronomical, universe\n"
            "JSON: {\"topic\": \"Astronomy and celestial phenomena\"}\n\n"
            
            "Keywords: politics, election, vote, democracy, government, policy\n"
            "JSON: {\"topic\": \"Electoral politics and public policy\"}\n\n"
            
            f"Keywords: {keystring}\n"
        )
        
        try:
            response_str = self.get_response(prompt, json_mode=True)
            response_data = json.loads(response_str)
            return response_data.get("topic", "Topic extraction failed")
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error parsing LLM response: {e}")
            return "Topic extraction failed"

    def get_topics_from_keywords(self, cluster_labels: np.ndarray, all_keywords: list[str]) -> tuple[dict, dict]:
        """Creates and dispatches prompts to LLM from clustered keyword arrays."""
        label_to_semantic_topic = {-1: "Outlier"}
        keyword_to_semantic_topic = {}
        
        np_keywords = np.asarray(all_keywords)
        unique_labels = np.unique(cluster_labels)

        for label in tqdm(unique_labels, desc="Generating Labels", leave=False):
            if label == -1:
                continue
                
            class_indices = np.where(cluster_labels == label)[0]
            topic_keywords = np_keywords[class_indices]
            
            keystring = ",".join(topic_keywords[:50])
            topic_label = self.get_semantic_label(keystring)
            
            for key in topic_keywords:
                keyword_to_semantic_topic[key] = topic_label
            label_to_semantic_topic[label] = topic_label

        return label_to_semantic_topic, keyword_to_semantic_topic
