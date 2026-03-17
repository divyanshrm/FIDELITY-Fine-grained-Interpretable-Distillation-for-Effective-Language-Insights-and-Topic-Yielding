import os
import ast
import pickle
import pandas as pd
import numpy as np
import torch
import hdbscan
from collections import defaultdict
from tqdm import tqdm

from .keyword_extractor import KeywordExtractor
from .embedder import Embedder
from .dimension_reducer import DimensionReducer
from .clusterer import Clusterer
from .label_generator import LabelGenerator


class FidelityModule:
    """Master class orchestrating the topic detection pipeline."""

    def __init__(self, resource_path: str = "resources", scenario: str = "lith", enable_llm: bool = True):
        """
        Initializes the Fidelity Module and tracking resources.
        """
        # Set up resource directories
        os.makedirs(resource_path, exist_ok=True)
        self.resource_path = os.path.join(resource_path, scenario)
        os.makedirs(self.resource_path, exist_ok=True)
        self.scenario = scenario

        # Initialize sub-modules
        self.embedder = Embedder()
        self.keyword_extractor = KeywordExtractor(embedder=self.embedder)
        self.dimension_reducer = DimensionReducer()
        self.clusterer = Clusterer()
        self.label_generator = LabelGenerator(enable_llm=enable_llm)

        # Stateful resources
        self.topic_to_keywords = {}
        self.label_to_semantic_topic = {}
        self.output_df = pd.DataFrame()
        self.mapping = {}
        self.resource_built = self.check_resources()

    @staticmethod
    def _create_index_to_id_mapping(ids: list) -> dict:
        """Creates a mapping from dataframe list indices back to original document IDs."""
        return {idx: doc_id for idx, doc_id in enumerate(ids)}

    def _assign_topics_to_documents(self, keywords_per_doc: list, keyword_to_semantic_topic: dict,
                                    doc_embeddings: np.ndarray, threshold: float) -> list:
        """Assigns matching semantic topics to each document efficiently using matrix similarity."""
        all_topics = []
        
        # Pre-embed unique topics
        unique_topics = list(set(keyword_to_semantic_topic.values()))
        if not unique_topics:
            return [[] for _ in range(len(keywords_per_doc))]

        topic_embs = self.embedder.embed(unique_topics)
        
        # Convert to torch for fast matrix operations
        doc_tensor = torch.from_numpy(doc_embeddings).float()
        topic_tensor = torch.from_numpy(topic_embs).float()

        # Normalize for cosine similarity via dot product
        doc_norm = torch.nn.functional.normalize(doc_tensor, p=2, dim=1)
        topic_norm = torch.nn.functional.normalize(topic_tensor, p=2, dim=1)
        
        # Matrix of similarities: [num_docs, num_topics]
        similarities = torch.mm(doc_norm, topic_norm.t())
        
        # Map topic index to its semantic name
        topic_idx_to_name = {idx: name for idx, name in enumerate(unique_topics)}

        for i, doc_keywords in enumerate(keywords_per_doc):
            topics = []
            # First, check topics associated with keywords in this doc
            candidate_topics = set()
            for kw in doc_keywords:
                if kw in keyword_to_semantic_topic:
                    candidate_topics.add(keyword_to_semantic_topic[kw])
            
            # For each candidate topic, check if similarity is above threshold
            for topic in candidate_topics:
                # Find index of this topic in our similarity matrix
                topic_idx = unique_topics.index(topic)
                
                topics.append(topic)
                    
            all_topics.append(topics)
        return all_topics

    def _build_output_dataframe(self, all_topics: list, keywords_per_doc: list, text: list, 
                                index_to_id_mapping: dict, all_keywords: list, 
                                keyword_to_semantic_topic: dict):
        """Constructs the initial output dataframe associating concepts with documents."""
        # Map topics back to all supporting keywords
        topic_key = defaultdict(list)
        for kw in all_keywords:
            if kw in keyword_to_semantic_topic:
                semantic_topic = keyword_to_semantic_topic[kw]
                if kw not in topic_key[semantic_topic]:
                    topic_key[semantic_topic].append(kw)

        # Map each topic to the documents that mention it
        topic_doc = defaultdict(list)
        for x, topics in enumerate(all_topics):
            for topic in topics:
                topic_doc[topic].append(index_to_id_mapping[x])

        output_df = pd.DataFrame({
            'Topics': list(topic_doc.keys()),
            'Documents': list(topic_doc.values())
        })
        
        output_df['Number of Documents'] = output_df['Documents'].apply(len)
        output_df['Topic_Cluster'] = output_df['Topics'].apply(lambda t: topic_key.get(t, []))
        
        # Sort by popularity
        output_df = output_df.sort_values('Number of Documents', ascending=False).reset_index(drop=True)
        return output_df[['Documents', 'Topics', 'Topic_Cluster']], dict(topic_key)

    @staticmethod
    def _merge_duplicate_topics(df: pd.DataFrame) -> pd.DataFrame:
        """Concatenates lists in duplicate topic rows and dedupes the documents."""
        def concat_lists(series):
            concatenated = []
            for sublist in series:
                if isinstance(sublist, list):
                    concatenated.extend(sublist)
            return concatenated

        # Group by topic and concatenate all columns (like Documents and Topic_Cluster)
        grouped_df = df.groupby('Topics').agg({
            col: concat_lists for col in df.columns if col != 'Topics'
        }).reset_index()

        # Deduplicate the document lists and sort by frequency
        for idx in range(len(grouped_df)):
            grouped_df.at[idx, 'Documents'] = list(np.unique(grouped_df.at[idx, 'Documents']))

        sorted_df = grouped_df.iloc[grouped_df['Documents'].apply(len).argsort()[::-1]].reset_index(drop=True)
        return sorted_df[['Documents', 'Topics', 'Topic_Cluster']]

    def _group_similar_sentences(self, sentences: list, threshold: float = 0.85) -> list:
        """Groups similar sentences/topics together based on their embeddings."""
        groups = []
        grouped_indices = set()
        embeddings = torch.tensor(self.embedder.find_embeddings_using_transformers(sentences))

        for i in range(len(sentences)):
            if i not in grouped_indices:
                similar_group = [sentences[i]]
                for j in range(i + 1, len(sentences)):
                    if j not in grouped_indices:
                        score = self.embedder.get_vector_similarity(embeddings[i], embeddings[j])
                        if score >= threshold:
                            similar_group.append(sentences[j])
                            grouped_indices.add(j)
                groups.append(similar_group)
        return groups

    def _collapse_similar_topics(self, df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, dict]:
        """Merges highly similar topics into a single parent topic."""
        if threshold >= 1.0:
            return df, {topic: topic for topic in df.Topics}

        df_copy = df.copy()
        sentences = df_copy.Topics.tolist()
        
        similar_groups = self._group_similar_sentences(sentences, threshold)
        
        mapping = {}
        for group in similar_groups:
            parent_topic = group[0]
            for topic in group:
                mapping[topic] = parent_topic

        df_copy['Topics'] = df_copy['Topics'].map(lambda t: mapping.get(t, t))
        merged_df = self._merge_duplicate_topics(df_copy)
        
        return merged_df, mapping

    def topic_modelling(self, docs: list, ids: list, threshold: float):
        """Executes the mathematical pipeline to cluster documents into semantic topics."""
        with tqdm(total=7, desc="Pipeline Progress") as pbar:
            pbar.set_postfix_str("Preprocessing text")
            text = [' '.join(self.keyword_extractor.preprocess_clean(str(doc))) for doc in tqdm(docs, desc="Preprocessing", leave=False)]
            index_to_id_mapping = self._create_index_to_id_mapping(ids)
            pbar.update(1)

            pbar.set_postfix_str("Embedding documents")
            doc_embeddings = self.embedder.find_embeddings_using_transformers(text)
            pbar.update(1)

            pbar.set_postfix_str("Extracting keywords")
            all_keywords, keywords_per_doc = self.keyword_extractor.get_thresholded_keywords(text)
            pbar.update(1)

            pbar.set_postfix_str("Embedding keywords")
            all_keywords_list = list(all_keywords)
            _, word_values = self.embedder.embed_keywords(all_keywords_list)
            pbar.update(1)

            pbar.set_postfix_str("Reducing dimensions & clustering")
            reduced_embeddings, _ = self.dimension_reducer.dimension_reduce(np.asarray(word_values))
            _, cluster_labels = self.clusterer.clustering(np.array(reduced_embeddings))
            pbar.update(1)

            pbar.set_postfix_str("Generating topics with LLM")
            label_to_semantic, keyword_to_semantic = self.label_generator.get_topics_from_keywords(
                cluster_labels, all_keywords_list
            )
            pbar.update(1)

            pbar.set_postfix_str("Assigning topics to documents")
            all_topics = self._assign_topics_to_documents(
                keywords_per_doc, keyword_to_semantic, doc_embeddings, 0.0
            )
            odf, topic_key = self._build_output_dataframe(
                all_topics, keywords_per_doc, text, index_to_id_mapping, all_keywords_list, keyword_to_semantic
            )
            pbar.update(1)

        return (odf, topic_key), label_to_semantic

    def resource_building(self, docs_ids: list[dict], redo: bool = False, threshold: float = 0.85) -> pd.DataFrame:
        """Main entry point to process a corpus and build cluster resources."""
        if self.resource_built and not redo and self.check_resources():
            print("Loading cached resources...")
            self.load_resources()
            # Re-apply mappings to current dataframe
            df_mapped = self.output_df.copy()
            df_mapped['Topics'] = df_mapped['Topics'].map(lambda t: self.mapping.get(t, t))
            odf = self._merge_duplicate_topics(df_mapped)
            odf['Scenario'] = self.scenario
            return odf

        print("Building resources from scratch...")
        docs = [d['text_english'] for d in docs_ids]
        ids = [d['doc_id'] for d in docs_ids]

        (output_df, self.topic_to_keywords), self.label_to_semantic_topic = self.topic_modelling(docs, ids, threshold)
        self.output_df = output_df
        self.resource_built = True
        
        odf, self.mapping = self._collapse_similar_topics(self.output_df, threshold)
        self.save_resources()
        
        odf['Scenario'] = self.scenario
        return odf

    def predict(self, docs_ids: dict, threshold: float = 0.85) -> pd.DataFrame:
        """Predicts topics for a single new document using the saved clusters."""
        if not self.resource_built:
            raise Exception("Resources have not been built.")

        doc_id = docs_ids['doc_id']
        doc = docs_ids['text_english']

        text = [' '.join(self.keyword_extractor.preprocess_clean(str(doc)))]
        all_keywords, _ = self.keyword_extractor.get_thresholded_keywords(text)

        if not all_keywords:
            return pd.DataFrame()

        _, word_values = self.embedder.embed_keywords(list(all_keywords))
        reduced_values = self.dimension_reducer.reducer.transform(word_values)
        
        detected_topics = set()
        labels, _ = hdbscan.approximate_predict(self.clusterer.clusterer, reduced_values)
        
        for label in labels:
            if label != -1 and label in self.label_to_semantic_topic:
                temp_topic = self.label_to_semantic_topic[label]
                
                # Compare snippet embedding directly to topic string embeding
                topic_emb = torch.tensor(self.embedder.embed(temp_topic))
                text_emb = torch.tensor(self.embedder.embed(text[0]))
                score = self.embedder.get_vector_similarity(topic_emb, text_emb)
                
                if score >= 0.4 and temp_topic in self.mapping:
                    detected_topics.add(self.mapping[temp_topic])

        all_topics = list(detected_topics)
        keywords_per_topic = [self.topic_to_keywords.get(t, []) for t in all_topics]

        output_df = pd.DataFrame({
            'Documents': [[doc_id]] * len(all_topics),
            'Topics': all_topics,
            'Topic_Cluster': keywords_per_topic
        })
        
        return self._merge_duplicate_topics(output_df)

    def _get_resource_paths(self) -> dict:
        """Returns the file paths for pickled components and CSV data."""
        return {
            'topic_to_keywords': os.path.join(self.resource_path, 'topic_to_keywords.pkl'),
            'clusterer': os.path.join(self.resource_path, 'clusterer.pkl'),
            'umap_module': os.path.join(self.resource_path, 'umap_module.pkl'),
            'label_to_semantic_topic': os.path.join(self.resource_path, 'label_to_semantic_topic.pkl'),
            'output_df': os.path.join(self.resource_path, 'output_df.csv'),
            'mapping': os.path.join(self.resource_path, 'mapping.pkl')
        }

    def save_resources(self):
        """Pickles instances and saves output_df to CSV."""
        if not self.resource_built:
            raise Exception("Resources have not been built.")

        paths = self._get_resource_paths()
        
        with open(paths['clusterer'], 'wb') as f:
            pickle.dump(self.clusterer.clusterer, f)
        with open(paths['topic_to_keywords'], 'wb') as f:
            pickle.dump(self.topic_to_keywords, f)
        with open(paths['umap_module'], 'wb') as f:
            pickle.dump(self.dimension_reducer.reducer, f)
        with open(paths['label_to_semantic_topic'], 'wb') as f:
            pickle.dump(self.label_to_semantic_topic, f)
        # Convert lists to comma-separated strings inside CSV for easy readability
        save_df = self.output_df.copy()
        for col in ['Documents', 'Topic_Cluster']:
            if col in save_df.columns:
                save_df[col] = save_df[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)

        save_df.to_csv(paths['output_df'], index=False)

    def check_resources(self) -> bool:
        """Verifies that all required cache files exist in the resource directory."""
        paths = self._get_resource_paths()
        res_dir_exists = os.path.isdir(self.resource_path)
        
        files_to_check = ['topic_to_keywords', 'clusterer', 'umap_module', 'label_to_semantic_topic', 'output_df']
        all_main_files = all(os.path.isfile(paths[key]) for key in files_to_check)
        mapping_exists = os.path.isfile(paths['mapping'])

        # Legacy correction map
        if all_main_files and not mapping_exists:
            df = pd.read_csv(paths['output_df'])
            mapping = {t: t for t in df['Topics'].unique()}
            with open(paths['mapping'], 'wb') as f:
                pickle.dump(mapping, f)

        self.resource_built = res_dir_exists and all_main_files
        return self.resource_built

    def load_resources(self):
        """Loads cached models and parsed the CSV list columns safely."""
        if not self.check_resources() or not self.resource_built:
            raise Exception("Resources are incomplete or missing.")

        paths = self._get_resource_paths()
        
        with open(paths['topic_to_keywords'], 'rb') as f:
            self.topic_to_keywords = pickle.load(f)
        with open(paths['clusterer'], 'rb') as f:
            self.clusterer.clusterer = pickle.load(f)
        with open(paths['umap_module'], 'rb') as f:
            self.dimension_reducer.reducer = pickle.load(f)
        with open(paths['label_to_semantic_topic'], 'rb') as f:
            self.label_to_semantic_topic = pickle.load(f)
        with open(paths['mapping'], 'rb') as f:
            self.mapping = pickle.load(f)

        # Parse comma-separated strings from CSV back into valid lists
        df = pd.read_csv(paths['output_df'])
        for col in ['Documents', 'Topic_Cluster']:
            if col in df.columns:
                # Handle cases where the string might be empty or pandas parsed it as NaN
                df[col] = df[col].astype(str).apply(
                    lambda x: [item.strip() for item in x.split(',')] if x and x.lower() != 'nan' else []
                )
                
        self.output_df = df

    def collapse_topics(self, threshold: float) -> pd.DataFrame:
        """Forces an explicit re-collapse of the dataframe topics using a generic threshold."""
        df, self.mapping = self._collapse_similar_topics(self.output_df, threshold)
        self.save_resources()
        return df
