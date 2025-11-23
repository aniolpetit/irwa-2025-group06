import json
import math
import sys
import os
from typing import Dict, List, Set, Tuple, Any
from collections import Counter, defaultdict
from array import array

# Add part_2 directory to path to import TF-IDF ranker
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
PART2_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "project_progress", "part_2"))
if PART2_DIR not in sys.path:
    sys.path.append(PART2_DIR)

from inverted_index import InvertedIndex
from tfidf_ranking import TFIDFRanker


class SearchAlgorithm:
    """
    Search algorithm wrapper that integrates TF-IDF ranking for web application use.
    Optimized for performance with pre-initialized index and ranker.
    """
    
    def __init__(self, corpus_data_path: str):
        """
        Initialize the search algorithm with the corpus data.
        Builds inverted index and TF-IDF ranker at initialization for optimal performance.
        
        :param corpus_data_path: Path to the processed corpus JSON file
        """
        self.corpus_data_path = corpus_data_path
        self.corpus_data = self._load_corpus_data()
        self.inverted_index = self._build_inverted_index()
        self.tfidf_ranker = TFIDFRanker(self.inverted_index, self.corpus_data)
        print(f"Search algorithm initialized with {len(self.corpus_data)} documents")
    
    def _load_corpus_data(self) -> List[Dict[str, Any]]:
        """
        Load the processed corpus from JSON file.
        This preserves the tokens field needed for indexing.
        """
        with open(self.corpus_data_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        return corpus
    
    def _build_inverted_index(self) -> InvertedIndex:
        """
        Build the inverted index from the corpus data.
        Uses the 'tokens' field which contains preprocessed tokens.
        """
        index = InvertedIndex()
        index.build_from_corpus(self.corpus_data, text_field='tokens', verbose=False)
        return index
    
    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Perform search using TF-IDF ranking.
        
        :param query: Search query string
        :param top_k: Number of top results to return
        :return: List of (doc_id, score) tuples sorted by relevance
        """
        if not query or not query.strip():
            return []
        
        # Preprocess query: lowercase and split
        query_terms = query.lower().strip().split()
        
        if not query_terms:
            return []
        
        # Perform conjunctive query to find candidate documents
        candidate_docs = self.inverted_index.conjunctive_query(query_terms)
        
        if not candidate_docs:
            return []
        
        # Rank documents using TF-IDF
        ranked_results = self.tfidf_ranker.rank_documents(query_terms, candidate_docs)
        
        # Return top_k results
        return ranked_results[:top_k]
    
    def get_document_by_id(self, doc_id: str) -> Dict[str, Any]:
        """
        Retrieve a document from the corpus by its ID.
        
        :param doc_id: Document ID (pid)
        :return: Document dictionary or None if not found
        """
        for doc in self.corpus_data:
            if doc.get('pid') == doc_id:
                return doc
        return None
