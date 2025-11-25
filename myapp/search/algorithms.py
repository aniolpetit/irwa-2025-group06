import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Add part_2 directory to path to import TF-IDF ranker
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
PART2_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "project_progress", "part_2"))
if PART2_DIR not in sys.path:
    sys.path.append(PART2_DIR)

PART3_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "project_progress", "part_3"))
if PART3_DIR not in sys.path:
    sys.path.append(PART3_DIR)

from inverted_index import InvertedIndex
from tfidf_ranking import TFIDFRanker
from bm25_ranking import BM25Ranker
from word2vec_ranking import Word2VecRanker
from custom_ranking import CustomRanker


class SearchAlgorithm:
    """
    Search algorithm wrapper that integrates TF-IDF ranking for web application use.
    Optimized for performance with pre-initialized index and ranker.
    """

    AVAILABLE_RANKING_METHODS: List[Tuple[str, str]] = [
        ("tfidf", "TF-IDF (cosine)"),
        ("bm25", "BM25"),
        ("word2vec", "Word2Vec (cosine)"),
        ("custom", "Custom hybrid"),
    ]
    DEFAULT_RANKING_METHOD = "tfidf"
    _METHOD_LABEL_MAP = {method: label for method, label in AVAILABLE_RANKING_METHODS}
    
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
        self._ranker_cache: Dict[str, Any] = {"tfidf": self.tfidf_ranker}
        self.word2vec_model_name = os.getenv("WORD2VEC_MODEL_NAME", "glove-wiki-gigaword-100")
        self.word2vec_model_path = os.getenv("WORD2VEC_MODEL_PATH")
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
    
    def search(self, query: str, top_k: int = 20, ranking_method: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Perform search using the selected ranking strategy.
        
        :param query: Search query string
        :param top_k: Number of top results to return
        :param ranking_method: Identifier of the ranking algorithm to use
        :return: List of (doc_id, score) tuples sorted by relevance
        """
        if not query or not query.strip():
            return []
        
        # Preprocess query: lowercase and split
        query_terms = query.lower().strip().split()
        
        if not query_terms:
            return []

        method = (ranking_method or self.DEFAULT_RANKING_METHOD).lower()
        if method not in self._METHOD_LABEL_MAP:
            method = self.DEFAULT_RANKING_METHOD

        # Perform conjunctive query to find candidate documents
        candidate_docs = self.inverted_index.conjunctive_query(query_terms)
        
        if not candidate_docs:
            return []
        
        ranker = self._get_ranker(method)
        ranked_results = ranker.rank_documents(query_terms, candidate_docs)
        
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

    def get_available_methods(self) -> List[Dict[str, str]]:
        return [{"id": method, "label": label} for method, label in self.AVAILABLE_RANKING_METHODS]

    def get_method_label(self, method: Optional[str]) -> str:
        if not method:
            method = self.DEFAULT_RANKING_METHOD
        return self._METHOD_LABEL_MAP.get(method.lower(), self._METHOD_LABEL_MAP[self.DEFAULT_RANKING_METHOD])

    def _get_ranker(self, method: str):
        if method in self._ranker_cache:
            return self._ranker_cache[method]

        if method == "bm25":
            self._ranker_cache[method] = BM25Ranker(self.inverted_index, self.corpus_data, text_field="tokens")
        elif method == "word2vec":
            self._ranker_cache[method] = Word2VecRanker(
                self.inverted_index,
                self.corpus_data,
                text_field="tokens",
                model_name=self.word2vec_model_name,
                model_path=self.word2vec_model_path,
            )
        elif method == "custom":
            self._ranker_cache[method] = CustomRanker(
                self.inverted_index,
                self.corpus_data,
                tfidf_ranker=self.tfidf_ranker,
            )
        else:
            # fall back to default tfidf
            return self.tfidf_ranker

        return self._ranker_cache[method]
