import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PART2_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "part_2"))
if PART2_DIR not in sys.path:
    sys.path.append(PART2_DIR)

try:
    from gensim.models import KeyedVectors
    from gensim import downloader
except ImportError:
    print("Warning: gensim not found. Please install it with: pip install gensim")
    KeyedVectors = None
    downloader = None

from inverted_index import InvertedIndex


class Word2VecRanker:
    """
    Word2Vec ranking implementation using averaged word vectors and cosine similarity.
    
    For each document and query:
    1. Get word vectors for each token
    2. Average the word vectors to get a single vector representation
    3. Compute cosine similarity between query vector and document vectors
    4. Rank documents by cosine similarity score
    """
    
    def __init__(
        self,
        inverted_index: InvertedIndex,
        corpus_data: List[Dict],
        text_field: str = "tokens",
        model_name: str = "word2vec-google-news-300",
        model_path: Optional[str] = None,
    ):
        self.index = inverted_index
        self.corpus_data = corpus_data
        self.text_field = text_field
        
        # Load word2vec model
        self.word_vectors = self._load_word2vec_model(model_name, model_path)
        self.vector_dim = self.word_vectors.vector_size if self.word_vectors else None
        
        # Precompute document vectors
        self.doc_vectors: Dict[str, np.ndarray] = self._precompute_document_vectors()
    
    def _load_word2vec_model(self, model_name: str, model_path: Optional[str]) -> Optional[KeyedVectors]:
        if KeyedVectors is None:
            raise ImportError("gensim is required. Install with: pip install gensim")
        
        if model_path and os.path.exists(model_path):
            print(f"Loading word2vec model from {model_path}...")
            try:
                return KeyedVectors.load(model_path)
            except Exception:
                try:
                    return KeyedVectors.load_word2vec_format(model_path, binary=True)
                except Exception:
                    return KeyedVectors.load_word2vec_format(model_path, binary=False)
        else:
            print(f"Downloading/loading pre-trained model: {model_name}...")
            try:
                return downloader.load(model_name)
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                print("Trying to use a smaller model: glove-wiki-gigaword-100")
                try:
                    return downloader.load("glove-wiki-gigaword-100")
                except Exception as e2:
                    print(f"Error loading fallback model: {e2}")
                    raise RuntimeError("Could not load word2vec model. Please ensure gensim is installed.")
    
    def _get_word_vector(self, word: str) -> Optional[np.ndarray]:
        # Get word vector for a given word, return None if word not in vocabulary.
        if not self.word_vectors:
            return None
        try:
            return self.word_vectors[word]
        except KeyError:
            return None
    
    def _average_word_vectors(self, tokens: List[str]) -> Optional[np.ndarray]:        
        # Average word vectors for a list of tokens to create a single text vector.
        if not tokens:
            return None
        
        vectors = []
        for token in tokens:
            vec = self._get_word_vector(token)
            if vec is not None:
                vectors.append(vec)
        
        if not vectors:
            return None
        
        avg_vector = np.mean(vectors, axis=0)
        return avg_vector
    
    def _precompute_document_vectors(self) -> Dict[str, np.ndarray]:
        # Precompute averaged word vectors for all documents.
        doc_vectors: Dict[str, np.ndarray] = {}
        
        print("Precomputing document vectors...")
        for doc in self.corpus_data:
            doc_id = doc.get("pid")
            if not doc_id:
                continue
            
            tokens = doc.get(self.text_field, [])
            if isinstance(tokens, str):
                tokens = tokens.split()
            
            if not tokens:
                continue
            
            doc_vector = self._average_word_vectors(tokens)
            if doc_vector is not None:
                doc_vectors[doc_id] = doc_vector
        
        print(f"Precomputed vectors for {len(doc_vectors)} documents")
        return doc_vectors
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:        
        # Compute cosine similarity between two vectors.

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def rank_documents(
        self, 
        query_terms: List[str], 
        candidate_docs: Set[str]
    ) -> List[Tuple[str, float]]:
        # Rank documents using word2vec cosine similarity.

        if not query_terms or not candidate_docs:
            return []
        
        # Compute query vector by averaging query term vectors
        query_vector = self._average_word_vectors(query_terms)
        
        if query_vector is None:
            # No valid word vectors found for query terms
            return []
        
        # Compute cosine similarity for each candidate document
        scored_docs: List[Tuple[str, float]] = []
        
        for doc_id in candidate_docs:
            doc_vector = self.doc_vectors.get(doc_id)
            
            if doc_vector is None:
                # Document has no valid vector representation
                continue
            
            # Compute cosine similarity
            similarity = self._cosine_similarity(query_vector, doc_vector)
            scored_docs.append((doc_id, similarity))
        
        # Sort by score in descending order
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs
