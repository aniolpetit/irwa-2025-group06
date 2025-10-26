"""
TF-IDF Ranking Implementation for IRWA Final Project - Part 2
Simple TF-IDF scoring and ranking for search results.
"""

import math
from typing import Dict, List, Tuple, Set
from collections import Counter, defaultdict
from inverted_index import InvertedIndex


class TFIDFRanker:
    """
    Simple TF-IDF ranking implementation.
    Calculates TF-IDF scores and ranks documents by relevance.
    """
    
    def __init__(self, inverted_index: InvertedIndex, corpus_data: List[Dict]):
        self.index = inverted_index
        self.corpus_data = corpus_data
        self.total_documents = len(corpus_data)
        
        # Build term frequency and document frequency mappings
        self.term_frequencies = self.build_term_frequencies()
        self.document_frequencies = self.build_document_frequencies()
        self.normalized_tf = self.build_normalized_tf()
    
    def build_term_frequencies(self) -> Dict[Tuple[str, str], int]:
        """
        Build term frequency mapping from inverted index positions.
        term_freqs[(term, doc_id)] = count of term in doc_id
        """
        term_freqs = {}
        
        # Iterate through all terms in the index
        for term, postings in self.index.term_to_docs.items():
            for posting in postings:
                doc_id = posting[0]
                positions = posting[1]  # array of positions
                # Term frequency is the number of times the term appears (length of positions array)
                term_freqs[(term, doc_id)] = len(positions)
        
        return term_freqs
    
    def build_document_frequencies(self) -> Dict[str, int]:
        """Document frequency is the number of documents containing each term."""
        doc_freqs = {}
        
        for term, postings in self.index.term_to_docs.items():
            # Number of postings = number of documents containing the term
            doc_freqs[term] = len(postings)
        
        return doc_freqs
    
    def build_normalized_tf(self) -> Dict[Tuple[str, str], float]:
        """
        Build normalized term frequency mapping (TF normalized by Euclidean norm).
        Formula: tf(t,d) = freq(t,d) / ||D|| where ||D|| = sqrt(sum of freq²)
        This matches the solution notebook approach.
        """
        normalized_tf = {}
        
        # First, group terms by document
        doc_term_counts = defaultdict(dict)
        for term, postings in self.index.term_to_docs.items():
            for posting in postings:
                doc_id = posting[0]
                positions = posting[1]
                count = len(positions)
                doc_term_counts[doc_id][term] = count
        
        # Calculate normalized TF for each term in each document
        for doc_id, term_counts in doc_term_counts.items():
            # Calculate Euclidean norm for this document
            norm = math.sqrt(sum(freq ** 2 for freq in term_counts.values()))
            
            # Normalize each term frequency
            for term, freq in term_counts.items():
                normalized_tf[(term, doc_id)] = freq / norm if norm > 0 else 0.0
        
        return normalized_tf
    
    def calculate_tf(self, term: str, doc_id: str) -> float:
        """
        Calculate normalized TF using the solution notebook formula.
        Formula: tf(t,d) = freq(t,d) / ||D||
        """
        # Use normalized TF that was precomputed
        return self.normalized_tf.get((term, doc_id), 0.0)
    
    def calculate_idf(self, term: str) -> float:
        df = self.document_frequencies.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(self.total_documents / df)
    
    def calculate_tfidf(self, term: str, doc_id: str) -> float:
        tf = self.calculate_tf(term, doc_id)
        idf = self.calculate_idf(term)
        return tf * idf
    
    def rank_documents(self, query_terms: List[str], candidate_docs: Set[str]) -> List[Tuple[str, float]]:
        """
        Rank documents using cosine similarity (dot product of query and document vectors).
        Matches the solution notebook approach.
        """
        from collections import Counter
        
        # Build query vector with normalized TF
        query_term_counts = Counter(query_terms)
        query_norm = math.sqrt(sum(count ** 2 for count in query_term_counts.values()))
        query_vector = []
        
        # Build document vectors
        doc_vectors = {}
        
        for term_index, term in enumerate(query_terms):
            if term not in self.index.term_to_docs:
                query_vector.append(0.0)
                continue
            
            # Calculate query vector component (normalized TF * IDF)
            query_tf_norm = query_term_counts[term] / query_norm if query_norm > 0 else 0
            query_idf = self.calculate_idf(term)
            query_vector.append(query_tf_norm * query_idf)
            
            # Build document vector components for this term
            postings = self.index.term_to_docs[term]
            for posting in postings:
                doc_id = posting[0]
                if doc_id not in candidate_docs:
                    continue
                
                if doc_id not in doc_vectors:
                    doc_vectors[doc_id] = [0.0] * len(query_terms)
                
                # Calculate TF-IDF for this term in this document
                doc_tf = self.calculate_tf(term, doc_id)
                doc_idf = query_idf  # Same IDF
                doc_vectors[doc_id][term_index] = doc_tf * doc_idf
        
        # Calculate cosine similarity (dot product) for each document
        scored_docs = []
        for doc_id, doc_vector in doc_vectors.items():
            # Compute dot product between query and document vectors
            score = sum(q * d for q, d in zip(query_vector, doc_vector))
            scored_docs.append((doc_id, score))
        
        # Sort by score in descending order
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs


if __name__ == "__main__":
    # Example usage
    from inverted_index import load_processed_corpus, InvertedIndex
    
    # Load corpus and build index
    corpus_path = "../part_1/data/processed_corpus.json"
    corpus = load_processed_corpus(corpus_path)
    
    index = InvertedIndex()
    index.build_from_corpus(corpus)
    
    # Create ranker
    ranker = TFIDFRanker(index, corpus)
    
    # Example query
    query_terms = ["women", "dress"]
    candidate_docs = index.conjunctive_query(query_terms)
    
    print(f"Query: {' '.join(query_terms)}")
    print(f"Found {len(candidate_docs)} candidate documents")
    
    # Rank results
    ranked_results = ranker.rank_documents(query_terms, candidate_docs)
    print(f"\nTop 5 results:")
    for i, (doc_id, score) in enumerate(ranked_results[:5]):
        print(f"{i+1}. Document {doc_id}: {score:.4f}")
