"""
TF-IDF Ranking Implementation for IRWA Final Project - Part 2
Simple TF-IDF scoring and ranking for search results.
"""

import math
from typing import Dict, List, Tuple, Set
from collections import Counter
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
        
        # Build term frequency mapping
        self.term_frequencies = self.build_term_frequencies()
        self.document_frequencies = self.build_document_frequencies()
    
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
    
    def calculate_tf(self, term: str, doc_id: str) -> float:
        raw_freq = self.term_frequencies.get((term, doc_id), 0)
        if raw_freq == 0:
            return 0.0
        return 1.0 + math.log(raw_freq)
    
    def calculate_idf(self, term: str) -> float:
        df = self.document_frequencies.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(self.total_documents / df)
    
    def calculate_tfidf(self, term: str, doc_id: str) -> float:
        tf = self.calculate_tf(term, doc_id)
        idf = self.calculate_idf(term)
        return tf * idf
    
    def calculate_document_score(self, query_terms: List[str], doc_id: str) -> float:
        """Calculate TF-IDF score for a document given query terms."""
        total_score = 0.0
        
        for term in query_terms:
            # Check if term exists in index
            if term in self.index.term_to_docs:
                # Check if this document contains the term by examining postings
                postings = self.index.term_to_docs[term]
                if any(posting[0] == doc_id for posting in postings):
                    tfidf_score = self.calculate_tfidf(term, doc_id)
                    total_score += tfidf_score
        
        return total_score
    
    def rank_documents(self, query_terms: List[str], candidate_docs: Set[str]) -> List[Tuple[str, float]]:
        scored_docs = []
        
        for doc_id in candidate_docs:
            score = self.calculate_document_score(query_terms, doc_id)
            if score > 0:  # Only include documents with non-zero scores
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
