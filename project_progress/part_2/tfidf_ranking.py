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
    TF-IDF ranking implementation using logarithmic TF and cosine similarity.
    
    Formulas (matching the slide):
    - TF: w_tf = 1 + log₂(freq)
    - IDF: w_idf = log₂(N / df) where N = total docs, df = doc frequency
    - TF-IDF: w = (1 + log₂ f) × log₂(N / df)
    - Document length: length(d) = sqrt(Σ w_i²)
    - Score: score = (query_vector · doc_vector) / doc_length
    
    Uses log base 2 for all calculations.
    """
    
    def __init__(self, inverted_index: InvertedIndex, corpus_data: List[Dict]):
        self.index = inverted_index
        self.corpus_data = corpus_data
        self.total_documents = len(corpus_data)
        
        # Build term frequency and document frequency mappings
        self.term_frequencies = self.build_term_frequencies()
        self.document_frequencies = self.build_document_frequencies()
        
        # Build logarithmic TF and document lengths for cosine similarity
        self.log_tf = self.build_log_tf()
        self.doc_lengths = self.build_document_lengths()
    
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
    
    def build_log_tf(self) -> Dict[Tuple[str, str], float]:
        """
        Build logarithmic term frequency mapping.
        Formula: tf(t,d) = 1 + log₂(freq) where freq is raw term frequency
        Matches the slide formula: w_i,j = (1 + log f_i,j) × log (N / df_i)
        """
        log_tf = {}
        
        # Calculate logarithmic TF for each term in each document
        for term, postings in self.index.term_to_docs.items():
            for posting in postings:
                doc_id = posting[0]
                positions = posting[1]
                raw_freq = len(positions)
                
                # Formula: tf = 1 + log₂(freq)
                if raw_freq > 0:
                    # Use natural log and convert: log₂(x) = ln(x) / ln(2)
                    log_tf[(term, doc_id)] = 1.0 + math.log2(raw_freq) if raw_freq > 0 else 0.0
                else:
                    log_tf[(term, doc_id)] = 0.0
        
        return log_tf
    
    def build_document_lengths(self) -> Dict[str, float]:
        """
        Compute document vector lengths for cosine similarity.
        Formula: length(d) = sqrt(Σ w_i²) where w_i are TF-IDF weights
        """
        doc_lengths = {}
        
        # For each document, calculate its vector length
        doc_weights = defaultdict(lambda: [])
        
        for term, postings in self.index.term_to_docs.items():
            # Calculate IDF for this term
            idf = self.calculate_idf(term)
            
            for posting in postings:
                doc_id = posting[0]
                positions = posting[1]
                raw_freq = len(positions)
                
                # Calculate TF-IDF weight: (1 + log₂ freq) × idf
                if raw_freq > 0:
                    tf = 1.0 + math.log2(raw_freq)
                    weight = tf * idf
                    doc_weights[doc_id].append(weight)
        
        # Calculate length for each document
        for doc_id, weights in doc_weights.items():
            # length = sqrt(sum of weights²)
            doc_lengths[doc_id] = math.sqrt(sum(w ** 2 for w in weights))
        
        return doc_lengths
    
    def calculate_tf(self, term: str, doc_id: str) -> float:
        """
        Calculate logarithmic TF using the slide formula.
        Formula: tf(t,d) = 1 + log₂(freq)
        """
        # Use logarithmic TF that was precomputed
        return self.log_tf.get((term, doc_id), 0.0)
    
    def calculate_idf(self, term: str) -> float:
        """
        Calculate IDF using the slide formula with log base 2.
        Formula: idf(t) = log₂(N / df_t)
        """
        df = self.document_frequencies.get(term, 0)
        if df == 0:
            return 0.0
        return math.log2(self.total_documents / df)
    
    def calculate_tfidf(self, term: str, doc_id: str) -> float:
        tf = self.calculate_tf(term, doc_id)
        idf = self.calculate_idf(term)
        return tf * idf
    
    def rank_documents(self, query_terms: List[str], candidate_docs: Set[str]) -> List[Tuple[str, float]]:
        """
        Rank documents using cosine similarity matching the slide formula.
        Formula: score(d, q) = (query_vector · doc_vector) / doc_length
        
        Where:
        - query_vector[i] = (1 + log₂ f_i,q) × log₂(N / df_i)
        - doc_vector[i] = (1 + log₂ f_i,j) × log₂(N / df_i)
        - doc_length = sqrt(Σ w_i²)
        """
        from collections import Counter
        
        # Count query term frequencies
        query_term_counts = Counter(query_terms)
        
        # Build query vector with TF-IDF weights
        query_vector = [0.0] * len(query_terms)
        
        for term_index, term in enumerate(query_terms):
            if term not in self.index.term_to_docs:
                continue
            
            # Query TF: 1 + log₂(freq_in_query)
            query_freq = query_term_counts[term]
            if query_freq > 0:
                query_tf = 1.0 + math.log2(query_freq)
            else:
                query_tf = 0.0
            
            # IDF for this term
            query_idf = self.calculate_idf(term)
            
            # Query vector component: w_i,q = (1 + log₂ f_i,q) × log₂(N / df_i)
            query_vector[term_index] = query_tf * query_idf
        
        # Build document vectors and calculate scores
        scored_docs = []
        
        for doc_id in candidate_docs:
            doc_vector = [0.0] * len(query_terms)
            
            # Build TF-IDF vector for this document
            for term_index, term in enumerate(query_terms):
                if term not in self.index.term_to_docs:
                    continue
                
                # Get TF-IDF weight for this term in this document
                tfidf_weight = self.calculate_tfidf(term, doc_id)
                doc_vector[term_index] = tfidf_weight
            
            # Calculate dot product: query_vector · doc_vector
            dot_product = sum(q * d for q, d in zip(query_vector, doc_vector))
            
            # Get document length (precomputed)
            doc_length = self.doc_lengths.get(doc_id, 1.0)
            
            # Final score: dot product divided by document length
            score = dot_product / doc_length if doc_length > 0 else 0.0
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
    
    # Build a lookup dictionary for document information by pid
    doc_lookup = {doc.get('pid'): doc for doc in corpus if doc.get('pid')}
    
    index = InvertedIndex()
    index.build_from_corpus(corpus)
    
    # Create ranker
    ranker = TFIDFRanker(index, corpus)
    
    # Test queries
    queries = [
        "ecko unl shirt",
        "ecko unlmen shirt round neck",
        "women polo cotton",
        "casual clothes slim fit",
        "biowash innerwear"
    ]
    
    # Process each query
    for query in queries:
        query_terms = query.lower().split()
        candidate_docs = index.conjunctive_query(query_terms)
        
        print(f"Query: {query}")
        print(f"Query terms: {query_terms}")
        
        if len(candidate_docs) == 0:
            print("No documents found matching all query terms.")
            print(f"\n{'='*80}")
            continue
        
        # Rank results
        ranked_results = ranker.rank_documents(query_terms, candidate_docs)
        total_results = len(ranked_results)
        
        print(f"\nTotal results: {total_results}")
        print(f"Showing top {min(20, total_results)} results:\n")
        
        # Display top 20 results with document information
        display_count = min(20, total_results)
        for i, (doc_id, score) in enumerate(ranked_results[:display_count]):
            doc_info = doc_lookup.get(doc_id, {})
            title = doc_info.get('title', 'N/A')
            brand = doc_info.get('brand', 'N/A')
            category = doc_info.get('category', 'N/A')
            sub_category = doc_info.get('sub_category', 'N/A')
            
            # Get description snippet (first 100 chars)
            description = doc_info.get('description', '')
            if description:
                desc_snippet = description[:100] + ('...' if len(description) > 100 else '')
            else:
                desc_snippet = 'N/A'
            
            print(f"{i+1:3d}. [Score: {score:.6f}] | PID: {doc_id}")
            print(f"     Title: {title}")
            print(f"     Brand: {brand} | Category: {category} | Sub-category: {sub_category}")
            print(f"     Description: {desc_snippet}")
            print()
        
        if total_results > 20:
            print(f"... and {total_results - 20} more results (not shown)")
        print(f"\n{'='*80}")
        print()
