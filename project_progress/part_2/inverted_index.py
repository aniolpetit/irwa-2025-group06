import json
from collections import defaultdict
from typing import Dict, List, Set, Any


class InvertedIndex:
    """
    Simple inverted index for conjunctive queries (AND operations).
    Maps terms to lists of documents containing those terms.
    """
    
    def __init__(self):
        self.term_to_docs: Dict[str, Set[str]] = defaultdict(set) # by now we keep a single index for all terms combined, we could consider creating a separate index for each field (title tokens, brand tokens, etc.)
        self.total_documents: int = 0
    
    def add_document(self, doc_id: str, tokens: List[str]) -> None:
        if not tokens:
            return
            
        # Add document to each term's posting list
        for term in tokens: # right now it adds the document only once for each term regardless of how many times it appears in the document, maybe we should add it multiple times? We handle this in the TF-IDF ranking though.
            self.term_to_docs[term].add(doc_id)
        
        self.total_documents += 1
    
    def build_from_corpus(self, corpus_data: List[Dict[str, Any]], 
                         text_field: str = 'tokens') -> None:

        print(f"Building inverted index from {len(corpus_data)} documents...")
        
        for doc in corpus_data:
            doc_id = doc.get('pid', '')
            if not doc_id:
                continue
                
            tokens = doc.get(text_field, [])
            if isinstance(tokens, str):
                tokens = tokens.split()
            
            self.add_document(doc_id, tokens)
        
        print(f"Index built successfully. Vocabulary size: {len(self.term_to_docs)}")
    
    def conjunctive_query(self, terms: List[str]) -> Set[str]:
        if not terms:
            return set()
        
        # Start with documents containing the first term
        result = set(self.term_to_docs.get(terms[0], set()))
        
        # Intersect with documents containing each subsequent term
        for term in terms[1:]:
            term_docs = set(self.term_to_docs.get(term, set()))
            result = result.intersection(term_docs)
            
            # Early termination if no documents match
            if not result:
                break
        
        return result
    
    def get_documents_for_term(self, term: str) -> Set[str]:
        return set(self.term_to_docs.get(term, set()))
    
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        return {
            'total_terms': len(self.term_to_docs),
            'total_documents': self.total_documents,
            'most_frequent_terms': self.get_most_frequent_terms(10)
        }
    
    def get_most_frequent_terms(self, n: int) -> List[tuple]:

        sorted_terms = sorted(self.term_to_docs.items(), 
                            key=lambda x: len(x[1]), reverse=True)
        return [(term, len(docs)) for term, docs in sorted_terms[:n]]


def load_processed_corpus(filepath: str) -> List[Dict[str, Any]]:

    print(f"Loading processed corpus from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} documents")
    return corpus


if __name__ == "__main__":
    # Example usage
    corpus_path = "../part_1/data/processed_corpus.json"
    corpus = load_processed_corpus(corpus_path)
    
    # Build inverted index
    index = InvertedIndex()
    index.build_from_corpus(corpus)
    
    # Print statistics
    stats = index.get_vocabulary_stats()
    print("\nVocabulary Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Example query
    query_terms = ["women", "dress"]
    results = index.conjunctive_query(query_terms)
    print(f"\nQuery '{' '.join(query_terms)}' returned {len(results)} documents")
