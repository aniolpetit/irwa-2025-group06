import json
from collections import defaultdict
from typing import Dict, List, Set, Any
from array import array


class InvertedIndex:
    """
    Simple inverted index for conjunctive queries (AND operations).
    Stores term positions in documents to enable TF-IDF calculation.
    
    Structure:
    - term_to_docs[term] = [[doc_id, array([pos1, pos2, ...])], [doc_id, array([pos1, pos2, ...])], ...]
    - Each posting contains: [document_id, array of term positions in that document]
    """
    
    def __init__(self):
        self.term_to_docs: Dict[str, List[List]] = defaultdict(list)
        self.total_documents: int = 0
    
    def add_document(self, doc_id: str, tokens: List[str]) -> None:
        if not tokens:
            return
        
        # Track positions for each term in this document
        term_positions = defaultdict(list)
        for position, term in enumerate(tokens):
            term_positions[term].append(position)
        
        # Add postings to the index: [doc_id, array of positions]
        for term, positions in term_positions.items():
            posting = [doc_id, array('I', positions)]  # 'I' = unsigned int
            self.term_to_docs[term].append(posting)
        
        self.total_documents += 1
    
    def build_from_corpus(self, corpus_data: List[Dict[str, Any]], 
                         text_field: str = 'tokens', verbose: bool = False) -> None:

        if verbose:
            print(f"Building inverted index from {len(corpus_data)} documents...")
        
        for doc in corpus_data:
            doc_id = doc.get('pid', '')
            if not doc_id:
                continue
                
            tokens = doc.get(text_field, [])
            if isinstance(tokens, str):
                tokens = tokens.split()
            
            self.add_document(doc_id, tokens)
        
        if verbose:
            print(f"Index built successfully. Vocabulary size: {len(self.term_to_docs)}")
            # Debug: Show first 10 terms in the inverted index
            self.debug_print_index_samples(n=10)
    
    def conjunctive_query(self, terms: List[str]) -> Set[str]:
        if not terms:
            return set()
        
        # Extract document IDs from postings: posting = [doc_id, array(positions)]
        def get_doc_ids(term: str) -> Set[str]:
            postings = self.term_to_docs.get(term, [])
            return {posting[0] for posting in postings}
        
        # Start with documents containing the first term
        result = get_doc_ids(terms[0])
        
        # Intersect with documents containing each subsequent term
        for term in terms[1:]:
            term_docs = get_doc_ids(term)
            result = result.intersection(term_docs)
            
            # Early termination if no documents match
            if not result:
                break
        
        return result
    
    def get_documents_for_term(self, term: str) -> Set[str]:
        # We get all document IDs containing the given term
        postings = self.term_to_docs.get(term, [])
        return {posting[0] for posting in postings}
    
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        return {
            'total_terms': len(self.term_to_docs),
            'total_documents': self.total_documents,
            'most_frequent_terms': self.get_most_frequent_terms(10)
        }
    
    def get_most_frequent_terms(self, n: int) -> List[tuple]:
        # We get the top N most frequent terms by document count
        sorted_terms = sorted(self.term_to_docs.items(), 
                            key=lambda x: len(x[1]), reverse=True)
        return [(term, len(docs)) for term, docs in sorted_terms[:n]]
    
    def debug_print_index_samples(self, n: int = 10) -> None:
        # We print the first N terms in the inverted index for debugging purposes
        print(f"\nFirst {n} terms in the inverted index:")
        for i, (term, postings) in enumerate(list(self.term_to_docs.items())[:n]):
            print(f"\n'{term}': {len(postings)} documents")
            for j, posting in enumerate(postings[:3]):
                print(f"  [{posting[0]}, {list(posting[1])}]")
            if len(postings) > 3:
                print(f"  ... ({len(postings) - 3} more)")


def load_processed_corpus(filepath: str, verbose: bool = False) -> List[Dict[str, Any]]:
    # We load the processed corpus from the given file
    if verbose:
        print(f"Loading processed corpus from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    if verbose:
        print(f"Loaded {len(corpus)} documents")
    return corpus


if __name__ == "__main__":
    # Example usage
    corpus_path = "../part_1/data/processed_corpus.json"
    corpus = load_processed_corpus(corpus_path, verbose=True)
    
    # Build inverted index
    index = InvertedIndex()
    index.build_from_corpus(corpus, verbose=True)
    
    # Print statistics
    stats = index.get_vocabulary_stats()
    print("\nVocabulary Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
