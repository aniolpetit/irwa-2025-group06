from typing import List, Dict, Any, Tuple
from collections import Counter
import json
import random
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# TF


def compute_doc_frequencies_from_index(index) -> Dict[str, int]:
    """
    Compute document frequencies from the inverted index.
    This is more efficient than iterating through the corpus.
    Returns: dict mapping term -> number of documents containing it
    """
    doc_freqs = {}
    for term, postings in index.term_to_docs.items():
        doc_freqs[term] = len(postings)
    return doc_freqs


def create_test_queries(
    corpus_data: List[Dict[str, Any]],
    doc_freqs: Dict[str,int],
    top_k_terms: int = 200,
    n_queries: int = 5,
    index_lookup=None
) -> List[Dict[str, Any]]:

    total_docs = len(corpus_data)
    if total_docs == 0:
        return []

    # Candidate pool: top terms by document frequency (exclude 1-letter tokens and stopwords later if needed)
    sorted_terms = sorted(doc_freqs.items(), key=lambda x: x[1], reverse=True)
    candidates = [t for t, _ in sorted_terms[:top_k_terms]]

    # Helper to check if a conjunctive combination yields at least one doc (if index_lookup provided)
    def conjunctive_nonempty(terms: List[str]) -> bool:
        if index_lookup is None:
            return True
        sets = [set(index_lookup(term)) for term in terms if index_lookup(term)]
        if not sets:
            return False
        # intersect
        res = sets[0]
        for s in sets[1:]:
            res = res.intersection(s)
            if not res:
                return False
        return len(res) > 0

    queries = []
    used_terms = set()

    # Strategy plan for 5 queries:
    # 1) two very frequent generic terms -> high recall
    # 2) one frequent + one attribute (e.g., color) -> medium specificity
    # 3) brand (if present) + product type -> precision/brand test
    # 4) mid-frequency combination -> harder retrieval
    # 5) seasonal/rare combination or longer query -> precision test

    # Precompute lists by frequency bands
    freqs = [t for t in candidates]
    mid_start = max(10, len(freqs)//10)
    mid_end = min(len(freqs), mid_start + 50)
    mid_pool = freqs[mid_start:mid_end] if mid_start < mid_end else freqs[:50]

    # 1) very frequent pair
    def pick_pair_from(pool):
        for a in pool:
            for b in pool:
                if a != b and (a not in used_terms and b not in used_terms):
                    if conjunctive_nonempty([a,b]):
                        return [a,b]
        return None

    # Construct queries
    # Q1
    q1_terms = pick_pair_from(freqs[:20]) or pick_pair_from(freqs)
    # Q2 (freq + likely attribute: try to find color or size tokens heuristically)
    color_candidates = [t for t in candidates if t in ('black','white','red','blue','green','navy','beige','yellow')]
    q2_terms = None
    if color_candidates:
        for c in color_candidates:
            for prod in freqs[:100]:
                if c != prod and conjunctive_nonempty([c, prod]):
                    q2_terms = [c, prod]
                    break
            if q2_terms:
                break
    if q2_terms is None:
        q2_terms = pick_pair_from(freqs[10:80]) or q1_terms

    # Q3 (brand + product) - try to detect brands as tokens with capitalized forms in raw corpus or a small heuristic
    # If index_lookup provided, we could scan for tokens with moderate DF that look like brands; fallback to pick pair
    q3_terms = None
    # fallback:
    q3_terms = pick_pair_from(freqs[5:60]) or q2_terms

    # Q4 mid-frequency pair
    q4_terms = pick_pair_from(mid_pool) or pick_pair_from(freqs)

    # Q5 harder: pick a frequent + a rarer term
    q5_terms = None
    for a in freqs[:40]:
        for b in freqs[::-1][:200]:
            if a != b and conjunctive_nonempty([a, b]):
                q5_terms = [a, b]
                break
        if q5_terms:
            break
    if not q5_terms:
        q5_terms = q4_terms

    candidate_term_lists = [q1_terms, q2_terms, q3_terms, q4_terms, q5_terms]

    # Build query dicts and compute proper term statistics using doc_freqs
    for i, terms in enumerate(candidate_term_lists, start=1):
        if not terms:
            continue
        # mark used
        used_terms.update(terms)
        term_stats = []
        for term in terms:
            df = doc_freqs.get(term, 0)
            percentage = (df / total_docs) * 100
            term_stats.append({'term': term, 'doc_frequency': df, 'doc_percentage': percentage})
        q_text = " ".join(terms)
        queries.append({
            'id': i,
            'query': q_text,
            'terms': terms,
            'description': f'Auto-generated query #{i}',
            'reasoning': 'Selected based on document-frequency bands and conjunctive viability',
            'term_statistics': term_stats
        })

    return queries


def save_test_queries(queries: List[Dict[str, Any]], filepath: str) -> None:
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    print(f"Test queries saved to {filepath}")


if __name__ == "__main__":
    # Example: Generate test queries with actual inverted index
    from inverted_index import InvertedIndex, load_processed_corpus
    
    # Load corpus
    corpus_path = "../part_1/data/processed_corpus.json"
    corpus = load_processed_corpus(corpus_path)
    
    print("\n" + "="*80)
    print("STEP 1: Build inverted index")
    print("="*80)
    index = InvertedIndex()
    index.build_from_corpus(corpus)
    
    print("\n" + "="*80)
    print("STEP 2: Compute document frequencies from index (more efficient)")
    print("="*80)
    doc_freqs = compute_doc_frequencies_from_index(index)
    
    print(f"Total unique terms: {len(doc_freqs)}")
    print(f"\nTop 10 most frequent terms (by document frequency):")
    sorted_terms = sorted(doc_freqs.items(), key=lambda x: x[1], reverse=True)
    for i, (term, df) in enumerate(sorted_terms[:10], 1):
        print(f"{i:2d}. {term:15s} appears in {df:5d} documents ({df/len(corpus)*100:.1f}%)")
    
    # Create index lookup function for conjunctive query checking
    def index_lookup(term: str) -> set:
        return index.get_documents_for_term(term)
    
    print("\n" + "="*80)
    print("STEP 3: Generate test queries")
    print("="*80)
    queries = create_test_queries(
        corpus_data=corpus,
        doc_freqs=doc_freqs,
        top_k_terms=200,
        n_queries=5,
        index_lookup=index_lookup
    )
    
    # Display queries
    for query in queries:
        print(f"\nQuery {query['id']}: '{query['query']}'")
        print(f"  Terms: {query['terms']}")
        print(f"  Term statistics:")
        for term_stat in query['term_statistics']:
            print(f"    - '{term_stat['term']}': in {term_stat['doc_frequency']} docs ({term_stat['doc_percentage']:.1f}%)")
    
    # Save queries to file
    print("\n" + "="*80)
    print("STEP 4: Save queries to file")
    print("="*80)
    save_test_queries(queries, "test_queries.json")
