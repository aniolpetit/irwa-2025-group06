import os
import sys
from typing import List, Tuple, Set, Dict, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PART2_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "part_2"))
if PART2_DIR not in sys.path:
    sys.path.append(PART2_DIR)

from inverted_index import InvertedIndex, load_processed_corpus  
from word2vec_ranking import Word2VecRanker  


def preprocess_query(query: str) -> List[str]:
    # Preprocess query by lowercasing and splitting into tokens.
    return query.lower().split()


def run_word2vec_cosine_for_queries(
    queries: List[str],
    corpus_path: str,
    top_k: int = 20,
    model_name: str = "word2vec-google-news-300",
    model_path: Optional[str] = None,
) -> Dict[str, List[Tuple[str, float]]]:
    corpus = load_processed_corpus(corpus_path)
    doc_lookup = {doc.get("pid"): doc for doc in corpus if doc.get("pid")}

    index = InvertedIndex()
    index.build_from_corpus(corpus)

    ranker = Word2VecRanker(
        index,
        corpus,
        text_field="tokens",
        model_name=model_name,
        model_path=model_path,
    )

    results: Dict[str, List[Tuple[str, float]]] = {}

    for query in queries:
        query_terms = preprocess_query(query)
        candidate_docs: Set[str] = index.conjunctive_query(query_terms)

        print(f"\nQuery: {query}")
        print(f"Query terms: {query_terms}")
        if not candidate_docs:
            print("No documents found matching all query terms.")
            print(f"{'='*80}")
            results[query] = []
            continue

        ranked = ranker.rank_documents(query_terms, candidate_docs)
        results[query] = ranked[:top_k]

        total_results = len(ranked)
        display_count = min(top_k, total_results)
        print(f"\nTotal results: {total_results}")
        print(f"Showing top {display_count} results:\n")

        for i, (doc_id, score) in enumerate(ranked[:display_count]):
            doc = doc_lookup.get(doc_id, {})
            title = doc.get("title", "N/A")
            brand = doc.get("brand", "N/A")
            category = doc.get("category", "N/A")
            sub_category = doc.get("sub_category", "N/A")
            description = doc.get("description", "")
            desc_snippet = description if description else "N/A"

            print(f"{i+1:3d}. [Score: {score:.6f}] | PID: {doc_id}")
            print(f"     Title: {title}")
            print(f"     Brand: {brand} | Category: {category} | Sub-category: {sub_category}")
            print(f"     Description: {desc_snippet}")
            print()

        if total_results > top_k:
            print(f"... and {total_results - top_k} more results (not shown)")
        print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    queries = [
        "ecko unl shirt",
        "ecko unl men shirt round neck",
        "women polo cotton",
        "casual clothes slim fit",
        "biowash innerwear",
    ]
    corpus_path = os.path.abspath(
        os.path.join(CURRENT_DIR, "..", "part_1", "data", "processed_corpus.json")
    )
    
    # We use a smaller model by default to avoid long download times, heavier (and probably better) options are available.
    run_word2vec_cosine_for_queries(
        queries, 
        corpus_path, 
        top_k=20,
        model_name="glove-wiki-gigaword-100"  # Smaller model for faster loading
    )

