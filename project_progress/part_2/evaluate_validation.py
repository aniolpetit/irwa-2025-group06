"""
Evaluation script for validation_labels.csv queries.
Applies the evaluation metrics to the predefined queries.
"""

import csv
from typing import Dict, List

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

from inverted_index import InvertedIndex, load_processed_corpus
from tfidf_ranking import TFIDFRanker
from evaluation_metrics import EvaluationMetrics

# Download NLTK data if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Initialize NLTK components
STOP_WORDS = set(stopwords.words('english'))
STEMMER = SnowballStemmer('english')


def preprocess_query(query_text: str) -> List[str]:
    """
    Preprocess query using the same pipeline as the corpus.
    """
    # Tokenize
    tokens = word_tokenize(query_text.lower())
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in STOP_WORDS]
    
    # Stem
    tokens = [STEMMER.stem(t) for t in tokens if len(t) > 2]
    
    return tokens


def load_validation_labels(filepath: str) -> Dict[int, List[tuple]]:
    """
    Load validation labels from CSV.
    
    Returns a dictionary where:
    - Key: query_id (1 or 2)
    - Value: List of tuples (pid, label) in the order they appear in CSV
    """
    validation_data = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = int(row['query_id'])
            pid = row['pid']
            label = int(row['labels'])
            
            if query_id not in validation_data:
                validation_data[query_id] = []
            validation_data[query_id].append((pid, label))
    
    return validation_data






def run_tfidf_ranking():
    """
    Run TF-IDF ranking on the validation queries.
    Returns ranked document IDs for each query.
    """
    # Load corpus
    corpus_path = "../part_1/data/processed_corpus.json"
    corpus = load_processed_corpus(corpus_path)
    
    # Build index
    index = InvertedIndex()
    index.build_from_corpus(corpus)
    
    # Create ranker
    ranker = TFIDFRanker(index, corpus)
    
    # Define queries
    query_texts = {
        1: "women full sleeve sweatshirt cotton",
        2: "men slim jeans blue"
    }
    
    ranked_results = {}
    
    for query_id, query_text in query_texts.items():
        print(f"\nRunning query {query_id}: '{query_text}'")
        
        # Preprocess query
        query_terms = preprocess_query(query_text)
        print(f"  Preprocessed: {query_terms}")
        
        # Get candidates
        candidate_docs = index.conjunctive_query(query_terms)
        print(f"  Candidate documents: {len(candidate_docs)}")
        
        if candidate_docs:
            # Rank documents
            ranked = ranker.rank_documents(query_terms, candidate_docs)
            # Extract just document IDs
            ranked_ids = [doc_id for doc_id, score in ranked]
            print(f"  Ranked results: {len(ranked_ids)} documents")
        else:
            ranked_ids = []
            print("  No results found!")
        
        ranked_results[query_id] = ranked_ids
    
    return ranked_results


def create_relevance_labels(ranked_docs: List[str], validation_data: List[tuple]) -> List[int]:
    """
    Create relevance label list matching the ranked order.
    
    Args:
        ranked_docs: List of document IDs in ranked order
        validation_data: List of (pid, label) tuples from validation_labels.csv
    
    Returns:
        List of relevance labels (1 or 0) in the same order as ranked_docs
    """
    # Create a dict for quick lookup: pid -> label
    pid_to_label = {pid: label for pid, label in validation_data}
    
    # Create relevance labels matching the ranked order
    relevance_labels = []
    for doc_id in ranked_docs:
        # Look up label, default to 0 if not in validation data
        label = pid_to_label.get(doc_id, 0)
        relevance_labels.append(label)
    
    return relevance_labels


def evaluate_queries():
    """
    Main evaluation function.
    """
    print("=" * 80)
    print("Evaluation Results for Validation Labels")
    print("=" * 80)
    
    # Load validation labels
    validation_path = "validation_labels.csv"
    validation_data = load_validation_labels(validation_path)
    
    # Run TF-IDF ranking
    print("\n" + "=" * 80)
    print("Running TF-IDF Ranking")
    print("=" * 80)
    ranked_results = run_tfidf_ranking()
    
    # Create evaluator instance
    evaluator = EvaluationMetrics()
    
    # Evaluate each query
    results = {}
    
    for query_id in [1, 2]:
        print(f"\nEvaluating Query {query_id}:")
        if query_id == 1:
            print("  Query: 'women full sleeve sweatshirt cotton'")
        else:
            print("  Query: 'men slim jeans blue'")
        
        # Get ranked documents from TF-IDF
        ranked_docs = ranked_results.get(query_id, [])
        validation_labels_data = validation_data.get(query_id, [])
        
        print(f"  Retrieved: {len(ranked_docs)} documents")
        print(f"  Validation data: {len(validation_labels_data)} documents")
        
        # Create relevance labels in ranked order from TF-IDF results
        relevance_labels = create_relevance_labels(ranked_docs, validation_labels_data)
        
        # For metrics that need k, we'll use the full length
        # But we need to handle cases where there might be more documents in validation than retrieved
        
        # Calculate metrics for different k values (1, 3, 5, 10, and full length)
        k_values = [1, 3, 5, 10]
        if len(relevance_labels) > 10:
            k_values.append(len(relevance_labels))
        
        query_results = {}
        
        for k in k_values:
            if k > len(relevance_labels):
                continue
            
            p_at_k = evaluator.precision_at_k(relevance_labels, k)
            r_at_k = evaluator.recall_at_k(relevance_labels, k)
            f1_at_k = evaluator.f1_score_at_k(relevance_labels, k)
            ap_at_k = evaluator.average_precision_at_k(relevance_labels, k)
            ndcg_at_k = evaluator.normalized_discounted_cumulative_gain(relevance_labels, k)
            
            query_results[f'P@{k}'] = p_at_k
            query_results[f'R@{k}'] = r_at_k
            query_results[f'F1@{k}'] = f1_at_k
            query_results[f'AP@{k}'] = ap_at_k
            query_results[f'NDCG@{k}'] = ndcg_at_k
        
        results[query_id] = query_results
    
    # Calculate MRR across both queries
    relevance_labels_list = []
    for query_id in [1, 2]:
        ranked_docs = ranked_results.get(query_id, [])
        validation_labels_data = validation_data.get(query_id, [])
        relevance_labels = create_relevance_labels(ranked_docs, validation_labels_data)
        relevance_labels_list.append(relevance_labels)
    
    mrr = evaluator.mean_reciprocal_rank(relevance_labels_list)
    
    # Calculate MAP - use the max k that was calculated
    max_k = 10  # We calculated up to k=10
    ap_scores = []
    for q in [1, 2]:
        if f'AP@{max_k}' in results[q]:
            ap_scores.append(results[q][f'AP@{max_k}'])
    map_score = evaluator.mean_average_precision(ap_scores) if ap_scores else 0.0
    
    # Print formatted results
    print("\n" + "=" * 80)
    print("Numeric Results (rounded to 3 decimal places)")
    print("=" * 80)
    
    for query_id in [1, 2]:
        print(f"\nQuery {query_id}:")
        query_results = results[query_id]
        
        for metric_name in sorted(query_results.keys()):
            value = query_results[metric_name]
            print(f"  {metric_name}: {value:.3f}")
    
    print(f"\nAggregate Metrics:")
    print(f"  MAP: {map_score:.3f}")
    print(f"  MRR: {mrr:.3f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    evaluate_queries()

