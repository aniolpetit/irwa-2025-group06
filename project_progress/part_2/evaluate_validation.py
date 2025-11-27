import csv
from typing import Dict, List

from evaluation_metrics import EvaluationMetrics


def load_validation_labels(filepath: str) -> Dict[int, List[tuple]]:
    # Returns a dictionary where:
    # - Key: query_id (1 or 2)
    # - Value: List of tuples (pid, label) in the order they appear in CSV
    
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



def get_retrieved_documents_and_labels(validation_data: Dict[int, List[tuple]]) -> Dict[int, tuple]:
    # We extract the retrieved documents and their relevance labels from the validation data.
    # The validation CSV contains 20 documents per query in retrieval order.
    results = {}
    
    for query_id, data_list in validation_data.items():
        # Extract document IDs (pids) in order they appear in CSV (retrieval order)
        retrieved_docs = [pid for pid, label in data_list]
        # Extract relevance labels in the same order
        relevance_labels = [label for pid, label in data_list]
        
        results[query_id] = (retrieved_docs, relevance_labels)
    
    return results


def evaluate_queries():    
    # Load validation labels - these contain 20 retrieved documents per query with ground truth labels
    validation_path = "../../data/validation_labels.csv"
    validation_data = load_validation_labels(validation_path)
    
    # Extract retrieved documents and relevance labels from validation CSV
    # The CSV order represents the retrieval order
    retrieved_data = get_retrieved_documents_and_labels(validation_data)
    
    # Create evaluator instance
    evaluator = EvaluationMetrics()
    
    # Query texts for display
    query_texts = {
        1: "women full sleeve sweatshirt cotton",
        2: "men slim jeans blue"
    }
    
    # Evaluate each query
    results = {}
    relevance_labels_list = []
    
    for query_id in [1, 2]:
        print(f"\nEvaluating Query {query_id}:")
        print(f"  Query: '{query_texts[query_id]}'")
        
        # Get retrieved documents and relevance labels from CSV (in retrieval order)
        retrieved_docs, relevance_labels = retrieved_data[query_id]
        
        print(f"  Retrieved documents: {len(retrieved_docs)}")
        print(f"  Relevant documents: {sum(relevance_labels)}")
        print(f"  Non-relevant documents: {len(relevance_labels) - sum(relevance_labels)}")
        
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
        relevance_labels_list.append(relevance_labels)
    
    # Calculate MRR across both queries
    mrr = evaluator.mean_reciprocal_rank(relevance_labels_list)
    
    # Calculate MAP - use the max k that was calculated (use full length for MAP)
    max_k = len(relevance_labels_list[0]) if relevance_labels_list else 20
    ap_scores = []
    for q in [1, 2]:
        if f'AP@{max_k}' in results[q]:
            ap_scores.append(results[q][f'AP@{max_k}'])
    map_score = evaluator.mean_average_precision(ap_scores) if ap_scores else 0.0
    
    print("Numeric Results (rounded to 3 decimal places)")
    
    for query_id in [1, 2]:
        print(f"\nQuery {query_id} ('{query_texts[query_id]}'):")
        query_results = results[query_id]
        
        for metric_name in sorted(query_results.keys()):
            value = query_results[metric_name]
            print(f"  {metric_name}: {value:.3f}")
    
    print(f"\nAggregate Metrics:")
    print(f"  MAP: {map_score:.3f}")
    print(f"  MRR: {mrr:.3f}")
    
if __name__ == "__main__":
    evaluate_queries()

