import math
from typing import Dict, List, Set, Tuple


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for information retrieval systems.
    
    All metrics support binary relevance judgments (0=non-relevant, 1=relevant).
    Some metrics also support graded relevance if needed.
    """
    
    def precision_at_k(self, relevance_labels: List[int], k: int) -> float:        
        # Precision@K = (number of relevant docs in top K) / K
        
        if k <= 0:
            return 0.0
        
        # Get top K results
        top_k_labels = relevance_labels[:k]
        
        # Count relevant documents (labels == 1) in top K
        relevant_in_topk = sum(1 for label in top_k_labels if label == 1)
        
        return relevant_in_topk / k
    
    def recall_at_k(self, relevance_labels: List[int], k: int) -> float:  
        # Recall@K = (number of relevant docs in top K) / total_relevant_docs
        
        # Total number of relevant documents
        total_relevant = sum(1 for label in relevance_labels if label == 1)
        if total_relevant == 0:
            return 0.0
        
        # Get top K results
        top_k_labels = relevance_labels[:k]
        
        # Count relevant documents (labels == 1) in top K
        relevant_in_topk = sum(1 for label in top_k_labels if label == 1)
        
        return relevant_in_topk / total_relevant
    
    def average_precision_at_k(self, relevance_labels: List[int], k: int) -> float:        
        # AP@K = (1/R) * Σ(Precision_i) for positions i where doc is relevant where R = total number of relevant documents in the collection

        # Total number of relevant documents in the entire set
        total_relevant_docs = sum(1 for label in relevance_labels if label == 1)
        if total_relevant_docs == 0:
            return 0.0
        
        # Limit to top K results
        top_k_labels = relevance_labels[:k]
        
        # Calculate AP by accumulating precision at each relevant position
        precision_sum = 0.0
        relevant_count = 0
        
        for position in range(len(top_k_labels)):
            label = top_k_labels[position]
            if label == 1:
                relevant_count += 1
                # Precision at this position = relevant_count / (position + 1)
                precision_at_pos = relevant_count / (position + 1)
                precision_sum += precision_at_pos
        
        # Average precision = sum of precisions / total number of relevant docs
        return precision_sum / total_relevant_docs
    
    def f1_score_at_k(self, relevance_labels: List[int], k: int) -> float:
        # F1@K = 2 * (P@K * R@K) / (P@K + R@K)

        precision = self.precision_at_k(relevance_labels, k)
        recall = self.recall_at_k(relevance_labels, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def mean_average_precision(self, ap_scores: List[float]) -> float:
        # MAP = (1 / N) * Σ AP_i for all queries

        if len(ap_scores) == 0:
            return 0.0
        
        return sum(ap_scores) / len(ap_scores)
    
    def mean_reciprocal_rank(self, relevance_labels_list: List[List[int]]) -> float:
        #MRR = (1 / N) * Σ (1 / rank_i) for all queries where rank_i is the position of the first relevant document

        if len(relevance_labels_list) == 0:
            return 0.0
        
        reciprocal_ranks = []
        
        for relevance_labels in relevance_labels_list:
            # Find the first relevant document (label == 1)
            found = False
            for position, label in enumerate(relevance_labels):
                if label == 1:
                    rank = position + 1  # 1-indexed rank
                    reciprocal_ranks.append(1.0 / rank)
                    found = True
                    break
            if not found:
                # No relevant document found
                reciprocal_ranks.append(0.0)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks)
    
    def normalized_discounted_cumulative_gain(self, relevance_labels: List[int], k: int) -> float: 
        # NDCG@K = DCG@K / IDCG@K
        # Where:
        # - DCG@K = Σ (relevance_i / log2(i + 1)) for positions 1 to K
        # - IDCG@K = ideal DCG@K (ranking all relevant docs first)

        def dcg(scores: List[float]) -> float:
            # Helper function to compute the dcg
            dcg_value = 0.0
            for i, score in enumerate(scores):
                position = i + 1
                dcg_value += score / math.log2(position + 1)
            return dcg_value
        
        if k <= 0:
            return 0.0
        
        # Get top K results
        top_k_labels = relevance_labels[:k]
        
        # Convert labels to relevance scores (for binary, label 1 = 1.0, label 0 = 0.0)
        relevance_scores = [float(label) for label in top_k_labels]
        
        # Calculate DCG@K
        dcg_at_k = dcg(relevance_scores)
        
        # Calculate IDCG@K (ideal DCG by sorting all scores descending)
        all_labels = [float(label) for label in relevance_labels]
        ideal_relevance_scores = sorted(all_labels, reverse=True)[:k]
        idcg_at_k = dcg(ideal_relevance_scores)
        
        # Normalize
        if idcg_at_k == 0:
            return 0.0
        
        return dcg_at_k / idcg_at_k
