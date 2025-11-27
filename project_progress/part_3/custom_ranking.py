import math
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PART2_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "part_2"))
if PART2_DIR not in sys.path:
    sys.path.append(PART2_DIR)

from tfidf_ranking import TFIDFRanker
from inverted_index import InvertedIndex 


class CustomRanker:
    """
    Custom score combining TF-IDF base with field-aware boosts, term proximity, metadata signals,
    and length normalization.
    """

    FIELD_WEIGHTS = {
        "title_tokens": 1.0,
        "brand_tokens": 0.6,
        "subcategory_tokens": 0.5,
        "details_tokens": 0.3,
        "description_tokens": 0.2,
    }

    def __init__(
        self,
        inverted_index: InvertedIndex,
        corpus_data: List[Dict],
        tfidf_ranker: TFIDFRanker | None = None,
        description_penalty_lambda: float = 0.5,
        proximity_weight: float = 0.2,
        field_weight_scale: float = 0.8,
        rating_weight: float = 0.25,
        out_of_stock_penalty: float = 0.1,
        exact_match_bonus: float = 0.2,
    ):
        # Initializes the ranker with TF-IDF base and precomputes caches for efficient scoring
        self.index = inverted_index
        self.corpus_data = corpus_data
        self.tfidf_ranker = tfidf_ranker or TFIDFRanker(inverted_index, corpus_data)

        self.description_penalty_lambda = description_penalty_lambda
        self.proximity_weight = proximity_weight
        self.field_weight_scale = field_weight_scale
        self.rating_weight = rating_weight
        self.out_of_stock_penalty = out_of_stock_penalty
        self.exact_match_bonus = exact_match_bonus

        self.doc_lookup: Dict[str, Dict] = {
            doc.get("pid"): doc for doc in corpus_data if doc.get("pid")
        }

        self.avg_description_length = self._compute_average_description_length()
        self.field_token_cache = self._build_field_token_cache()

    def _compute_average_description_length(self) -> float:
        # Computes the average description length across all documents for normalization purposes
        lengths = []
        for doc in self.corpus_data:
            tokens = doc.get("description_tokens", [])
            if isinstance(tokens, str):
                tokens = tokens.split()
            if tokens:
                lengths.append(len(tokens))
        if not lengths:
            return 1.0
        return sum(lengths) / len(lengths)

    def _build_field_token_cache(self) -> Dict[str, Dict[str, Set[str]]]:
        # Pre-computes a cache mapping each document to sets of tokens in each weighted field for fast lookup
        cache: Dict[str, Dict[str, Set[str]]] = {}
        for doc in self.corpus_data:
            doc_id = doc.get("pid")
            if not doc_id:
                continue
            field_map: Dict[str, Set[str]] = {}
            for field in self.FIELD_WEIGHTS.keys():
                tokens = doc.get(field, [])
                if isinstance(tokens, str):
                    tokens = tokens.split()
                field_map[field] = set(tokens)
            cache[doc_id] = field_map
        return cache

    def _compute_field_score(self, doc_id: str, query_term_set: Set[str]) -> float:
        # Calculates a weighted score based on how many query terms appear in each document field
        field_tokens = self.field_token_cache.get(doc_id, {})
        if not field_tokens:
            return 0.0

        total_weight = 0.0
        for field, weight in self.FIELD_WEIGHTS.items():
            tokens = field_tokens.get(field, set())
            if not tokens:
                continue
            matches = sum(1 for term in query_term_set if term in tokens)
            if matches == 0:
                continue
            total_weight += weight * (matches / len(query_term_set))
        return total_weight

    def _compute_proximity_score(self, doc_id: str, query_terms: Set[str]) -> float:
        # Computes a proximity score based on how close query terms appear to each other in the document
        if len(query_terms) <= 1:
            return 1.0

        position_lists: List[List[int]] = []
        for term in query_terms:
            postings = self.index.term_to_docs.get(term, [])
            doc_positions = None
            for posting_doc_id, positions in postings:
                if posting_doc_id == doc_id:
                    doc_positions = list(positions)
                    break
            if not doc_positions:
                # term missing or no positions
                return 0.0
            position_lists.append(sorted(doc_positions))

        if not position_lists:
            return 0.0

        # Heap-based approach to compute minimal span covering all query terms
        import heapq

        indices = [0] * len(position_lists)
        heap: List[Tuple[int, int]] = []
        current_max = -math.inf

        for idx, pos_list in enumerate(position_lists):
            if not pos_list:
                return 0.0
            pos = pos_list[0]
            heapq.heappush(heap, (pos, idx))
            current_max = max(current_max, pos)

        best_span = math.inf
        while True:
            current_min, list_idx = heap[0]
            span = current_max - current_min
            if span < best_span:
                best_span = span

            indices[list_idx] += 1
            if indices[list_idx] >= len(position_lists[list_idx]):
                break

            next_pos = position_lists[list_idx][indices[list_idx]]
            heapq.heapreplace(heap, (next_pos, list_idx))
            current_max = max(current_max, next_pos)

        if best_span == math.inf:
            return 0.0
        return 1.0 / (1.0 + best_span)

    def _compute_rating_score(self, doc: Dict) -> float:
        # Normalizes the average rating to a 0-1 scale for scoring
        rating = doc.get("average_rating")
        if rating is None:
            return 0.0
        try:
            rating = float(rating)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(rating / 5.0, 1.0))

    def _compute_stock_penalty(self, doc: Dict) -> float:
        # Returns a penalty value if the product is out of stock, otherwise returns zero
        out_of_stock = doc.get("out_of_stock")
        if isinstance(out_of_stock, bool) and out_of_stock:
            return self.out_of_stock_penalty
        return 0.0

    def _compute_exact_match_bonus(self, doc: Dict, query_string: str) -> float:
        # Adds a bonus if the entire query string appears in the title
        title = doc.get("title", "")
        if isinstance(title, str) and query_string in title.lower():
            return self.exact_match_bonus
        return 0.0

    def _compute_length_factor(self, doc: Dict) -> float:
        # Applies a penalty factor to documents with longer than average descriptions
        tokens = doc.get("description_tokens", [])
        if isinstance(tokens, str):
            tokens = tokens.split()
        length = len(tokens) if tokens else self.avg_description_length
        if self.avg_description_length <= 0:
            return 1.0
        ratio = length / self.avg_description_length
        penalty = 1.0 / (1.0 + self.description_penalty_lambda * max(0.0, math.log1p(ratio - 1.0)))
        return penalty

    def rank_documents(self, query_terms: List[str], candidate_docs: Set[str]) -> List[Tuple[str, float]]:
        # Main ranking function that combines all scoring components to rank candidate documents
        if not candidate_docs:
            return []

        base_scores_list = self.tfidf_ranker.rank_documents(query_terms, candidate_docs)
        base_scores = {doc_id: score for doc_id, score in base_scores_list}

        scored_docs: List[Tuple[str, float]] = []
        query_term_set = set(query_terms)
        query_string = " ".join(query_terms)

        for doc_id in candidate_docs:
            base_score = base_scores.get(doc_id, 0.0)
            doc = self.doc_lookup.get(doc_id, {})

            field_score = self._compute_field_score(doc_id, query_term_set)
            proximity_score = self._compute_proximity_score(doc_id, query_term_set)
            rating_score = self._compute_rating_score(doc)
            stock_penalty = self._compute_stock_penalty(doc)
            exact_bonus = self._compute_exact_match_bonus(doc, query_string)
            length_factor = self._compute_length_factor(doc)

            composite = base_score
            composite += self.field_weight_scale * field_score
            composite += self.proximity_weight * proximity_score
            composite += self.rating_weight * rating_score
            composite -= stock_penalty
            composite += exact_bonus

            composite *= length_factor

            scored_docs.append((doc_id, composite))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs