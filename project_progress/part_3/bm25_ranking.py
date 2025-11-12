import math
from collections import Counter
from typing import Dict, List, Set, Tuple

from typing import Any

try:
    from inverted_index import InvertedIndex
except ImportError:
    
    InvertedIndex = Any  


class BM25Ranker:
    
    # BM25 ranking implementation using the formula seen in theory (simple version, no k3 for long queries)
    

    def __init__(
        self,
        inverted_index: InvertedIndex,
        corpus_data: List[Dict],
        text_field: str = "tokens",
        k1: float = 1.2,
        b: float = 0.75,
    ):
        self.index = inverted_index
        self.corpus_data = corpus_data
        self.text_field = text_field
        self.k1 = k1
        self.b = b

        self.total_documents = len(corpus_data)
        self.document_frequencies: Dict[str, int] = self.build_document_frequencies()
        self.document_lengths: Dict[str, int] = self.build_document_lengths()
        self.avg_document_length: float = (
            sum(self.document_lengths.values()) / self.total_documents
            if self.total_documents > 0
            else 0.0
        )

    def build_document_frequencies(self) -> Dict[str, int]:
        doc_freqs: Dict[str, int] = {}
        for term, postings in self.index.term_to_docs.items():
            doc_freqs[term] = len(postings)
        return doc_freqs

    def build_document_lengths(self) -> Dict[str, int]:
        lengths: Dict[str, int] = {}
        for doc in self.corpus_data:
            doc_id = doc.get("pid")
            if not doc_id:
                continue
            tokens = doc.get(self.text_field, [])
            if isinstance(tokens, str):
                tokens = tokens.split()
            lengths[doc_id] = len(tokens)
        return lengths

    def idf(self, term: str) -> float:
        df = self.document_frequencies.get(term, 0)
        # BM25 idf; guard df extremes
        numerator = (self.total_documents)
        denominator = (df)
        if denominator <= 0:
            return 0.0
        ratio = numerator / denominator
        if ratio <= 0:
            return 0.0
        return math.log(ratio)

    def bm25_tf(self, tf: int, doc_len: int) -> float:
        if tf <= 0:
            return 0.0
        denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_document_length if self.avg_document_length > 0 else 0.0))
        if denom == 0.0:
            return 0.0
        return (tf * (self.k1 + 1.0)) / denom

    def rank_documents(self, query_terms: List[str], candidate_docs: Set[str]) -> List[Tuple[str, float]]:
        query_tf = Counter(query_terms)
        scores: List[Tuple[str, float]] = []

        # Precompute idf for query terms appearing in index to avoid repeated work
        idf_map: Dict[str, float] = {t: self.idf(t) for t in set(query_terms) if t in self.index.term_to_docs}

        # Build a quick access map from term -> dict(doc_id -> tf)
        term_doc_tf: Dict[str, Dict[str, int]] = {}
        for term in idf_map.keys():
            postings = self.index.term_to_docs.get(term, [])
            doc_tf_map: Dict[str, int] = {}
            for doc_id, positions in postings:
                doc_tf_map[doc_id] = len(positions)
            term_doc_tf[term] = doc_tf_map

        for doc_id in candidate_docs:
            doc_len = self.document_lengths.get(doc_id, 0)
            score = 0.0
            for term, idf_val in idf_map.items():
                tf = term_doc_tf[term].get(doc_id, 0)
                if tf == 0:
                    continue
                tf_component = self.bm25_tf(tf, doc_len)
                score += idf_val * tf_component
            scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


