import os
import sys
from typing import List, Set, Tuple, Dict, Any

from myapp.search.objects import Document

# Add Part 2 and Part 3 directories to path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
PART2_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "project_progress", "part_2"))
PART3_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "project_progress", "part_3"))

if PART2_DIR not in sys.path:
    sys.path.append(PART2_DIR)
if PART3_DIR not in sys.path:
    sys.path.append(PART3_DIR)

from inverted_index import InvertedIndex, load_processed_corpus
from tfidf_ranking import TFIDFRanker
from custom_ranking import CustomRanker


class SearchEngine:
    """Class that implements the search engine logic with caching for performance"""
    
    def __init__(self):
        self.index = None
        self.processed_corpus = None
        self.tfidf_ranker = None
        self.custom_ranker = None
        self.doc_lookup = None
        self._initialized = False
    
    def _initialize(self):
        """Initialize the search engine with inverted index and rankers (cached)"""
        if self._initialized:
            return
        
        print("Initializing search engine...")
        
        # Load processed corpus from Part 1
        corpus_path = os.path.abspath(
            os.path.join(PROJECT_ROOT, "project_progress", "part_1", "data", "processed_corpus.json")
        )
        
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Processed corpus not found at {corpus_path}. Please run Part 1 first.")
        
        self.processed_corpus = load_processed_corpus(corpus_path, verbose=False)
        
        # Build inverted index
        self.index = InvertedIndex()
        self.index.build_from_corpus(self.processed_corpus, text_field="tokens", verbose=False)
        
        # Initialize rankers
        self.tfidf_ranker = TFIDFRanker(self.index, self.processed_corpus)
        self.custom_ranker = CustomRanker(
            self.index,
            self.processed_corpus,
            tfidf_ranker=self.tfidf_ranker
        )
        
        # Build lookup dictionary for fast document access
        self.doc_lookup = {doc.get("pid"): doc for doc in self.processed_corpus if doc.get("pid")}
        
        self._initialized = True
        print(f"Search engine initialized with {len(self.processed_corpus)} documents")
    
    def _preprocess_query(self, query: str) -> List[str]:
        """Preprocess query by lowercasing and splitting into tokens"""
        return query.lower().split()
    
    def _convert_to_document_objects(
        self, 
        ranked_results: List[Tuple[str, float]], 
        web_corpus: Dict[str, Document],
        search_id: int,
        top_k: int = 20
    ) -> List[Document]:
        """Convert ranked results to Document objects with all required fields"""
        results = []
        
        for doc_id, score in ranked_results[:top_k]:
            # Get document from processed corpus (has tokens)
            processed_doc = self.doc_lookup.get(doc_id)
            if not processed_doc:
                continue
            
            # Get document from web corpus (has full metadata)
            web_doc = web_corpus.get(doc_id)
            if not web_doc:
                continue
            
            # Create result document with all required fields
            result_doc = Document(
                _id=web_doc._id,
                pid=web_doc.pid,
                title=web_doc.title,
                description=web_doc.description,
                brand=web_doc.brand,
                category=web_doc.category,
                sub_category=web_doc.sub_category,
                product_details=web_doc.product_details,
                seller=web_doc.seller,
                out_of_stock=web_doc.out_of_stock,
                selling_price=web_doc.selling_price,
                discount=web_doc.discount,
                actual_price=web_doc.actual_price,
                average_rating=web_doc.average_rating,
                url=f"/doc_details?pid={web_doc.pid}&search_id={search_id}",
                images=web_doc.images
            )
            
            # Store ranking score in a custom attribute (not in Document model, but useful)
            result_doc.ranking = score
            
            results.append(result_doc)
        
        return results
    
    def search(self, search_query: str, search_id: int, corpus: Dict[str, Document], top_k: int = 20):
        """
        Main search function that receives a query string and returns ranked results
        
        :param search_query: The user's search query
        :param search_id: Unique identifier for this search (for analytics)
        :param corpus: The web corpus dictionary (Document objects with full metadata)
        :param top_k: Number of results to return
        :return: List of Document objects ranked by relevance
        """
        # Initialize if not already done
        self._initialize()
        
        print(f"Search query: '{search_query}' (search_id: {search_id})")
        
        # Preprocess query
        query_terms = self._preprocess_query(search_query)
        
        if not query_terms:
            print("Empty query after preprocessing")
            return []
        
        # Perform conjunctive query to get candidate documents
        candidate_docs: Set[str] = self.index.conjunctive_query(query_terms)
        
        if not candidate_docs:
            print(f"No documents found matching all query terms: {query_terms}")
            return []
        
        print(f"Found {len(candidate_docs)} candidate documents matching all query terms")
        
        # Rank documents using custom ranker (combines TF-IDF with field weights, proximity, ratings, etc.)
        ranked_results = self.custom_ranker.rank_documents(query_terms, candidate_docs)
        
        print(f"Ranked {len(ranked_results)} documents")
        
        # Convert to Document objects with all required fields
        results = self._convert_to_document_objects(ranked_results, corpus, search_id, top_k)
        
        print(f"Returning {len(results)} results")
        
        return results
