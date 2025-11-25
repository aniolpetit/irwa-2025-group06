from typing import List, Optional
from myapp.search.objects import Document
from myapp.search.algorithms import SearchAlgorithm


class SearchEngine:
    """
    Search engine that integrates TF-IDF ranking algorithm.
    Optimized for web application use with pre-initialized search algorithm.
    """
    
    def __init__(self, search_algorithm: SearchAlgorithm = None):
        """
        Initialize the search engine.
        
        :param search_algorithm: Pre-initialized SearchAlgorithm instance.
                                 If None, must be set via set_search_algorithm() before use.
        """
        self.search_algorithm = search_algorithm
    
    def set_search_algorithm(self, search_algorithm: SearchAlgorithm):
        """
        Set the search algorithm instance.
        This allows for lazy initialization if needed.
        
        :param search_algorithm: SearchAlgorithm instance
        """
        self.search_algorithm = search_algorithm
    
    def search(
        self,
        search_query: str,
        search_id: int,
        corpus: dict,
        top_k: int = 20,
        ranking_method: Optional[str] = None
    ) -> List[Document]:
        """
        Perform search using the integrated search algorithm.
        
        :param search_query: User's search query string
        :param search_id: Search session ID for analytics
        :param corpus: Dictionary of Document objects (pid -> Document) for result formatting (fallback)
        :param top_k: Number of top results to return
        :return: List of Document objects with ranking scores
        """
        if not self.search_algorithm:
            raise ValueError("Search algorithm not initialized. Call set_search_algorithm() first.")
        
        if not search_query or not search_query.strip():
            return []
        
        # Perform search using TF-IDF ranking
        ranked_results = self.search_algorithm.search(
            search_query,
            top_k=top_k,
            ranking_method=ranking_method
        )
        
        # Convert results to Document objects for web display
        results = []
        for doc_id, score in ranked_results:
            # Get document data from search algorithm's corpus (processed corpus with all fields)
            doc_data = self.search_algorithm.get_document_by_id(doc_id)
            
            if doc_data:
                # Get description with fallback to full_text or empty string
                description = doc_data.get('description')
                if not description or (isinstance(description, str) and description.strip() == ''):
                    description = doc_data.get('full_text', '')
                if not description or (isinstance(description, str) and description.strip() == ''):
                    description = doc_data.get('title', '')
                
                # Handle product_details - ensure it's a dict or None
                product_details = doc_data.get('product_details')
                if product_details is not None and not isinstance(product_details, dict):
                    product_details = None
                
                # Handle doc_date - convert timestamp to string if needed
                crawled_at = doc_data.get('crawled_at')
                if crawled_at is None:
                    doc_date = ''
                elif isinstance(crawled_at, (int, float)):
                    # Convert timestamp to string
                    doc_date = str(crawled_at)
                else:
                    doc_date = str(crawled_at)
                
                # Get original URL from corpus
                original_url = doc_data.get('url')
                
                # Create result document from processed corpus data
                result_doc = Document(
                    pid=doc_data.get('pid', doc_id),
                    title=doc_data.get('title', 'N/A'),
                    description=description,
                    brand=doc_data.get('brand'),
                    category=doc_data.get('category'),
                    sub_category=doc_data.get('sub_category'),
                    product_details=product_details,
                    seller=doc_data.get('seller'),
                    out_of_stock=doc_data.get('out_of_stock', False),
                    selling_price=doc_data.get('selling_price'),
                    discount=doc_data.get('discount'),
                    actual_price=doc_data.get('actual_price'),
                    average_rating=doc_data.get('average_rating'),
                    url=f"doc_details?pid={doc_data.get('pid', doc_id)}&search_id={search_id}",
                    original_url=original_url,
                    images=doc_data.get('images'),
                    ranking=score,
                    doc_date=doc_date
                )
                results.append(result_doc)
            else:
                # Fallback to display corpus if not found in processed corpus
                doc = corpus.get(doc_id)
                if doc:
                    result_doc = Document(
                        pid=doc.pid,
                        title=doc.title or 'N/A',
                        description=doc.description or '',
                        brand=doc.brand,
                        category=doc.category,
                        sub_category=doc.sub_category,
                        product_details=doc.product_details,
                        seller=doc.seller,
                        out_of_stock=doc.out_of_stock,
                        selling_price=doc.selling_price,
                        discount=doc.discount,
                        actual_price=doc.actual_price,
                        average_rating=doc.average_rating,
                        url=f"doc_details?pid={doc.pid}&search_id={search_id}",
                        original_url=doc.url,
                        images=doc.images,
                        ranking=score
                    )
                    results.append(result_doc)
        
        return results
