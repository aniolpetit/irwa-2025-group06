from typing import List, Optional
from urllib.parse import urlparse

from myapp.search.objects import Document
from myapp.search.algorithms import SearchAlgorithm


STOP_BRAND_TOKENS = {
    "men", "man", "women", "woman", "boys", "boy", "girls", "girl",
    "solid", "printed", "print", "striped", "strip", "slim", "fit", "regular", "relaxed",
    "round", "v", "neck", "crew", "tshirt", "t-shirt", "shirt", "t", "dress", "kurta",
    "top", "tops", "jeans", "denim", "track", "tracks", "pants", "pant", "shorts", "short",
    "jacket", "hoodie", "sweatshirt", "sweater", "sweat", "trouser", "capri", "legging", "leggings",
    "palazzo", "set", "combo", "pack", "cotton", "polyester", "blend", "full", "half",
    "sleeve", "sleeves", "three", "quarter", "ankle", "length", "graphic", "logo",
    "collar", "henley", "polo", "mock", "high", "rise", "mid", "low", "waist",
    "sports", "sport", "running", "gym", "yoga", "casual", "formal", "ethnic", "party",
    "wedding", "plain", "classic", "basic", "essential", "boys'", "girls'"
}


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
    
    def _infer_brand_from_url(self, url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        path = urlparse(url).path
        if not path:
            return None
        slug = next((segment for segment in path.split('/') if segment), '')
        if not slug:
            return None
        tokens = [token for token in slug.split('-') if token]
        brand_tokens: List[str] = []
        for token in tokens:
            token = token.strip().lower()
            if not token or token.isdigit() or token in STOP_BRAND_TOKENS:
                break
            brand_tokens.append(token)
        if not brand_tokens:
            return None
        brand = ' '.join(token.capitalize() for token in brand_tokens)
        return brand if len(brand) >= 3 else None

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
        for position, (doc_id, score) in enumerate(ranked_results, start=1):
            # Get document data from search algorithm's corpus (processed corpus with all fields)
            doc_data = self.search_algorithm.get_document_by_id(doc_id)
            display_doc = corpus.get(doc_id)
            brand_display = None
            category_display = None
            seller_display = None
            if display_doc:
                brand_display = display_doc.brand or None
                category_display = display_doc.category or None
                seller_display = display_doc.seller or None
            
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
                brand_value = brand_display or doc_data.get('brand')
                if not brand_value or len(brand_value) <= 2:
                    inferred = self._infer_brand_from_url(
                        original_url or (display_doc.original_url if display_doc else None)
                    )
                    if inferred:
                        brand_value = inferred
                
                # Create result document from processed corpus data
                result_doc = Document(
                    pid=doc_data.get('pid', doc_id),
                    title=doc_data.get('title', 'N/A'),
                    description=description,
                    brand=brand_value,
                    category=category_display or doc_data.get('category'),
                    sub_category=doc_data.get('sub_category'),
                    product_details=product_details,
                    seller=seller_display or doc_data.get('seller'),
                    out_of_stock=doc_data.get('out_of_stock', False),
                    selling_price=doc_data.get('selling_price'),
                    discount=doc_data.get('discount'),
                    actual_price=doc_data.get('actual_price'),
                    average_rating=doc_data.get('average_rating'),
                    url=f"doc_details?pid={doc_data.get('pid', doc_id)}&search_id={search_id}&rank={position}",
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
                    brand_value = doc.brand
                    if not brand_value or len(brand_value) <= 2:
                        inferred = self._infer_brand_from_url(doc.original_url or doc.url)
                        if inferred:
                            brand_value = inferred
                    result_doc = Document(
                        pid=doc.pid,
                        title=doc.title or 'N/A',
                        description=doc.description or '',
                        brand=brand_value,
                        category=doc.category,
                        sub_category=doc.sub_category,
                        product_details=doc.product_details,
                        seller=doc.seller,
                        out_of_stock=doc.out_of_stock,
                        selling_price=doc.selling_price,
                        discount=doc.discount,
                        actual_price=doc.actual_price,
                        average_rating=doc.average_rating,
                        url=f"doc_details?pid={doc.pid}&search_id={search_id}&rank={position}",
                        original_url=doc.url,
                        images=doc.images,
                        ranking=score
                    )
                    results.append(result_doc)
        
        return results
