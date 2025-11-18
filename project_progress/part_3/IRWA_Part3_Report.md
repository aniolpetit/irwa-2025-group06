# Part 3: Ranking & Filtering

## 1. Ranking Methods Implementation: TF-IDF + Cosine, BM25 and Custom Score

### 1.1 TF-IDF + Cosine Similarity

For the TF-IDF + cosine similarity ranking, we reused the implementation from Part 2 of the project (`TFIDFRanker` class). The ranking follows the classical approach:

- **TF (Term Frequency)**: Uses logarithmic weighting: `w_tf = 1 + log₂(freq)`
- **IDF (Inverse Document Frequency)**: `w_idf = log₂(N / df)` where N is the total number of documents and df is the document frequency
- **TF-IDF Weight**: `w = w_tf × w_idf`
- **Document Length Normalization**: `length(d) = sqrt(Σ w_i²)`
- **Cosine Similarity Score**: `score(d, q) = (query_vector · doc_vector) / length(d)`

The implementation performs conjunctive query filtering (AND) to find all documents containing all query terms, then ranks them using the cosine similarity between the TF-IDF-weighted query and document vectors.

### 1.2 BM25

BM25 is a probabilistic ranking function that improves upon TF-IDF in two key ways. First, it implements term frequency saturation, meaning that as a term appears more frequently in a document, each additional occurrence contributes less to the relevance score. This prevents documents from being over-rewarded simply because they repeat query terms many times, which is more realistic than TF-IDF's roughly linear relationship between term frequency and score. Second, BM25 uses tunable parameters to balance document length normalization, avoiding TF-IDF's tendency to over-penalize longer documents. These improvements make BM25 particularly effective for collections with documents of varying lengths, as it allows longer documents with detailed descriptions to rank appropriately without being unfairly penalized, while also preventing keyword-stuffed documents from dominating the results.

Our implementation uses a simplified version of BM25 that applies conjunctive query filtering before ranking. The ranking formula is:

```
score(d, q) = Σ [idf(t) × (tf(t,d) × (k1 + 1)) / (tf(t,d) + k1 × (1 - b + b × |d|/avgdl))]
```

Where:
- `idf(t) = log(N / df(t))` (simplified IDF formula)
- `tf(t,d)` is the term frequency of term t in document d
- `|d|` is the document length in tokens
- `avgdl` is the average document length
- `k1 = 1.2` (TREC value, controls term frequency saturation)
- `b = 0.75` (TREC value, controls length normalization)

Note that we use a simplified IDF formula `log(N/df)` rather than the full BM25 IDF formula `log((N - df + 0.5) / (df + 0.5))`, and we do not weight query term frequency (no k3 parameter), making this a streamlined version suitable for our use case.


### 1.2.1 Comparison: TF-IDF vs BM25

We compared **TF-IDF + cosine** similarity and **BM25** on our fashion e-commerce corpus using conjunctive queries. Since all returned documents contain all query terms, the main differences come from how each method handles **document length** and **term frequency**, especially in a setting with short titles and much richer descriptions.

For queries like **“ecko unl shirt”** or **“women polo cotton”**, we observed that TF-IDF tends to favor short, title-driven documents, often with *description = "N/A"*. Cosine normalization over-penalizes longer descriptions, so products whose titles match the query terms get boosted even if their descriptions add little information. In contrast, **BM25 ranks higher those items with longer, more detailed product descriptions** where the query terms appear multiple times (e.g., “ecko unltd slim fit cotton woven regular navy blue shirt” for the query “ecko unl shirt”). Its length normalization parameter (b) and TF saturation let descriptive documents benefit from repeated relevant terms without being overly punished for being longer.

For **“women polo cotton”**, the difference becomes even clearer. TF-IDF ranks polo t-shirts with very short descriptions (3–6 words), while BM25 pushes to the top items with long product descriptions containing multiple repetitions of “cotton”, “polo”, “t-shirt”, and related attributes. Some descriptions explicitly contain patterns like **“single jersey cotton slim fit t-shirt”**, which naturally receive higher BM25 scores. This shows BM25’s advantage in attribute-rich domains, where meaningful information is often found in long descriptions rather than in titles alone.

For **“ecko unl men shirt round neck”**, both models return zero results. This highlights a limitation of conjunctive retrieval: if any query term (e.g., “round” or “neck”) does not appear in documents that already satisfy the other terms, the entire query fails regardless of the ranking model. 

The query **“casual clothes slim fit”** also returns no results in both models, reinforcing the same limitation: conjunctive (AND) semantics can be too strict when rare terms are included.

Overall, in this corpus BM25 provides rankings that are **more aligned with what we would expect from a shopping scenario**: it better exploits detailed descriptions and rare terms, and is more robust to varying document length. TF-IDF remains a useful and simple baseline, especially for short, well-focused documents, but it underuses the rich textual information available in many product descriptions.

**Pros & Cons Summary**

- TF-IDF is simple, efficient, and reliable for short fields such as product titles. However, its cosine normalization tends to suppress long descriptions and undervalue repeated attribute terms, which is a drawback in our dataset where descriptions often contain crucial fashion-related attributes.

- BM25, on the other hand, models term frequency saturation more realistically and handles length normalization more flexibly. This makes it better suited for our heterogeneous corpus, though it introduces additional parameters (k1, b) and can overly favor long template-like descriptions. Despite these trade-offs, BM25 generally produces rankings that feel more consistent with user search intent in a product-oriented search engine.

### 1.3 Custom Score

Our custom ranking function combines multiple signals to improve relevance ranking beyond what TF-IDF or BM25 alone can achieve. The score integrates:

**Base Component:**
- **TF-IDF cosine similarity** as the foundation, providing a solid baseline for term-based relevance

**Field-Aware Boosting:**
- Different weights for query term matches in different document fields:
  - Title: 1.0 (highest weight, as titles are most indicative of relevance)
  - Brand: 0.6 (important for brand-specific queries)
  - Subcategory: 0.5 (helps with categorical matching)
  - Details: 0.3 (moderate importance)
  - Description: 0.2 (lowest weight, as descriptions can be verbose and less focused)

This addresses the limitation identified in Part 2 where all fields were treated uniformly. A term match in the title should be more valuable than one in a long description.

**Term Proximity Score:**
- Rewards documents where query terms appear close together, using a minimal-span algorithm to find the shortest distance covering all query terms
- Score: `1 / (1 + best_span)`, where smaller spans yield higher scores
- This captures phrase-level relevance: documents where "women polo cotton" appear as a phrase are more relevant than those where the terms are scattered

**Metadata Signals:**
- **Average rating boost**: Higher-rated products receive a small boost (normalized to 0-1 scale), reflecting user satisfaction as a relevance signal
- **Out-of-stock penalty**: Products that are out of stock are penalized, as they are less useful to users even if textually relevant

**Length Normalization:**
- Applies a logarithmic penalty to documents with longer-than-average descriptions
- Prevents verbose, keyword-stuffed descriptions from dominating the ranking
- Formula: `penalty = 1 / (1 + λ × log(ratio))` where ratio is description length relative to average

**Exact Match Bonus:**
- Small bonus when the entire query string appears in the document title, rewarding perfect matches

**Final Score Formula:**
```
score = (TF-IDF_base) 
      + (field_weight_scale × field_score)
      + (proximity_weight × proximity_score)
      + (rating_weight × normalized_rating)
      - (out_of_stock_penalty)
      + (exact_match_bonus)
score = score × length_factor
```

**Justification and Trade-offs:**

**Pros:**
- **Multi-signal integration**: Combines textual relevance (TF-IDF), structural importance (field weights), semantic proximity, and user value signals (ratings, availability)
- **Domain-specific optimization**: Tailored for e-commerce search where metadata (ratings, stock status) directly impacts user satisfaction
- **Addresses known limitations**: Directly tackles issues identified in Part 2: field uniformity, lack of phrase awareness, and ignoring product quality signals
- **Flexible weighting**: All components are parameterized, allowing fine-tuning based on evaluation results

**Cons:**
- **Complexity**: More complex than TF-IDF or BM25, making it harder to interpret and debug
- **Parameter sensitivity**: Requires careful tuning of multiple weights (field weights, proximity weight, rating weight, etc.) which may need query-specific or domain-specific optimization
- **Computational overhead**: Field-aware scoring and proximity calculation add computational cost compared to simpler methods
- **Description more penalized**: using tf-idf as the base model penalizes long descriptions, reflecting that in the output where most of the top-20 results have as description "N/A"

**Design Choices:**
- We chose TF-IDF as the base rather than BM25 to maintain consistency with Part 2 and because the additional signals (field weights, proximity) already address some of BM25's advantages
- Field weights were set based on intuitive importance (title > brand > category > description)
- The proximity score uses a minimal-span approach rather than average distance, as it better captures phrase-level relevance
- Rating and stock signals use small weights to avoid overwhelming textual relevance, but still provide meaningful differentiation

## 2. Preprocessing Improvements: Handling Hyphenated Terms

In Part 2, we identified a critical issue with how hyphenated clothing terms were handled during preprocessing. The original dataset contains titles with hyphenated terms like "T-Shirt" or "t-shirt", but during preprocessing, these were being converted to "t shirt" (with a space), which then tokenized into separate tokens ["t", "shirt"]. This caused semantic confusion: a query for "shirt" would incorrectly match documents containing "t-shirt" because both shared the token "shirt", despite representing fundamentally different product types. This problem was evident in two ways: (1) for the query "ecko unl shirt", TF-IDF results included documents with "t-shirt" in the title, which were marked as non-relevant since t-shirts are distinct from shirts; (2) for the query "ecko unl men shirt round neck", all retrieved documents were marked as non-relevant because they all contained "t-shirt" rather than "shirt".

To address this issue, we modified the preprocessing pipeline in Part 1 to properly handle hyphenated terms. The key changes in the `clean_text()` function include normalizing hyphen variants (en-dash, em-dash) to standard hyphens and ensuring that both spaced versions ("t shirt") and already-hyphenated versions ("t-shirt") are normalized to "t-shirt" before tokenization. This ensures that NLTK's `word_tokenize()` treats "t-shirt" as a single token rather than splitting it. As a result, the corpus now contains tokens like "t-shirt" and "round-neck" as atomic units.

The impact of this fix is visible in our current results: for the query "ecko unl shirt", we now retrieve 679 relevant documents, all of which are actual shirts (not t-shirts), demonstrating that "shirt" queries no longer incorrectly match "t-shirt" documents. However, this fix also reveals a limitation: for queries like "ecko unl men shirt round neck", we get no documents matching all query terms because the query contains "round" and "neck" as separate tokens, while documents contain "round-neck" as a single token. This highlights that queries need to be normalized using the same rules as documents to achieve proper matching, which is a consideration for future improvements.

## 3. Ranking Methods Implementation: Word2Vec + Cosine


### 3.1 Word2Vec Limitations and Better Alternatives

While Word2Vec provides a solid foundation for semantic search, our implementation has inherent limitations that could be addressed with more sophisticated embedding approaches.

The main weakness of our Word2Vec + averaging approach is that it treats documents as simple bags of words. By averaging individual word vectors, we lose crucial information about word order, context, and document-level semantics. For example, "women polo cotton" and "cotton polo women" would produce identical document vectors despite having different meanings, and we cannot distinguish between documents where query terms appear as a cohesive phrase versus scattered throughout the text.

**Doc2Vec (Paragraph Vector)** offers a direct improvement by learning document-level embeddings rather than composing them from word vectors. Doc2Vec extends Word2Vec with a document ID that acts as a memory token, allowing the model to capture document-specific semantics. The two main variants are PV-DM (Distributed Memory), which predicts words from context and document ID, and PV-DBOW (Distributed Bag of Words), which predicts context words from the document ID. The key advantage is that Doc2Vec embeddings are optimized specifically for document-level tasks, potentially capturing thematic coherence and document structure better than averaged word vectors. However, Doc2Vec requires training on your corpus, which adds significant computational overhead and time compared to using pre-trained Word2Vec models. It also needs careful hyperparameter tuning and may struggle with very short documents or queries.

**Sentence2Vec (Sentence Embeddings)** refers to models specifically designed to encode entire sentences or short texts into fixed-size vectors. Popular examples include Universal Sentence Encoder, Sentence-BERT, and similar transformer-based or siamese network architectures trained on sentence similarity tasks. They excel at capturing semantic relationships between phrases and can understand that "women's polo shirt" and "ladies polo top" are semantically similar even with minimal word overlap. Pre-trained sentence embedding models are readily available and can be used out-of-the-box, making them practical for many applications. The main drawbacks are increased computational cost compared to Word2Vec, potential overfitting to the training domain, and the fact that they may be overkill for very short queries or when exact term matching is important.

For our e-commerce use case, **Doc2Vec would likely provide the best balance** between improvement and practicality. It directly addresses the document-level representation problem without requiring the massive computational resources of transformer-based sentence encoders, and it can be fine-tuned on our product corpus to capture domain-specific semantics. Sentence2Vec would offer superior semantic understanding but at a higher cost, and might actually hurt performance for queries where exact brand names or product codes matter. Ultimately, the choice depends on the trade-off between semantic sophistication and computational efficiency that fits your specific requirements.

