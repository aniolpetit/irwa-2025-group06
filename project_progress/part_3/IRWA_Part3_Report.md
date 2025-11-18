# Part 3: Ranking & Filtering
**Github URL**: https://github.com/aniolpetit/irwa-2025-group06

**Github TAG**: IRWA-2025-part-3

**Date:** November 11, 2025

---

*Note: At the end of the report there's an Appendix showing with results of the code execution, showing the rankings for every metric* 

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
Our custom ranking function builds on TF-IDF + cosine and adds field importance, proximity, and product metadata to better reflect an e-commerce scenario. We keep the TF-IDF cosine score as the main textual relevance signal, using the same index as in Part 2.

**Field-Aware Boosting:**
Query matches in different fields are weighted unequally:
  - *Title*: 1.0 (highest weight, as titles are most indicative of relevance)
  - *Brand*: 0.6 (important for brand-specific queries)
  - *Subcategory*: 0.5 (helps with categorical matching)
  - *Details*: 0.3 (moderate importance)
  - *Description*: 0.2 (lowest weight, as descriptions can be verbose and less focused)

This fixes the “all fields equal” limitation from Part 2: a match in the title or brand matters more than a match buried in a long description. For “ecko unl shirt”, top results are shirts whose titles and brand clearly match the query, even if the description is "N/A".

**Term Proximity Score:**
We add a proximity score based on the minimal span that covers all query terms:
- Score: `1 / (1 + best_span)`, where smaller spans yield higher scores
- This rewards documents where query terms appear close together, using a minimal-span algorithm to find the shortest distance that covers all query terms.

This favors documents where terms like “women polo cotton” or “biowash innerwear” appear close together rather than scattered.

**Metadata Signals:**
We use numerical fields as hinted: 
- **Average Rating Boost**: Higher-rated products receive a small boost (normalized to 0-1 scale), reflecting user satisfaction as a relevance signal.
- **Out-Of-Stock Penalty**: Products that are out of stock are penalized, as they are less useful to users even if textually relevant.

For **“women polo cotton”**, the top items are polo t-shirts with good ratings and reasonable discounts; for **“biowash innerwear”**, biowashed vests with rating 3.9 and in stock appear at the top. So metadata refines the ranking among textually similar products.

**Length Normalization:**
- Very long descriptions are slightly penalized through a logarithmic length factor, to avoid template-like text dominating and to punish documents with longer-than-average description lenghts. (A small exact match bonus is added if the full query string appears in the title. ES CORRECTE AIXÒ?) 
- It helps preventing verbose, keyword-stuffed descriptions from dominating the ranking
- Formula: `penalty = 1 / (1 + λ × log(ratio))` where ratio is description length relative to average

Conjunctive retrieval is still used, so “ecko unl men shirt round neck” and “casual clothes slim fit” return no results, as in TF-IDF and BM25.

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

**Pros**
- *Multi-signal relevance*: The score combines textual similarity (TF-IDF), field importance, proximity, and product metadata (ratings, stock status), producing richer rankings than TF-IDF or BM25 alone.
- *E-commerce oriented*: Metadata such as rating and availability matter in online shopping; integrating them helps separate highly similar products.
- *Fixes earlier limitations*: Field weighting corrects the uniform-field issue seen in Part 2, and proximity captures phrase-level relevance ignored by TF-IDF and BM25.
- *Tunable*: All weights are explicit and can be adapted to observations or evaluation metrics.

**Cons**
- *Higher complexity*: The score is less interpretable and harder to debug than classic IR models.
- *Sensitive to parameters*: Field weights, proximity weights, and boosts require tuning and may vary by domain or query type.
- *More expensive*: Field-aware matching and proximity calculations increase computational cost.
- *Still title-biased*: Because TF-IDF + cosine is the base, long descriptions remain penalized, which explains why many top results (e.g., “ecko unl shirt”) still show `description = "N/A"`.

**Design Choices**
- *TF-IDF as the base*: We kept TF-IDF instead of BM25 to remain consistent with earlier parts and because field weights and proximity partially compensate for BM25’s advantages.
- *Field weights reflect semantic importance*: Title > brand > subcategory > details > description, mirroring how users interpret product relevance.
- *Minimal-span proximity*: We use the shortest window covering all query terms (rather than average distance) since it captures phrase-like structures more accurately.
- *Lightweight metadata integration*: Rating and stock availability are included with small weights so they refine, but do not dominate the score.


## 2. Preprocessing Improvements: Handling Hyphenated Terms

In Part 2, we identified a critical issue with how hyphenated clothing terms were handled during preprocessing. The original dataset contains titles with hyphenated terms like "T-Shirt" or "t-shirt", but during preprocessing, these were being converted to "t shirt" (with a space), which then tokenized into separate tokens ["t", "shirt"]. This caused semantic confusion: a query for "shirt" would incorrectly match documents containing "t-shirt" because both shared the token "shirt", despite representing fundamentally different product types. This problem was evident in two ways: (1) for the query "ecko unl shirt", TF-IDF results included documents with "t-shirt" in the title, which were marked as non-relevant since t-shirts are distinct from shirts; (2) for the query "ecko unl men shirt round neck", all retrieved documents were marked as non-relevant because they all contained "t-shirt" rather than "shirt".

To address this issue, we modified the preprocessing pipeline in Part 1 to properly handle hyphenated terms. The key changes in the `clean_text()` function include normalizing hyphen variants (en-dash, em-dash) to standard hyphens and ensuring that both spaced versions ("t shirt") and already-hyphenated versions ("t-shirt") are normalized to "t-shirt" before tokenization. This ensures that NLTK's `word_tokenize()` treats "t-shirt" as a single token rather than splitting it. As a result, the corpus now contains tokens like "t-shirt" and "round-neck" as atomic units.

The impact of this fix is visible in our current results: for the query "ecko unl shirt", we now retrieve 679 relevant documents, all of which are actual shirts (not t-shirts), demonstrating that "shirt" queries no longer incorrectly match "t-shirt" documents. However, this fix also reveals a limitation: for queries like "ecko unl men shirt round neck", we get no documents matching all query terms because the query contains "round" and "neck" as separate tokens, while documents contain "round-neck" as a single token. This highlights that queries need to be normalized using the same rules as documents to achieve proper matching, which is a consideration for future improvements.

## 3. Ranking Methods Implementation: Word2Vec + Cosine

### 3.1 Empirical Results
Running `word2vec_cosine_search.py` with the same five benchmark queries reveals how the semantic averaging approach behaves under conjunctive filtering.

- **“ecko unl shirt”** still returns the 679 true-shirt products uncovered after the hyphenation fix, but rankings look flatter than with TF-IDF/BM25: Word2Vec surface scores stay within a narrow band (≈0.46–0.48) and mix men’s and women’s shirts interchangeably. Because the model focuses on embedding similarity rather than field-specific signals, descriptive nuances (collar type, gender) barely influence ordering even when present in the metadata.
- **“ecko unl men shirt round neck”** yields no hits, mirroring the lexical models. Since conjunctive matching happens before embedding scoring, Word2Vec cannot rescue queries that contain terms absent from the filtered postings list (e.g., “round”/“neck” vs. “round-neck”). This underlines that semantic matching only applies to ranking, not retrieval.
- **“women polo cotton”** demonstrates Word2Vec’s broader semantic reach: alongside polo t-shirts, the top ranks now include cotton trousers and loungewear from the same brand, indicating that embeddings capture brand-level proximity but sometimes blur categorical intent. Items with verbose descriptions packed with brand boilerplate dominate because every word in the long text contributes to the averaged document vector and pulls it closer to the query direction.
- **“casual clothes slim fit”** again returns zero results due to the AND semantics—Word2Vec cannot compensate for missing exact tokens such as “clothes.”
- **“biowash innerwear”** produces 59 matches, but their cosine scores are negative and tightly grouped (≈−0.18 to −0.19). This happens because most candidate documents belong to the same “sayitloud” vest family with repetitive descriptions; averaging many near-identical vectors collapses their distinction, so the ranker offers little separation beyond arbitrary ordering.

Overall, Word2Vec + cosine brings semantic smoothing once candidate documents exist, but it still inherits the conjunctive recall ceiling and can over-generalize toward brand or style cues while ignoring fine-grained categorical preferences.

### 3.2 Word2Vec Limitations and Better Alternatives

While Word2Vec provides a solid foundation for semantic search, our implementation has inherent limitations that could be addressed with more sophisticated embedding approaches.

The main weakness of our Word2Vec + averaging approach is that it treats documents as simple bags of words. By averaging individual word vectors, we lose crucial information about word order, context, and document-level semantics. For example, "women polo cotton" and "cotton polo women" would produce identical document vectors despite having different meanings, and we cannot distinguish between documents where query terms appear as a cohesive phrase versus scattered throughout the text.

**Doc2Vec (Paragraph Vector)** offers a direct improvement by learning document-level embeddings rather than composing them from word vectors. Doc2Vec extends Word2Vec with a document ID that acts as a memory token, allowing the model to capture document-specific semantics. The two main variants are PV-DM (Distributed Memory), which predicts words from context and document ID, and PV-DBOW (Distributed Bag of Words), which predicts context words from the document ID. The key advantage is that Doc2Vec embeddings are optimized specifically for document-level tasks, potentially capturing thematic coherence and document structure better than averaged word vectors. However, Doc2Vec requires training on your corpus, which adds significant computational overhead and time compared to using pre-trained Word2Vec models. It also needs careful hyperparameter tuning and may struggle with very short documents or queries.

**Sentence2Vec (Sentence Embeddings)** refers to models specifically designed to encode entire sentences or short texts into fixed-size vectors. Popular examples include Universal Sentence Encoder, Sentence-BERT, and similar transformer-based or siamese network architectures trained on sentence similarity tasks. They excel at capturing semantic relationships between phrases and can understand that "women's polo shirt" and "ladies polo top" are semantically similar even with minimal word overlap. Pre-trained sentence embedding models are readily available and can be used out-of-the-box, making them practical for many applications. The main drawbacks are increased computational cost compared to Word2Vec, potential overfitting to the training domain, and the fact that they may be overkill for very short queries or when exact term matching is important.

For our e-commerce use case, **Doc2Vec would likely provide the best balance** between improvement and practicality. It directly addresses the document-level representation problem without requiring the massive computational resources of transformer-based sentence encoders, and it can be fine-tuned on our product corpus to capture domain-specific semantics. Sentence2Vec would offer superior semantic understanding but at a higher cost, and might actually hurt performance for queries where exact brand names or product codes matter. Ultimately, the choice depends on the trade-off between semantic sophistication and computational efficiency that fits your specific requirements.

## APPENDIX


