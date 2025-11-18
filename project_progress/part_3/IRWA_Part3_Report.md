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
**Tf-Idf + Cosine Results:**
Query: ecko unl shirt
Query terms: ['ecko', 'unl', 'shirt']

Total results: 679
Showing top 20 results:

  1. [Score: 4.199581] | PID: SHTFWBZB2BYWYNEE
     Title: men slim fit printed casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: N/A

  2. [Score: 4.055410] | PID: SHTFRR7FHGVAZHTB
     Title: women slim fit printed casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: N/A

  3. [Score: 4.047385] | PID: SHTFWBZ8ARDZDVFF
     Title: men slim fit printed casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd slim fit cotton woven regular navy blue shirt

  4. [Score: 3.999182] | PID: SHTFXV5FFATYDWZT
     Title: men slim fit solid casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: N/A

  5. [Score: 3.997231] | PID: SHTFSKF6ARDQ7RYU
     Title: men slim fit solid casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: N/A

  6. [Score: 3.972969] | PID: SHTFUPTXS69YVUNX
     Title: women slim fit solid casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: N/A

  7. [Score: 3.963152] | PID: SHTFSKF6AKNYH3YR
     Title: men slim fit solid casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: N/A

  8. [Score: 3.923543] | PID: SHTFQ252ZKDSWEZT
     Title: women slim fit solid casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: N/A

  9. [Score: 3.909655] | PID: SHTFXV5EC6JYYVZA
     Title: women slim fit printed casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: N/A

 10. [Score: 3.899316] | PID: SHTFWBZ9XSCJP5BA
     Title: women slim fit printed casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: N/A

 11. [Score: 3.898068] | PID: SHTFWBZ7NRMENHFX
     Title: men slim fit printed casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd slim fit cotton woven regular olive shirt

 12. [Score: 3.893602] | PID: SHTFXV5EQ3AYZEB8
     Title: men slim fit solid casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: N/A

 13. [Score: 3.892624] | PID: SHTFXV5EFHMGTGFG
     Title: women slim fit solid casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: N/A

 14. [Score: 3.874011] | PID: SHTFXV5EVCR468PS
     Title: women slim fit printed casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: N/A

 15. [Score: 3.862603] | PID: SHTFXV5E6PYFPPGH
     Title: men slim fit solid casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: N/A

 16. [Score: 3.852632] | PID: SHTFXV5ENY5FYHPH
     Title: men slim fit solid casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: N/A

 17. [Score: 3.852632] | PID: SHTFXV5F7ZBGAHAH
     Title: men slim fit solid casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: N/A

 18. [Score: 3.834261] | PID: TSHFWDYTDGBAUEMH
     Title: solid women round-neck white t-shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd slim fit cotton regular white t-shirt

 19. [Score: 3.834261] | PID: TSHFWDYR6BTJYYUR
     Title: solid women round-neck white t-shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd slim fit cotton regular white t-shirt

 20. [Score: 3.830654] | PID: SHTFXV5EMMPNBFGF
     Title: men slim fit printed casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: N/A

... and 659 more results (not shown)
================================================================================


Query: ecko unl men shirt round neck
Query terms: ['ecko', 'unl', 'men', 'shirt', 'round', 'neck']
No documents found matching all query terms.
================================================================================

Query: women polo cotton
Query terms: ['women', 'polo', 'cotton']

Total results: 729
Showing top 20 results:

  1. [Score: 1.800157] | PID: TSHFXVJZYXJNUZH6
     Title: solid women polo-neck white t-shirt
     Brand: u s polo ass | Category: clothing and accessories | Sub-category: topwear
     Description: half sleeve polo s

  2. [Score: 1.735609] | PID: TSHFXVJZZPMGZ5QE
     Title: solid women polo-neck red t-shirt
     Brand: u s polo ass | Category: clothing and accessories | Sub-category: topwear
     Description: half sleeve polo l

  3. [Score: 1.482805] | PID: TSHFGNKBMZQFFXCG
     Title: self design women polo-neck dark blue t-shirt
     Brand: u s polo ass | Category: clothing and accessories | Sub-category: topwear
     Description: half sleeve polo t-shirt

  4. [Score: 1.439206] | PID: TSHFJFVBKFMMDPHC
     Title: striped women polo-neck white blue orange t-shirt
     Brand: u s polo ass | Category: clothing and accessories | Sub-category: topwear
     Description: half sleeve polo t-shirt

  5. [Score: 1.427149] | PID: TSHFHF38WGUUTKGU
     Title: striped women polo-neck multicolor t-shirt
     Brand: u s polo ass | Category: clothing and accessories | Sub-category: topwear
     Description: half sleeve polo t-shirt

  6. [Score: 1.416099] | PID: TSHFGQKHTYSS8YWE
     Title: solid women polo-neck dark blue t-shirt
     Brand: pu | Category: clothing and accessories | Sub-category: topwear
     Description: womens graphic polo ii peacoat

  7. [Score: 1.383284] | PID: TSHFPDP6XBUKAYTC
     Title: typography women round-neck grey t-shirt
     Brand: u s polo associati | Category: clothing and accessories | Sub-category: topwear     
     Description: u s polo assn captures the authenticity of polo and stays true to a classic american style updated to complement todays on-the-go lifestyle us polo womens tops t-shirts sycamore cotton s j gsm

  8. [Score: 1.354952] | PID: TSHFPDP64QDDAXDZ
     Title: typography women round-neck red t-shirt
     Brand: u s polo associati | Category: clothing and accessories | Sub-category: topwear     
     Description: u s polo assn captures the authenticity of polo and stays true to a classic american style updated to complement todays on-the-go lifestyle us polo womens tops t-shirts sycamore cotton s j gsm

  9. [Score: 1.349155] | PID: TSHFWCPHXDAHAHZZ
     Title: solid women polo-neck black t-shirt
     Brand: scott internation | Category: clothing and accessories | Sub-category: topwear      
     Description: refresh your wardrobe with the new range of summer collections of organic cotton polo t-shirts from scott international this polo t-shirts are made of is pure premium organic cotton fabric super combed solid polo and gives utmost comfort during all temperatures durable stitch and finish make this t-shirt a perfect formal and casual wear contrast tipping on the collar and sleeves makes this product have a unique look as compared to other products in the polo segment pair it with jeans or casual for a perfect casual or party look

 10. [Score: 1.332935] | PID: TSHFWCPHNECZFJ2E
     Title: solid women polo-neck yellow t-shirt
     Brand: scott internation | Category: clothing and accessories | Sub-category: topwear      
     Description: refresh your wardrobe with the new range of summer collections of organic cotton polo t-shirts from scott international this polo t-shirts are made of is pure premium organic cotton fabric super combed solid polo and gives utmost comfort during all temperatures durable stitch and finish make this t-shirt a perfect formal and casual wear contrast tipping on the collar and sleeves makes this product have a unique look as compared to other products in the polo segment pair it with jeans or casual for a perfect casual or party look

 11. [Score: 1.316427] | PID: TSHFWCPHZSFTENKX
     Title: solid women polo-neck green t-shirt
     Brand: scott internation | Category: clothing and accessories | Sub-category: topwear      
     Description: refresh your wardrobe with the new range of summer collections of organic cotton polo t-shirts from scott international this polo t-shirts are made of is pure premium organic cotton fabric super combed solid polo and gives utmost comfort during all temperatures durable stitch and finish make this t-shirt a perfect formal and casual wear contrast tipping on the collar and sleeves makes this product have a unique look as compared to other products in the polo segment pair it with jeans or casual for a perfect casual or party look

 12. [Score: 1.302422] | PID: TSHFWCPHCSTTHNZR
     Title: solid women polo-neck black t-shirt
     Brand: scott internation | Category: clothing and accessories | Sub-category: topwear      
     Description: refresh your wardrobe with the new range of summer collections of organic cotton polo t-shirts from scott international this polo t-shirts are made of is pure premium organic cotton fabric super combed solid polo and gives utmost comfort during all temperatures durable stitch and finish make this t-shirt a perfect formal and casual wear contrast tipping on the collar and sleeves makes this product have a unique look as compared to other products in the polo segment pair it with jeans or casual for a perfect casual or party look

 13. [Score: 1.287447] | PID: TSHFWJZEMZTBCSJN
     Title: solid women polo-neck white t-shirt
     Brand: scott internation | Category: clothing and accessories | Sub-category: topwear      
     Description: refresh your wardrobe with the new range of summer collections of organic cotton polo t-shirts from scott international this polo t-shirts are made of is pure premium organic cotton fabric super combed solid polo and gives utmost comfort during all temperatures durable stitch and finish make this t-shirt a perfect formal and casual wear contrast tipping on the collar and sleeves makes this product have a unique look as compared to other products in the polo segment pair it with jeans or casual for a perfect casual or party look

 14. [Score: 1.283834] | PID: TSHFWJZEVGQPDYZH
     Title: solid women polo-neck grey t-shirt
     Brand: scott internation | Category: clothing and accessories | Sub-category: topwear      
     Description: refresh your wardrobe with the new range of summer collections of organic cotton polo t-shirts from scott international this polo t-shirts are made of is pure premium organic cotton fabric super combed solid polo and gives utmost comfort during all temperatures durable stitch and finish make this t-shirt a perfect formal and casual wear contrast tipping on the collar and sleeves makes this product have a unique look as compared to other products in the polo segment pair it with jeans or casual for a perfect casual or party look

 15. [Score: 1.277270] | PID: TSHFWJZESHNWDPNZ
     Title: solid women polo-neck blue t-shirt
     Brand: scott internation | Category: clothing and accessories | Sub-category: topwear      
     Description: refresh your wardrobe with the new range of summer collections of organic cotton polo t-shirts from scott international this polo t-shirts are made of is pure premium organic cotton fabric super combed solid polo and gives utmost comfort during all temperatures durable stitch and finish make this t-shirt a perfect formal and casual wear contrast tipping on the collar and sleeves makes this product have a unique look as compared to other products in the polo segment pair it with jeans or casual for a perfect casual or party look

 16. [Score: 1.275732] | PID: TSHFWCPHZSWGYXGA
     Title: solid women polo-neck grey t-shirt
     Brand: scott internation | Category: clothing and accessories | Sub-category: topwear      
     Description: refresh your wardrobe with the new range of summer collections of organic cotton polo t-shirts from scott international this polo t-shirts are made of is pure premium organic cotton fabric super combed solid polo and gives utmost comfort during all temperatures durable stitch and finish make this t-shirt a perfect formal and casual wear contrast tipping on the collar and sleeves makes this product have a unique look as compared to other products in the polo segment pair it with jeans or casual for a perfect casual or party look

 17. [Score: 1.272188] | PID: TSHFWJZEGU7QFZMA
     Title: solid women polo-neck yellow t-shirt
     Brand: scott internation | Category: clothing and accessories | Sub-category: topwear      
     Description: refresh your wardrobe with the new range of summer collections of organic cotton polo t-shirts from scott international this polo t-shirts are made of is pure premium organic cotton fabric super combed solid polo and gives utmost comfort during all temperatures durable stitch and finish make this t-shirt a perfect formal and casual wear contrast tipping on the collar and sleeves makes this product have a unique look as compared to other products in the polo segment pair it with jeans or casual for a perfect casual or party look

 18. [Score: 1.270389] | PID: TSHFWCPHNFFZAWHB
     Title: solid women polo-neck grey t-shirt
     Brand: scott internation | Category: clothing and accessories | Sub-category: topwear      
     Description: refresh your wardrobe with the new range of summer collections of organic cotton polo t-shirts from scott international this polo t-shirts are made of is pure premium organic cotton fabric super combed solid polo and gives utmost comfort during all temperatures durable stitch and finish make this t-shirt a perfect formal and casual wear contrast tipping on the collar and sleeves makes this product have a unique look as compared to other products in the polo segment pair it with jeans or casual for a perfect casual or party look

 19. [Score: 1.266153] | PID: TSHEMA3YP8FYVBUF
     Title: solid women polo-neck beige t-shirt
     Brand: t spor | Category: clothing and accessories | Sub-category: topwear
     Description: t sports beige half polo t-shirt for women

 20. [Score: 1.259635] | PID: TSHFB4C7PVTBNGE4
     Title: solid women polo-neck blue t-shirt
     Brand: pu | Category: clothing and accessories | Sub-category: topwear
     Description: ess jersey polo

... and 709 more results (not shown)
================================================================================


Query: casual clothes slim fit
Query terms: ['casual', 'clothes', 'slim', 'fit']
No documents found matching all query terms.
================================================================================

Query: biowash innerwear
Query terms: ['biowash', 'innerwear']

Total results: 59
Showing top 20 results:

  1. [Score: 1.326865] | PID: VESFX3ZFMVZDRS4E
     Title: free authority men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: look trendy and feel comfortable with this character printed sleevless vest featuring friends crafted out of cotton which is biowashed for smooth feel and befriend to skin this featuring can be worn for any occasion a casual day at work or for a fun filled weekend or leisure wear

  2. [Score: 1.280713] | PID: VESFX3ZFC7XXJ5CR
     Title: free authority women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: look trendy and feel comfortable with this character printed sleevless vest featuring nasa crafted out of cotton which is biowashed for smooth feel and befriend to skin this featuring can be worn for any occasion a casual day at work or for a fun filled weekend or leisure wear

  3. [Score: 1.261435] | PID: VESFX3ZFJEKDYZPW
     Title: free authority women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: look trendy and feel comfortable with this character printed sleevless vest featuring the simpsons crafted out of cotton which is biowashed for smooth feel and befriend to skin this featuring can be worn for any occasion a casual day at work or for a fun filled weekend or leisure wear

  4. [Score: 1.234476] | PID: VESFX3ZFZFUJBJVV
     Title: free authority women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: look trendy and feel comfortable with this character printed sleevless vest featuring dragon ball z crafted out of cotton which is biowashed for smooth feel and befriend to skin this featuring can be worn for any occasion a casual day at work or for a fun filled weekend or leisure wear

  5. [Score: 1.121458] | PID: VESFX3ZFXTNQXDFN
     Title: free authority men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: look trendy and feel comfortable with this character printed sleevless vest featuring scooby doo crafted out of cotton which is biowashed for smooth feel and befriend to skin this featuring can be worn for any occasion a casual day at work or for a fun filled weekend or leisure wear

  6. [Score: 0.724028] | PID: BXRFKYKYF3SGUMFH
     Title: printed men boxer pack of
     Brand: sayitlo | Category: clothing and accessories | Sub-category: innerwear and swimwear 
     Description: jump into the comfort zone with the most trendy looking boxers a special elastic which leaves no mark and sweat absorbing

  7. [Score: 0.719805] | PID: VESFGYJDFBUZG6FQ
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest

  8. [Score: 0.713647] | PID: VESFGYJDWUH364V2
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest

  9. [Score: 0.657845] | PID: VESFHJHP4AYNWSAZ
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 10. [Score: 0.656428] | PID: VESFHRDCY32K427Y
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe brand color olive green cotton regular fit occasion casual round-neck western wear

 11. [Score: 0.655563] | PID: VESFHRDAQJ7PTFMM
     Title: sayitloud men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe brand color black white cotton regular fit occasion casual round-neck western wear

 12. [Score: 0.654147] | PID: VESFHJHPGC9CW6XA
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 13. [Score: 0.653563] | PID: VESFHRDBQWKBFHCZ
     Title: sayitloud men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe brand color steel grey cotton regular fit occasion casual round-neck western wear

 14. [Score: 0.650930] | PID: VESFHJHPBCYZGZHE
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 15. [Score: 0.648553] | PID: VESFHJHQREFDNKRQ
     Title: sayitloud men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 16. [Score: 0.648350] | PID: VESFHJHQPTY2ZC8E
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 17. [Score: 0.648051] | PID: VESFHJHPZUSZP2H2
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 18. [Score: 0.647999] | PID: VESFHJHPGM4MJADF
     Title: sayitloud men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 19. [Score: 0.647418] | PID: VESFHJHPZJUBPS49
     Title: sayitloud men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 20. [Score: 0.647157] | PID: VESFHJHPCZH2ACVK
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

... and 39 more results (not shown)
================================================================================

**BM25 Results:**
Query: ecko unl shirt
Query terms: ['ecko', 'unl', 'shirt']

Total results: 679
Showing top 20 results:

  1. [Score: 10.582421] | PID: SHTFWBZFGK4QGMC8
     Title: men slim fit checkered casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd slim fit regular dusty brown navy shirt

  2. [Score: 10.529570] | PID: SHTFWBNYBPFMZ6FN
     Title: men slim fit checkered casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd slim fit regular teal navy blue shirt

  3. [Score: 10.529570] | PID: SHTFWBZ7NRMENHFX
     Title: men slim fit printed casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd slim fit cotton woven regular olive shirt

  4. [Score: 10.529570] | PID: SHTFWBNZFFRVQUUB
     Title: men slim fit checkered casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd slim fit cotton woven regular olive brown shirt

  5. [Score: 10.477277] | PID: SHTFWBZ8ARDZDVFF
     Title: men slim fit printed casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd slim fit cotton woven regular navy blue shirt

  6. [Score: 10.477277] | PID: SHTFSKF6SHHGJVVJ
     Title: men slim fit printed mandarin collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd printed solid twill slim fit teal shirt

  7. [Score: 10.477277] | PID: SHTFWBZ8ACHNGUHD
     Title: women slim fit checkered casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd slim fit cotton woven regular indigo white shirt

  8. [Score: 10.425533] | PID: SHTFWBZ73UXSPYTY
     Title: men slim fit checkered casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd slim fit cotton woven regular navy blue mustard shirt

  9. [Score: 10.425533] | PID: SHTFWBZ6TWZHBUQJ
     Title: women slim fit checkered casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd slim fit cotton woven regular navy blue tobacco shirt

 10. [Score: 10.374329] | PID: SHTFUPTTUZXPF9QF
     Title: men slim fit solid spread collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltdslim fit solid cotton woven brown shirt

 11. [Score: 10.374329] | PID: SHTFZEPTBEGGFJHB
     Title: women slim fit solid hood collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd solid cotton woven slim fit black shirt

 12. [Score: 10.374329] | PID: SHTFSKF6CFCYQJBG
     Title: women slim fit printed cut away collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd printed cotton woven slim fit olive shirt

 13. [Score: 10.374329] | PID: SHTFVY2N3DMNFX7J
     Title: women slim fit solid mandarin collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd slim fit woven melange maroon shirt

 14. [Score: 10.323656] | PID: SHTFVYFUQEVZYH4H
     Title: women slim fit solid mandarin collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd slim fit cotton woven grey black shirt

 15. [Score: 10.323656] | PID: SHTFUPTTFERGJBQU
     Title: men slim fit checkered spread collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltdslim fit printed cotton woven black white shirt

 16. [Score: 10.323656] | PID: SHTFUPTTYZFFBQCK
     Title: men slim fit solid spread collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltdslim fit solid cotton woven slate shirt

 17. [Score: 10.323656] | PID: SHTFWBZF5E44ABFZ
     Title: women slim fit solid spread collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd slim fit regular white shirt

 18. [Score: 10.323656] | PID: SHTFV5GAVPJZ7DFZ
     Title: women slim fit printed mandarin collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd printed cotton woven slim fit olive shirt

 19. [Score: 10.323656] | PID: SHTFSKF7HYPC8529
     Title: women slim fit checkered cut away collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd yd check cotton woven slim fit black blue shirt

 20. [Score: 10.323656] | PID: SHTFWBZ28VYZND8Z
     Title: women slim fit solid slim collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd slim fit regular beige shirt

... and 659 more results (not shown)
================================================================================


Query: ecko unl men shirt round neck
Query terms: ['ecko', 'unl', 'men', 'shirt', 'round', 'neck']
No documents found matching all query terms.
================================================================================

Query: women polo cotton
Query terms: ['women', 'polo', 'cotton']

Total results: 729
Showing top 20 results:

  1. [Score: 6.252321] | PID: TSHFPDP6XBUKAYTC
     Title: typography women round-neck grey t-shirt
     Brand: u s polo associati | Category: clothing and accessories | Sub-category: topwear     
     Description: u s polo assn captures the authenticity of polo and stays true to a classic american style updated to complement todays on-the-go lifestyle us polo womens tops t-shirts sycamore cotton s j gsm

  2. [Score: 6.202622] | PID: TSHFPDP64QDDAXDZ
     Title: typography women round-neck red t-shirt
     Brand: u s polo associati | Category: clothing and accessories | Sub-category: topwear     
     Description: u s polo assn captures the authenticity of polo and stays true to a classic american style updated to complement todays on-the-go lifestyle us polo womens tops t-shirts sycamore cotton s j gsm

  3. [Score: 6.142606] | PID: TKPFW45FHTCU9SUE
     Title: solid women grey track pants
     Brand: u s polo associati | Category: clothing and accessories | Sub-category: bottomwear  
     Description: track pant with contrast zipper pockets broad waistband with flat multi color draw cord comfort fit u s polo assn has sub-brands under its umbrella brand u s polo assn denim co u s polo assn tailored and u s polo assn active u s polo assn denim co includes our range of smart sunday brunch options for women like crisp shirts subtle yet smart t-shirts classic polo t-shirts jeans jackets and much more u s polo assn tailored caters to formal wear the distinction of u s polo assn tailored from its other sub-brands lies in its attention to classic details each article is specially crafted to be inviting authentic classic and genuine and is the perfect do-over for your - work wardrobe smart button-downs in classic hues and toned-down prints well-tailored trousers blazers waistcoats and jackets are just a few from the wide array of categories to choose from

  4. [Score: 6.084729] | PID: TSHFWCPHXDAHAHZZ
     Title: solid women polo-neck black t-shirt
     Brand: scott internation | Category: clothing and accessories | Sub-category: topwear      
     Description: refresh your wardrobe with the new range of summer collections of organic cotton polo t-shirts from scott international this polo t-shirts are made of is pure premium organic cotton fabric super combed solid polo and gives utmost comfort during all temperatures durable stitch and finish make this t-shirt a perfect formal and casual wear contrast tipping on the collar and sleeves makes this product have a unique look as compared to other products in the polo segment pair it with jeans or casual for a perfect casual or party look

  5. [Score: 6.084729] | PID: TSHFWCPHZSFTENKX
     Title: solid women polo-neck green t-shirt
     Brand: scott internation | Category: clothing and accessories | Sub-category: topwear      
     Description: refresh your wardrobe with the new range of summer collections of organic cotton polo t-shirts from scott international this polo t-shirts are made of is pure premium organic cotton fabric super combed solid polo and gives utmost comfort during all temperatures durable stitch and finish make this t-shirt a perfect formal and casual wear contrast tipping on the collar and sleeves makes this product have a unique look as compared to other products in the polo segment pair it with jeans or casual for a perfect casual or party look

  6. [Score: 6.084729] | PID: TSHFWCPHNECZFJ2E
     Title: solid women polo-neck yellow t-shirt
     Brand: scott internation | Category: clothing and accessories | Sub-category: topwear      
     Description: refresh your wardrobe with the new range of summer collections of organic cotton polo t-shirts from scott international this polo t-shirts are made of is pure premium organic cotton fabric super combed solid polo and gives utmost comfort during all temperatures durable stitch and finish make this t-shirt a perfect formal and casual wear contrast tipping on the collar and sleeves makes this product have a unique look as compared to other products in the polo segment pair it with jeans or casual for a perfect casual or party look

  7. [Score: 6.072970] | PID: TSHFWCPHCSTTHNZR
     Title: solid women polo-neck black t-shirt
     Brand: scott internation | Category: clothing and accessories | Sub-category: topwear      
     Description: refresh your wardrobe with the new range of summer collections of organic cotton polo t-shirts from scott international this polo t-shirts are made of is pure premium organic cotton fabric super combed solid polo and gives utmost comfort during all temperatures durable stitch and finish make this t-shirt a perfect formal and casual wear contrast tipping on the collar and sleeves makes this product have a unique look as compared to other products in the polo segment pair it with jeans or casual for a perfect casual or party look

  8. [Score: 6.056719] | PID: TKPFW4YR4NC5D3C3
     Title: solid women black track pants
     Brand: u s polo associati | Category: clothing and accessories | Sub-category: bottomwear  
     Description: track pant with contrast zipper pockets broad waistband with flat multi color draw cord comfort fit u s polo assn has sub-brands under its umbrella brand u s polo assn denim co u s polo assn tailored and u s polo assn active u s polo assn denim co includes our range of smart sunday brunch options for women like crisp shirts subtle yet smart t-shirts classic polo t-shirts jeans jackets and much more u s polo assn tailored caters to formal wear the distinction of u s polo assn tailored from its other sub-brands lies in its attention to classic details each article is specially crafted to be inviting authentic classic and genuine and is the perfect do-over for your - work wardrobe smart button-downs in classic hues and toned-down prints well-tailored trousers blazers waistcoats and jackets are just a few from the wide array of categories to choose from

  9. [Score: 5.948586] | PID: TSHFWJZEVGQPDYZH
     Title: solid women polo-neck grey t-shirt
     Brand: scott internation | Category: clothing and accessories | Sub-category: topwear      
     Description: refresh your wardrobe with the new range of summer collections of organic cotton polo t-shirts from scott international this polo t-shirts are made of is pure premium organic cotton fabric super combed solid polo and gives utmost comfort during all temperatures durable stitch and finish make this t-shirt a perfect formal and casual wear contrast tipping on the collar and sleeves makes this product have a unique look as compared to other products in the polo segment pair it with jeans or casual for a perfect casual or party look

 10. [Score: 5.948586] | PID: TSHFWJZEMZTBCSJN
     Title: solid women polo-neck white t-shirt
     Brand: scott internation | Category: clothing and accessories | Sub-category: topwear      
     Description: refresh your wardrobe with the new range of summer collections of organic cotton polo t-shirts from scott international this polo t-shirts are made of is pure premium organic cotton fabric super combed solid polo and gives utmost comfort during all temperatures durable stitch and finish make this t-shirt a perfect formal and casual wear contrast tipping on the collar and sleeves makes this product have a unique look as compared to other products in the polo segment pair it with jeans or casual for a perfect casual or party look

 11. [Score: 5.935387] | PID: TSHFWJZEGU7QFZMA
     Title: solid women polo-neck yellow t-shirt
     Brand: scott internation | Category: clothing and accessories | Sub-category: topwear      
     Description: refresh your wardrobe with the new range of summer collections of organic cotton polo t-shirts from scott international this polo t-shirts are made of is pure premium organic cotton fabric super combed solid polo and gives utmost comfort during all temperatures durable stitch and finish make this t-shirt a perfect formal and casual wear contrast tipping on the collar and sleeves makes this product have a unique look as compared to other products in the polo segment pair it with jeans or casual for a perfect casual or party look

 12. [Score: 5.922249] | PID: TSHFWCPHZSWGYXGA
     Title: solid women polo-neck grey t-shirt
     Brand: scott internation | Category: clothing and accessories | Sub-category: topwear      
     Description: refresh your wardrobe with the new range of summer collections of organic cotton polo t-shirts from scott international this polo t-shirts are made of is pure premium organic cotton fabric super combed solid polo and gives utmost comfort during all temperatures durable stitch and finish make this t-shirt a perfect formal and casual wear contrast tipping on the collar and sleeves makes this product have a unique look as compared to other products in the polo segment pair it with jeans or casual for a perfect casual or party look

 13. [Score: 5.922249] | PID: TSHFWCPHNFFZAWHB
     Title: solid women polo-neck grey t-shirt
     Brand: scott internation | Category: clothing and accessories | Sub-category: topwear      
     Description: refresh your wardrobe with the new range of summer collections of organic cotton polo t-shirts from scott international this polo t-shirts are made of is pure premium organic cotton fabric super combed solid polo and gives utmost comfort during all temperatures durable stitch and finish make this t-shirt a perfect formal and casual wear contrast tipping on the collar and sleeves makes this product have a unique look as compared to other products in the polo segment pair it with jeans or casual for a perfect casual or party look

 14. [Score: 5.922249] | PID: TSHFWJZESHNWDPNZ
     Title: solid women polo-neck blue t-shirt
     Brand: scott internation | Category: clothing and accessories | Sub-category: topwear      
     Description: refresh your wardrobe with the new range of summer collections of organic cotton polo t-shirts from scott international this polo t-shirts are made of is pure premium organic cotton fabric super combed solid polo and gives utmost comfort during all temperatures durable stitch and finish make this t-shirt a perfect formal and casual wear contrast tipping on the collar and sleeves makes this product have a unique look as compared to other products in the polo segment pair it with jeans or casual for a perfect casual or party look

 15. [Score: 5.922249] | PID: TSHFWCPH6ZH6X8EK
     Title: solid women polo-neck brown t-shirt
     Brand: scott internation | Category: clothing and accessories | Sub-category: topwear      
     Description: refresh your wardrobe with the new range of summer collections of organic cotton polo t-shirts from scott international this polo t-shirts are made of is pure premium organic cotton fabric super combed solid polo and gives utmost comfort during all temperatures durable stitch and finish make this t-shirt a perfect formal and casual wear contrast tipping on the collar and sleeves makes this product have a unique look as compared to other products in the polo segment pair it with jeans or casual for a perfect casual or party look

 16. [Score: 5.909174] | PID: TSHFWJZEZYHJFJQU
     Title: solid women polo-neck light green t-shirt
     Brand: scott internation | Category: clothing and accessories | Sub-category: topwear      
     Description: refresh your wardrobe with the new range of summer collections of organic cotton polo t-shirts from scott international this polo t-shirts are made of is pure premium organic cotton fabric super combed solid polo and gives utmost comfort during all temperatures durable stitch and finish make this t-shirt a perfect formal and casual wear contrast tipping on the collar and sleeves makes this product have a unique look as compared to other products in the polo segment pair it with jeans or casual for a perfect casual or party look

 17. [Score: 5.879698] | PID: TOPF2ZBYTV6FAZ6S
     Title: solid women polo-neck dark blue t-shirt
     Brand: flexim | Category: clothing and accessories | Sub-category: topwear
     Description: fleximaa women s cotton half sleeve plain solid polo t-shirt made from cotton pls buy after confirming your size measurements refer size chart wear this t-shirt with a pair of blue or black jeans our brand fleximaa a product of flexible apparels

 18. [Score: 5.794313] | PID: TSHESGM3KJNVAV2J
     Title: solid women polo-neck maroon t-shirt
     Brand: t spor | Category: clothing and accessories | Sub-category: topwear
     Description: double mercerised polo the world finest quality double mercerized cotton polo shirt brought to you by t sportsthey are premium quality polo shirt crafted mercerized cotton for silky this piece is mercerized processes gives you better resistance to multiple washing and it keeps colors bright over a long durable knitted striped polo t-shirt has smooth hand a ribbed polo collar short button placket on the front brand logo embroidery on the left side of chest packet it gives you a fresh look and always you feel young and smart style it pair of jeans or chinos and casual shoes for a complete look feel classical short sleeve and neat top stitched edges cotton cotton

 19. [Score: 5.727363] | PID: TKPFW44SHG2VYJBT
     Title: solid women red track pants
     Brand: u s polo associati | Category: clothing and accessories | Sub-category: bottomwear  
     Description: track pant with contrast zipper pockets broad waistband with flat multi color draw cord comfort fit u s polo assn has sub-brands under its umbrella brand u s polo assn denim co u s polo assn tailored and u s polo assn active u s polo assn denim co includes our range of smart sunday brunch options for women like crisp shirts subtle yet smart t-shirts classic polo t-shirts jeans jackets and much more u s polo assn tailored caters to formal wear the distinction of u s polo assn tailored from its other sub-brands lies in its attention to classic details each article is specially crafted to be inviting authentic classic and genuine and is the perfect do-over for your - work wardrobe smart button-downs in classic hues and toned-down prints well-tailored trousers blazers waistcoats and jackets are just a few from the wide array of categories to choose from

 20. [Score: 5.719642] | PID: TSHFWVYDSKZH96N9
     Title: solid women polo-neck grey t-shirt
     Brand: onei | Category: clothing and accessories | Sub-category: topwear
     Description: refresh your clothing with the awesome collection of basic polo t-shirts from oneiro this t-shirt is made of cotton fabric and gives utmost comfort during all temperatures elegant stitch and solid colours makes this t-shirt a perfect formal and casual wear pair it with jeans or casual for a perfect casual look

... and 709 more results (not shown)
================================================================================


Query: casual clothes slim fit
Query terms: ['casual', 'clothes', 'slim', 'fit']
No documents found matching all query terms.
================================================================================

Query: biowash innerwear
Query terms: ['biowash', 'innerwear']

Total results: 59
Showing top 20 results:

  1. [Score: 7.122851] | PID: VESFX3ZFMVZDRS4E
     Title: free authority men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: look trendy and feel comfortable with this character printed sleevless vest featuring friends crafted out of cotton which is biowashed for smooth feel and befriend to skin this featuring can be worn for any occasion a casual day at work or for a fun filled weekend or leisure wear

  2. [Score: 7.122851] | PID: VESFX3ZFJEKDYZPW
     Title: free authority women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: look trendy and feel comfortable with this character printed sleevless vest featuring the simpsons crafted out of cotton which is biowashed for smooth feel and befriend to skin this featuring can be worn for any occasion a casual day at work or for a fun filled weekend or leisure wear

  3. [Score: 7.122851] | PID: VESFX3ZFC7XXJ5CR
     Title: free authority women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: look trendy and feel comfortable with this character printed sleevless vest featuring nasa crafted out of cotton which is biowashed for smooth feel and befriend to skin this featuring can be worn for any occasion a casual day at work or for a fun filled weekend or leisure wear

  4. [Score: 7.075515] | PID: VESFX3ZFXTNQXDFN
     Title: free authority men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: look trendy and feel comfortable with this character printed sleevless vest featuring scooby doo crafted out of cotton which is biowashed for smooth feel and befriend to skin this featuring can be worn for any occasion a casual day at work or for a fun filled weekend or leisure wear

  5. [Score: 7.075515] | PID: VESFX3ZFZFUJBJVV
     Title: free authority women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: look trendy and feel comfortable with this character printed sleevless vest featuring dragon ball z crafted out of cotton which is biowashed for smooth feel and befriend to skin this featuring can be worn for any occasion a casual day at work or for a fun filled weekend or leisure wear

  6. [Score: 5.965475] | PID: BXRFKYKYF3SGUMFH
     Title: printed men boxer pack of
     Brand: sayitlo | Category: clothing and accessories | Sub-category: innerwear and swimwear 
     Description: jump into the comfort zone with the most trendy looking boxers a special elastic which leaves no mark and sweat absorbing

  7. [Score: 5.679094] | PID: VESFGYJDFBUZG6FQ
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest

  8. [Score: 5.679094] | PID: VESFGYJDWUH364V2
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest

  9. [Score: 4.941122] | PID: VESFHJHQPTY2ZC8E
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 10. [Score: 4.941122] | PID: VESFHJHQREFDNKRQ
     Title: sayitloud men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 11. [Score: 4.918296] | PID: VESFHJHPGC9CW6XA
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 12. [Score: 4.918296] | PID: VESFHJHPHFHSSUJZ
     Title: sayitloud men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 13. [Score: 4.918296] | PID: VESFHJHP4AYNWSAZ
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 14. [Score: 4.873273] | PID: VESFHJHPZUSZP2H2
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 15. [Score: 4.873273] | PID: VESFHJHPZJUBPS49
     Title: sayitloud men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 16. [Score: 4.873273] | PID: VESFHJHPBCYZGZHE
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 17. [Score: 4.873273] | PID: VESFHJHPGM4MJADF
     Title: sayitloud men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 18. [Score: 4.873273] | PID: VESFHJHQF7YJ3DHH
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 19. [Score: 4.873273] | PID: VESFHJHPCZH2ACVK
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 20. [Score: 4.873273] | PID: VESFHJHQ2VPZV4F7
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

... and 39 more results (not shown)
================================================================================

**Custom Score Results:**
Query: ecko unl shirt
Query terms: ['ecko', 'unl', 'shirt']

Total results: 679
Showing top 20 results:

  1. [Score: 4.999581] | PID: SHTFWBZB2BYWYNEE
     Title: men slim fit printed casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Rating: 4.0 | Discount: 32.0 | Out of stock: True
     Description: N/A

  2. [Score: 4.979052] | PID: SHTFWBZ8ARDZDVFF
     Title: men slim fit printed casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Rating: 2.5 | Discount: 27.0 | Out of stock: False
     Description: ecko unltd slim fit cotton woven regular navy blue shirt

  3. [Score: 4.829735] | PID: SHTFWBZ7NRMENHFX
     Title: men slim fit printed casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Rating: 2.5 | Discount: 27.0 | Out of stock: False
     Description: ecko unltd slim fit cotton woven regular olive shirt

  4. [Score: 4.828152] | PID: SHTFSKF6AKNYH3YR
     Title: men slim fit solid casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Rating: 3.3 | Discount: 50.0 | Out of stock: False
     Description: N/A

  5. [Score: 4.824182] | PID: SHTFXV5FFATYDWZT
     Title: men slim fit solid casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Rating: 2.5 | Discount: 32.0 | Out of stock: False
     Description: N/A

  6. [Score: 4.802231] | PID: SHTFSKF6ARDQ7RYU
     Title: men slim fit solid casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Rating: 4.1 | Discount: 25.0 | Out of stock: True
     Description: N/A

  7. [Score: 4.797969] | PID: SHTFUPTXS69YVUNX
     Title: women slim fit solid casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Rating: 2.5 | Discount: 35.0 | Out of stock: False
     Description: N/A

  8. [Score: 4.734655] | PID: SHTFXV5EC6JYYVZA
     Title: women slim fit printed casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Rating: 2.5 | Discount: 32.0 | Out of stock: False
     Description: N/A

  9. [Score: 4.724316] | PID: SHTFWBZ9XSCJP5BA
     Title: women slim fit printed casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Rating: 2.5 | Discount: 32.0 | Out of stock: False
     Description: N/A

 10. [Score: 4.718602] | PID: SHTFXV5EQ3AYZEB8
     Title: men slim fit solid casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Rating: 2.5 | Discount: 32.0 | Out of stock: False
     Description: N/A

 11. [Score: 4.717624] | PID: SHTFXV5EFHMGTGFG
     Title: women slim fit solid casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Rating: 2.5 | Discount: 32.0 | Out of stock: False
     Description: N/A

 12. [Score: 4.708543] | PID: SHTFQ252ZKDSWEZT
     Title: women slim fit solid casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Rating: 3.7 | Discount: 30.0 | Out of stock: True
     Description: N/A

 13. [Score: 4.699011] | PID: SHTFXV5EVCR468PS
     Title: women slim fit printed casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Rating: 2.5 | Discount: 32.0 | Out of stock: False
     Description: N/A

 14. [Score: 4.695172] | PID: SHTFUPTU3GGDCGTE
     Title: women slim fit solid casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Rating: 3.8 | Discount: 35.0 | Out of stock: False
     Description: N/A

 15. [Score: 4.687603] | PID: SHTFXV5E6PYFPPGH
     Title: men slim fit solid casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Rating: 2.5 | Discount: 33.0 | Out of stock: False
     Description: N/A

 16. [Score: 4.677632] | PID: SHTFXV5ENY5FYHPH
     Title: men slim fit solid casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Rating: 2.5 | Discount: 9.0 | Out of stock: False
     Description: N/A

 17. [Score: 4.677632] | PID: SHTFXV5F7ZBGAHAH
     Title: men slim fit solid casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Rating: 2.5 | Discount: 37.0 | Out of stock: False
     Description: N/A

 18. [Score: 4.666762] | PID: SHTFSKF6C5ZA8KKZ
     Title: men slim fit printed casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Rating: 4.3 | Discount: 33.0 | Out of stock: False
     Description: N/A

 19. [Score: 4.655654] | PID: SHTFXV5EMMPNBFGF
     Title: men slim fit printed casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Rating: 2.5 | Discount: 36.0 | Out of stock: False
     Description: N/A

 20. [Score: 4.655410] | PID: SHTFRR7FHGVAZHTB
     Title: women slim fit printed casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Rating: None | Discount: 30.0 | Out of stock: True
     Description: N/A

... and 659 more results (not shown)
================================================================================


Query: ecko unl men shirt round neck
Query terms: ['ecko', 'unl', 'men', 'shirt', 'round', 'neck']
No documents found matching all query terms.
================================================================================

Query: women polo cotton
Query terms: ['women', 'polo', 'cotton']

Total results: 729
Showing top 20 results:

  1. [Score: 2.559681] | PID: TSHFXVJZYXJNUZH6
     Title: solid women polo-neck white t-shirt
     Brand: u s polo ass | Category: clothing and accessories | Sub-category: topwear
     Rating: 4.2 | Discount: 41.0 | Out of stock: True
     Description: half sleeve polo s

  2. [Score: 2.510132] | PID: TSHFXVJZZPMGZ5QE
     Title: solid women polo-neck red t-shirt
     Brand: u s polo ass | Category: clothing and accessories | Sub-category: topwear
     Rating: 4.5 | Discount: 42.0 | Out of stock: True
     Description: half sleeve polo l

  3. [Score: 2.388284] | PID: TSHFPDP6XBUKAYTC
     Title: typography women round-neck grey t-shirt
     Brand: u s polo associati | Category: clothing and accessories | Sub-category: topwear     
     Rating: 4.5 | Discount: 8.0 | Out of stock: False
     Description: u s polo assn captures the authenticity of polo and stays true to a classic american style updated to complement todays on-the-go lifestyle us polo womens tops t-shirts sycamore cotton s j gsm

  4. [Score: 2.359952] | PID: TSHFPDP64QDDAXDZ
     Title: typography women round-neck red t-shirt
     Brand: u s polo associati | Category: clothing and accessories | Sub-category: topwear     
     Rating: 4.5 | Discount: 8.0 | Out of stock: False
     Description: u s polo assn captures the authenticity of polo and stays true to a classic american style updated to complement todays on-the-go lifestyle us polo womens tops t-shirts sycamore cotton s j gsm

  5. [Score: 2.331896] | PID: TSHFGNKBMZQFFXCG
     Title: self design women polo-neck dark blue t-shirt
     Brand: u s polo ass | Category: clothing and accessories | Sub-category: topwear
     Rating: 4.0 | Discount: 43.0 | Out of stock: False
     Description: half sleeve polo t-shirt

  6. [Score: 2.293297] | PID: TSHFJFVBKFMMDPHC
     Title: striped women polo-neck white blue orange t-shirt
     Brand: u s polo ass | Category: clothing and accessories | Sub-category: topwear
     Rating: 4.1 | Discount: 36.0 | Out of stock: False
     Description: half sleeve polo t-shirt

  7. [Score: 2.286240] | PID: TSHFHF38WGUUTKGU
     Title: striped women polo-neck multicolor t-shirt
     Brand: u s polo ass | Category: clothing and accessories | Sub-category: topwear
     Rating: 4.2 | Discount: 45.0 | Out of stock: False
     Description: half sleeve polo t-shirt

  8. [Score: 2.150012] | PID: TOPF2ZBYTV6FAZ6S
     Title: solid women polo-neck dark blue t-shirt
     Brand: flexim | Category: clothing and accessories | Sub-category: topwear
     Rating: 4.0 | Discount: 46.0 | Out of stock: False
     Description: fleximaa women s cotton half sleeve plain solid polo t-shirt made from cotton pls buy after confirming your size measurements refer size chart wear this t-shirt with a pair of blue or black jeans our brand fleximaa a product of flexible apparels

  9. [Score: 2.141197] | PID: TSHFGQKHTYSS8YWE
     Title: solid women polo-neck dark blue t-shirt
     Brand: pu | Category: clothing and accessories | Sub-category: topwear
     Rating: 4.0 | Discount: 65.0 | Out of stock: True
     Description: womens graphic polo ii peacoat

 10. [Score: 2.098797] | PID: TSHFHF37GX8HHGSH
     Title: printed women polo-neck yellow t-shirt
     Brand: u s polo ass | Category: clothing and accessories | Sub-category: topwear
     Rating: 3.9 | Discount: 45.0 | Out of stock: False
     Description: half sleeve polo t-shirt

 11. [Score: 2.068393] | PID: TSHFWVYDSKZH96N9
     Title: solid women polo-neck grey t-shirt
     Brand: onei | Category: clothing and accessories | Sub-category: topwear
     Rating: 4.5 | Discount: 62.0 | Out of stock: False
     Description: refresh your clothing with the awesome collection of basic polo t-shirts from oneiro this t-shirt is made of cotton fabric and gives utmost comfort during all temperatures elegant stitch and solid colours makes this t-shirt a perfect formal and casual wear pair it with jeans or casual for a perfect casual look

 12. [Score: 2.056865] | PID: TSHFZ3JE9KCHWVYW
     Title: solid women polo-neck white t-shirt
     Brand: reeb | Category: clothing and accessories | Sub-category: topwear
     Rating: 4.6 | Discount: 33.0 | Out of stock: False
     Description: foundation cot polo

 13. [Score: 2.018726] | PID: TSHFB4C7PVTBNGE4
     Title: solid women polo-neck blue t-shirt
     Brand: pu | Category: clothing and accessories | Sub-category: topwear
     Rating: 3.8 | Discount: 65.0 | Out of stock: False
     Description: ess jersey polo

 14. [Score: 2.007111] | PID: TSHFWVYDSFCFG8GV
     Title: solid women polo-neck yellow t-shirt
     Brand: onei | Category: clothing and accessories | Sub-category: topwear
     Rating: 4.5 | Discount: 62.0 | Out of stock: False
     Description: refresh your clothing with the awesome collection of basic polo t-shirts from oneiro this t-shirt is made of cotton fabric and gives utmost comfort during all temperatures elegant stitch and solid colours makes this t-shirt a perfect formal and casual wear pair it with jeans or casual for a perfect casual look

 15. [Score: 2.006251] | PID: TSHEMA3YP8FYVBUF
     Title: solid women polo-neck beige t-shirt
     Brand: t spor | Category: clothing and accessories | Sub-category: topwear
     Rating: 2.3 | Discount: 52.0 | Out of stock: False
     Description: t sports beige half polo t-shirt for women

 16. [Score: 1.981224] | PID: TROFUHKCU8CSZY4H
     Title: slim fit women blue pure cotton trousers
     Brand: u s polo as | Category: clothing and accessories | Sub-category: bottomwear
     Rating: 4.1 | Discount: 42.0 | Out of stock: False
     Description: uspa womens trs

 17. [Score: 1.970174] | PID: TSHFVEV23EZJZREQ
     Title: solid women polo-neck black t-shirt
     Brand: reeb | Category: clothing and accessories | Sub-category: topwear
     Rating: 4.5 | Discount: 31.0 | Out of stock: False
     Description: a casual cotton pique polo designed to provide the style you need single dye effect elevates the look appeal to keep you at par in lifestyle quotient regular fit

 18. [Score: 1.968099] | PID: TROFUHK7P6ET2C8U
     Title: slim fit women black pure cotton trousers
     Brand: u s polo as | Category: clothing and accessories | Sub-category: bottomwear
     Rating: 3.9 | Discount: 42.0 | Out of stock: False
     Description: uspa womens trs

 19. [Score: 1.954880] | PID: TSHFB8TKW5AUNY9W
     Title: color block women polo-neck white black t-shirt
     Brand: flexim | Category: clothing and accessories | Sub-category: topwear
     Rating: 4.0 | Discount: 53.0 | Out of stock: False
     Description: women s polo collar t-shirt made from cotton and is an essential every day item for every wardrobe

 20. [Score: 1.954880] | PID: TSHFB8TKKBYQKHBG
     Title: color block women polo-neck white black t-shirt
     Brand: flexim | Category: clothing and accessories | Sub-category: topwear
     Rating: 4.0 | Discount: 53.0 | Out of stock: False
     Description: women s polo collar t-shirt made from cotton and is an essential every day item for every wardrobe

... and 709 more results (not shown)
================================================================================


Query: casual clothes slim fit
Query terms: ['casual', 'clothes', 'slim', 'fit']
No documents found matching all query terms.
================================================================================

Query: biowash innerwear
Query terms: ['biowash', 'innerwear']

Total results: 59
Showing top 20 results:

  1. [Score: 1.812391] | PID: VESFX3ZFMVZDRS4E
     Title: free authority men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Rating: 3.9 | Discount: 15.0 | Out of stock: False
     Description: look trendy and feel comfortable with this character printed sleevless vest featuring friends crafted out of cotton which is biowashed for smooth feel and befriend to skin this featuring can be worn for any occasion a casual day at work or for a fun filled weekend or leisure wear

  2. [Score: 1.766239] | PID: VESFX3ZFC7XXJ5CR
     Title: free authority women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Rating: 3.9 | Discount: 15.0 | Out of stock: False
     Description: look trendy and feel comfortable with this character printed sleevless vest featuring nasa crafted out of cotton which is biowashed for smooth feel and befriend to skin this featuring can be worn for any occasion a casual day at work or for a fun filled weekend or leisure wear

  3. [Score: 1.746961] | PID: VESFX3ZFJEKDYZPW
     Title: free authority women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Rating: 3.9 | Discount: 15.0 | Out of stock: False
     Description: look trendy and feel comfortable with this character printed sleevless vest featuring the simpsons crafted out of cotton which is biowashed for smooth feel and befriend to skin this featuring can be worn for any occasion a casual day at work or for a fun filled weekend or leisure wear

  4. [Score: 1.720002] | PID: VESFX3ZFZFUJBJVV
     Title: free authority women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Rating: 3.9 | Discount: 15.0 | Out of stock: False
     Description: look trendy and feel comfortable with this character printed sleevless vest featuring dragon ball z crafted out of cotton which is biowashed for smooth feel and befriend to skin this featuring can be worn for any occasion a casual day at work or for a fun filled weekend or leisure wear

  5. [Score: 1.606985] | PID: VESFX3ZFXTNQXDFN
     Title: free authority men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Rating: 3.9 | Discount: 15.0 | Out of stock: False
     Description: look trendy and feel comfortable with this character printed sleevless vest featuring scooby doo crafted out of cotton which is biowashed for smooth feel and befriend to skin this featuring can be worn for any occasion a casual day at work or for a fun filled weekend or leisure wear

  6. [Score: 1.302028] | PID: BXRFKYKYF3SGUMFH
     Title: printed men boxer pack of
     Brand: sayitlo | Category: clothing and accessories | Sub-category: innerwear and swimwear 
     Rating: 5.0 | Discount: 78.0 | Out of stock: False
     Description: jump into the comfort zone with the most trendy looking boxers a special elastic which leaves no mark and sweat absorbing

  7. [Score: 1.233254] | PID: VESFGYJDFBUZG6FQ
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Rating: 3.8 | Discount: 62.0 | Out of stock: False
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest

  8. [Score: 1.227095] | PID: VESFGYJDWUH364V2
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Rating: 3.8 | Discount: 66.0 | Out of stock: False
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest

  9. [Score: 1.021441] | PID: VESFHJHPZUSZP2H2
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Rating: 4.9 | Discount: 62.0 | Out of stock: False
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 10. [Score: 1.012512] | PID: VESFHJHPZJUBPS49
     Title: sayitloud men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Rating: 4.7 | Discount: 62.0 | Out of stock: False
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 11. [Score: 0.986073] | PID: VESFHJHPBCYZGZHE
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Rating: 4.0 | Discount: 62.0 | Out of stock: False
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 12. [Score: 0.983613] | PID: VESFHJHPGM4MJADF
     Title: sayitloud men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Rating: 4.0 | Discount: 62.0 | Out of stock: False
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 13. [Score: 0.969833] | PID: VESFHJHPHFHSSUJZ
     Title: sayitloud men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Rating: 3.7 | Discount: 62.0 | Out of stock: False
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 14. [Score: 0.967442] | PID: VESFHJHQREFDNKRQ
     Title: sayitloud men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Rating: 3.6 | Discount: 62.0 | Out of stock: False
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 15. [Score: 0.967272] | PID: VESFHJHQPTY2ZC8E
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Rating: 3.6 | Discount: 62.0 | Out of stock: False
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 16. [Score: 0.964426] | PID: VESFHJHQF7YJ3DHH
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Rating: 3.6 | Discount: 69.0 | Out of stock: False
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 17. [Score: 0.963624] | PID: VESFHJHQ2VPZV4F7
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Rating: 3.6 | Discount: 62.0 | Out of stock: False
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 18. [Score: 0.955188] | PID: VESFHJHPGC9CW6XA
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Rating: 3.2 | Discount: 62.0 | Out of stock: False
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 19. [Score: 0.949897] | PID: VESFHJHP4AYNWSAZ
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Rating: 3.0 | Discount: 62.0 | Out of stock: False
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

 20. [Score: 0.940923] | PID: VESFHJHPCZH2ACVK
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Rating: 3.0 | Discount: 62.0 | Out of stock: False
     Description: stay relaxed during your rigorous training sessions wearing this cotton vest combed cotton provides ideal comfort for your skin while a round-neck proves ideal for t-shirts and shirts wear this sleeveless vest from say it loud under your top layer to give your torso a fine finish and strong hold on our muscles it assures excellent freedom of movement featuring slim fit this vest will be a great addition to your wardrobe

... and 39 more results (not shown)
================================================================================

**Word2Vec + Cosine Results:**
Query: ecko unl shirt
Query terms: ['ecko', 'unl', 'shirt']

Total results: 679
Showing top 20 results:

  1. [Score: 0.482317] | PID: SHTFSKF6SHHGJVVJ
     Title: men slim fit printed mandarin collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd printed solid twill slim fit teal shirt

  2. [Score: 0.476759] | PID: SHTFSKF7GVD3HESV
     Title: men slim fit checkered cut away collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd yd geometric cotton woven slim fit beige navy blue shirt

  3. [Score: 0.472241] | PID: SHTFV5HP4CUAT7YP
     Title: men slim fit checkered cut away collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd yd check cotton woven slim fit maroon shirt

  4. [Score: 0.469760] | PID: SHTFWBNYBPFMZ6FN
     Title: men slim fit checkered casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd slim fit regular teal navy blue shirt

  5. [Score: 0.469700] | PID: SHTFV5HV74ZETGF9
     Title: women slim fit houndstooth spread collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd printed cotton woven slim fit navy blue blue shirt

  6. [Score: 0.469321] | PID: SHTFVY2GV5GJZSWJ
     Title: men slim fit checkered cut away collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd slim fit cotton woven rust maroon shirt

  7. [Score: 0.468982] | PID: SHTFV5GSZKWGGZUR
     Title: women slim fit checkered spread collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd yd check cotton woven slim fit indigo red shirt

  8. [Score: 0.467973] | PID: SHTFWBZFGK4QGMC8
     Title: men slim fit checkered casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd slim fit regular dusty brown navy shirt

  9. [Score: 0.467417] | PID: SHTFV5GYVQ8RGUG4
     Title: women slim fit checkered cut away collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd yd check cotton woven slim fit beige khaki navy shirt

 10. [Score: 0.465727] | PID: SHTFV5HZPDAJZHZB
     Title: men slim fit checkered cut away collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd yd check cotton woven slim fit navy blue green shirt

 11. [Score: 0.462739] | PID: SHTFVY2CGYM3SVGU
     Title: men slim fit checkered cut away collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd slim fit cotton woven blue navy blue shirt

 12. [Score: 0.462069] | PID: SHTFV5HSRMHNZMHP
     Title: men slim fit checkered spread collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd yd check cotton woven slim fit indigo yellow shirt

 13. [Score: 0.461958] | PID: SHTFV5GYQSYGVGZP
     Title: women slim fit checkered cut away collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd yd check cotton woven slim fit green lt green shirt

 14. [Score: 0.461824] | PID: SHTFWBZ73UXSPYTY
     Title: men slim fit checkered casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd slim fit cotton woven regular navy blue mustard shirt

 15. [Score: 0.461541] | PID: SHTFV5HUHKHPGXKJ
     Title: men slim fit checkered cut away collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd yd check cotton woven slim fit navy blue yellow shirt

 16. [Score: 0.461177] | PID: SHTFV5GFGT5DMHQZ
     Title: women slim fit checkered cut away collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd yd check cotton woven slim fit black maroon shirt

 17. [Score: 0.460830] | PID: SHTFV5HTEBSVEEHE
     Title: women slim fit checkered hood collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd yd check cotton woven slim fit navy blue red shirt

 18. [Score: 0.460799] | PID: SHTFVY2GY2GQBSEB
     Title: women slim fit checkered spread collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd slim fit cotton woven maroon navy blue shirt

 19. [Score: 0.460459] | PID: SHTFV5GHAZWWGCQH
     Title: men slim fit checkered cut away collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd yd check cotton woven slim fit khaki navy blue shirt

 20. [Score: 0.460372] | PID: SHTFV5JKYERFJFXD
     Title: men slim fit checkered spread collar casual shirt
     Brand: ecko unl | Category: clothing and accessories | Sub-category: topwear
     Description: ecko unltd yd check cotton woven slim fit olive indigo shirt

... and 659 more results (not shown)
================================================================================


Query: ecko unl men shirt round neck
Query terms: ['ecko', 'unl', 'men', 'shirt', 'round', 'neck']
No documents found matching all query terms.
================================================================================

Query: women polo cotton
Query terms: ['women', 'polo', 'cotton']

Total results: 729
Showing top 20 results:

  1. [Score: 0.774748] | PID: TSHFPDP6XBUKAYTC
     Title: typography women round-neck grey t-shirt
     Brand: u s polo associati | Category: clothing and accessories | Sub-category: topwear     
     Description: u s polo assn captures the authenticity of polo and stays true to a classic american style updated to complement todays on-the-go lifestyle us polo womens tops t-shirts sycamore cotton s j gsm

  2. [Score: 0.769322] | PID: TSHFPDP64QDDAXDZ
     Title: typography women round-neck red t-shirt
     Brand: u s polo associati | Category: clothing and accessories | Sub-category: topwear     
     Description: u s polo assn captures the authenticity of polo and stays true to a classic american style updated to complement todays on-the-go lifestyle us polo womens tops t-shirts sycamore cotton s j gsm

  3. [Score: 0.769011] | PID: TOPF2ZBYTV6FAZ6S
     Title: solid women polo-neck dark blue t-shirt
     Brand: flexim | Category: clothing and accessories | Sub-category: topwear
     Description: fleximaa women s cotton half sleeve plain solid polo t-shirt made from cotton pls buy after confirming your size measurements refer size chart wear this t-shirt with a pair of blue or black jeans our brand fleximaa a product of flexible apparels

  4. [Score: 0.761561] | PID: TROFUHK7P6ET2C8U
     Title: slim fit women black pure cotton trousers
     Brand: u s polo as | Category: clothing and accessories | Sub-category: bottomwear
     Description: uspa womens trs

  5. [Score: 0.760651] | PID: TROFUHKHWSSGEWAF
     Title: slim fit women khaki pure cotton trousers
     Brand: u s polo as | Category: clothing and accessories | Sub-category: bottomwear
     Description: uspa womens trs

  6. [Score: 0.759090] | PID: TROFUHK6AGZWZKQE
     Title: slim fit women blue pure cotton trousers
     Brand: u s polo ass | Category: clothing and accessories | Sub-category: bottomwear        
     Description: uspa womens trs

  7. [Score: 0.758119] | PID: TROFUHKCU8CSZY4H
     Title: slim fit women blue pure cotton trousers
     Brand: u s polo as | Category: clothing and accessories | Sub-category: bottomwear
     Description: uspa womens trs

  8. [Score: 0.752447] | PID: TSHFMAHCJZYFYY6K
     Title: sporty women polo-neck black t-shirt
     Brand: tee bud | Category: clothing and accessories | Sub-category: topwear
     Description: refresh your clothing with the awesome collection of round-neck tees from t-shirt express these t shirts are made of bio wash cotton and make a comfort wear for all seasons the feel of the fabric keeps you comfortable even at high humid conditions pair it with cotton trousers or denims for a perfect weekend wear

  9. [Score: 0.751301] | PID: PYJFZCT4GKTAEMBZ
     Title: women pyjama pack of
     Brand: u s polo associati | Category: clothing and accessories | Sub-category: sleepwear   
     Description: uspa lounge pants cotton fabric with extra softness check lounge pants in regular fit side pockets twill fabric in checks

 10. [Score: 0.751301] | PID: PYJFX3H3K7KE2YKV
     Title: women pyjama pack of
     Brand: u s polo associati | Category: clothing and accessories | Sub-category: sleepwear   
     Description: uspa lounge pants cotton fabric with extra softness check lounge pants in regular fit side pockets twill fabric in checks

 11. [Score: 0.747224] | PID: TSHFUWQAAECQFGJB
     Title: solid women v-neck maroon t-shirt
     Brand: eyetwist | Category: clothing and accessories | Sub-category: topwear
     Description: pc cotton bio-washed full sleeves t-shirts high quality s bio-washed cotton fabric gsm soft and wrinkle free fabric cotton bio washed maroon black waist coat t-shirts high quality smooth fabric and quality product and unique style t-shirt available in all sizes from s-xxl t-shirts combo for women t-shirts for women stylish latest t-shirts for women search terms blended designer v-neck o-neck round-neck zipper hooded hoodie polo collar henley button bio-washed t-shirt dress clothes clothing apparel t-shirts combo for women t-shirts for women stylish latest t-shirts for women maroon black t-shirt women hooded maroon black t-shirt women maroon black striped round-neck t-shirt solid women hooded black maroon t-shirt fresh trend maroon black white cotton round-neck t-shirt for women fresh trend maroon black white t-shirt for woman fresh trend maroon white black t shirts for women maroon black green cotton round-neck t-shirt for women maroon black whit black dark blue trip color black women v-neck maroon t-shirt women women stylish latest t-shirts maroon white waist coat combo pack collar girls fashion ladies kids black daily use western wear duke classic polo round v-neck seven sea button cotton polyster zipper bio washed navy red grey full sleeves seven sea cotton bio washed maroon black v-neck regular fit stylish t-shirts high quality smooth fabric and quality product and unique style t-shirt available in all sizes fro s-xxl t-shirts combo for women t-shirts for women stylish latest t-shirts for women search terms blended designer v-neck o-neck round-neck zipper hooded hoodie polo collar henley button bio-washed t-shirt dress clothes clothing apparel t-shirts combo for women t-shirts for women stylish latest t-shirts girls stylish latest t-shirts combo girls fashion ladies and kids tees black and white daily use t-shirts duke western wear womens wear women stylish latest t-shirts maroon white waist coat combo pack collar girls fashion ladies kids black daily use western wear duke classic polo round v-neck seven sea button cotton polyster zipper bio washed navy red grey full sleeves eyetwister cotton bio washed maroon black v-neck regular fit stylish t-shirts high quality smooth fabric and quality product and unique style t-shirt available in all sizes fro s-xxl t-shirts combo for women t-shirts for women stylish latest t-shirts for women search terms blended designer v-neck o-neck round-neck zipper hooded hoodie polo collar henley button bio-washed t-shirt dress clothes clothing apparel t-shirts combo for women t-shirts for women stylish latest t-shirts girls stylish latest t-shirts combo girls fashion ladies and kids tees black and white daily use t-shirts duke western wear menswearm women stylish latest t-shirts maroon white waist coat combo pack collar girls fashion ladies kids black daily use western wear duke classic polo round v-neck seven sea button cotton polyster zipper biowashed navy red grey full sleevesseven sea cotton bio washed maroon black v-neck regular fit stylish t-shirts high quality smooth fabric and quality product and unique style t-shirt available in all sizes fro s-xxl t-shirts combo for women t-shirts for women stylish latest t-shirts for women search terms blended designer v-neck o-neck round-neck zipper hooded hoodie polo collar henley button bio-washed t-shirt dress clothes clothing apparel t-shirts combo for women t-shirts for women stylish latest t-shirts girls stylish latest t-shirts combo girls fashion ladies and kids tees black and white daily use t-shirts duke western wear menswear women women stylish latest t-shirts maroon white waist coat combo pack collar girls fashion ladies kids black daily use western wear duke classic polo round v-neck seven sea button cotton polyster zipper biowashed navy red grey full

 12. [Score: 0.743135] | PID: TSHFYUSGMU8FHHTG
     Title: tie dye women polo-neck white orange t-shirt
     Brand: steenb | Category: clothing and accessories | Sub-category: topwear
     Description: steenbok women s orange dip dye polo t-shirt

 13. [Score: 0.741810] | PID: TSHFXDNVNAH2Y9SR
     Title: solid women polo-neck blue t-shirt
     Brand: steenb | Category: clothing and accessories | Sub-category: topwear
     Description: steenbok women s pima cotton polo t-shirt

 14. [Score: 0.738654] | PID: TSHFYUQZEX4XJCJH
     Title: solid women polo-neck black t-shirt
     Brand: steenb | Category: clothing and accessories | Sub-category: topwear
     Description: women s black denim collar polo t-shirt

 15. [Score: 0.738239] | PID: TSHEMA3YP8FYVBUF
     Title: solid women polo-neck beige t-shirt
     Brand: t spor | Category: clothing and accessories | Sub-category: topwear
     Description: t sports beige half polo t-shirt for women

 16. [Score: 0.738158] | PID: TKPFW45FHTCU9SUE
     Title: solid women grey track pants
     Brand: u s polo associati | Category: clothing and accessories | Sub-category: bottomwear  
     Description: track pant with contrast zipper pockets broad waistband with flat multi color draw cord comfort fit u s polo assn has sub-brands under its umbrella brand u s polo assn denim co u s polo assn tailored and u s polo assn active u s polo assn denim co includes our range of smart sunday brunch options for women like crisp shirts subtle yet smart t-shirts classic polo t-shirts jeans jackets and much more u s polo assn tailored caters to formal wear the distinction of u s polo assn tailored from its other sub-brands lies in its attention to classic details each article is specially crafted to be inviting authentic classic and genuine and is the perfect do-over for your - work wardrobe smart button-downs in classic hues and toned-down prints well-tailored trousers blazers waistcoats and jackets are just a few from the wide array of categories to choose from

 17. [Score: 0.738100] | PID: TSHESGM3KJNVAV2J
     Title: solid women polo-neck maroon t-shirt
     Brand: t spor | Category: clothing and accessories | Sub-category: topwear
     Description: double mercerised polo the world finest quality double mercerized cotton polo shirt brought to you by t sportsthey are premium quality polo shirt crafted mercerized cotton for silky this piece is mercerized processes gives you better resistance to multiple washing and it keeps colors bright over a long durable knitted striped polo t-shirt has smooth hand a ribbed polo collar short button placket on the front brand logo embroidery on the left side of chest packet it gives you a fresh look and always you feel young and smart style it pair of jeans or chinos and casual shoes for a complete look feel classical short sleeve and neat top stitched edges cotton cotton

 18. [Score: 0.737588] | PID: TSHFWVYDSFCFG8GV
     Title: solid women polo-neck yellow t-shirt
     Brand: onei | Category: clothing and accessories | Sub-category: topwear
     Description: refresh your clothing with the awesome collection of basic polo t-shirts from oneiro this t-shirt is made of cotton fabric and gives utmost comfort during all temperatures elegant stitch and solid colours makes this t-shirt a perfect formal and casual wear pair it with jeans or casual for a perfect casual look

 19. [Score: 0.737295] | PID: TSHFGQGQFXPZFDGW
     Title: solid women polo-neck green t-shirt
     Brand: pu | Category: clothing and accessories | Sub-category: topwear
     Description: womens graphic polo i amazon green

 20. [Score: 0.736534] | PID: TSHFXDNVSD3TGSXM
     Title: solid women polo-neck white t-shirt
     Brand: steenb | Category: clothing and accessories | Sub-category: topwear
     Description: steenbok women s pima cotton polo t-shirt

... and 709 more results (not shown)
================================================================================


Query: casual clothes slim fit
Query terms: ['casual', 'clothes', 'slim', 'fit']
No documents found matching all query terms.
================================================================================

Query: biowash innerwear
Query terms: ['biowash', 'innerwear']

Total results: 59
Showing top 20 results:

  1. [Score: -0.175926] | PID: VESFGYJCNNHV7KNG
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest we made your favourite sleeveless charcoal melange colored vest a little better by adding a quality material on the vest planning a house party or a reunion simply put on this vest and pair it with denim sneakers and youre ready to roll the vest is available in other color options choose your favourite one if you want to look for more options then do check out our entire collection of sayitloud for women t-shirts vest joggers track pants boxers shorts and jackets sayitloud fashion lifestyle workout vest sportswear gymwear vest vest trendy printed vest solid vest latest vest designer vest sayit loud say it loud slim fitvest regular fit vest cut and sew vest deal of the day round-neck vest v-neck vest casual vest regular vest branded vest sleeveless vest

  2. [Score: -0.178351] | PID: VESFGYJCHH6UX6VF
     Title: sayitloud men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest with a quotation chaos we made your favourite sleeveless charcoal colored vest a little better by adding a handsome cut and sew design and quotation on the vest planning a house party or a reunion simply put on this vest and pair it with denim sneakers and youre ready to roll the vest is available in other color options choose your favourite one if you want to look for more options then do check out our entire collection of sayitloud for men t-shirts vest joggers track pants boxers shorts and jackets sayitloud fashion lifestyle workout vest sportswear gymwear vest vest trendy printed vest solid vest latest vest designer vest sayit loud say it loud slim fitvest regular fit vest cut and sew vest deal of the day round-neck vest v-neck vest casual vest regular vest branded vest sleeveless vest

  3. [Score: -0.182520] | PID: VESFX3ZFXTNQXDFN
     Title: free authority men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: look trendy and feel comfortable with this character printed sleevless vest featuring scooby doo crafted out of cotton which is biowashed for smooth feel and befriend to skin this featuring can be worn for any occasion a casual day at work or for a fun filled weekend or leisure wear

  4. [Score: -0.184279] | PID: VESFGYJCTBYDZWE2
     Title: sayitloud men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest with a quotation challenge we made your favourite sleeveless navy blue colored vest a little better by adding a handsome cut and sew design and quotation on the vest planning a house party or a reunion simply put on this vest and pair it with denim sneakers and youre ready to roll the vest is available in other color options choose your favourite one if you want to look for more options then do check out our entire collection of sayitloud for men t-shirts vest joggers track pants boxers shorts and jackets sayitloud fashion lifestyle workout vest sportswear gymwear vest vest trendy printed vest solid vest latest vest designer vest sayit loud say it loud slim fitvest regular fit vest cut and sew vest deal of the day round-neck vest v-neck vest casual vest regular vest branded vest sleeveless vest

  5. [Score: -0.184614] | PID: VESFGYJCZD3JUZRV
     Title: sayitloud men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest we made your favourite sleeveless olive green colored vest a little better by adding a quality material on the vest planning a house party or a reunion simply put on this vest and pair it with denim sneakers and youre ready to roll the vest is available in other color options choose your favourite one if you want to look for more options then do check out our entire collection of sayitloud for men t-shirts vest joggers track pants boxers shorts and jackets sayitloud fashion lifestyle workout vest sportswear gymwear vest vest trendy printed vest solid vest latest vest designer vest sayit loud say it loud slim fitvest regular fit vest cut and sew vest deal of the day round-neck vest v-neck vest casual vest regular vest branded vest sleeveless vest

  6. [Score: -0.186819] | PID: VESFGYJDAGF4QZ3V
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest we made your favourite sleeveless wine grey colored vest a little better by adding a handsome cut and sew design on the vest planning a house party or a reunion simply put on this vest and pair it with denim sneakers and youre ready to roll the vest is available in other color options choose your favourite one if you want to look for more options then do check out our entire collection of sayitloud for women t-shirts vest joggers track pants boxers shorts and jackets sayitloud fashion lifestyle workout vest sportswear gymwear vest vest trendy printed vest solid vest latest vest designer vest sayit loud say it loud slim fitvest regular fit vest cut and sew vest deal of the day round-neck vest v-neck vest casual vest regular vest branded vest sleeveless vest

  7. [Score: -0.187226] | PID: VESFGYJDQHGZS3SM
     Title: sayitloud men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest with a quotation beast we made your favourite sleeveless charcoal black colored vest a little better by adding a handsome cut and sew design and quotation on the vest planning a house party or a reunion simply put on this vest and pair it with denim sneakers and youre ready to roll the vest is available in other color options choose your favourite one if you want to look for more options then do check out our entire collection of sayitloud for men t-shirts vest joggers track pants boxers shorts and jackets sayitloud fashion lifestyle workout vest sportswear gymwear vest vest trendy printed vest solid vest latest vest designer vest sayit loud say it loud slim fitvest regular fit vest cut and sew vest deal of the day round-neck vest v-neck vest casual vest regular vest branded vest sleeveless vest

  8. [Score: -0.187763] | PID: VESFGYJDQVNEA6DE
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest we made your favourite sleeveless charcoal white colored vest a little better by adding a handsome cut and sew design on the vest planning a house party or a reunion simply put on this vest and pair it with denim sneakers and youre ready to roll the vest is available in other color options choose your favourite one if you want to look for more options then do check out our entire collection of sayitloud for women t-shirts vest joggers track pants boxers shorts and jackets sayitloud fashion lifestyle workout vest sportswear gymwear vest vest trendy printed vest solid vest latest vest designer vest sayit loud say it loud slim fitvest regular fit vest cut and sew vest deal of the day round-neck vest v-neck vest casual vest regular vest branded vest sleeveless vest

  9. [Score: -0.188609] | PID: VESFX3ZFC7XXJ5CR
     Title: free authority women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: look trendy and feel comfortable with this character printed sleevless vest featuring nasa crafted out of cotton which is biowashed for smooth feel and befriend to skin this featuring can be worn for any occasion a casual day at work or for a fun filled weekend or leisure wear

 10. [Score: -0.189085] | PID: VESFGYJCZH7GMD4E
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest with a quotation we made your favourite sleeveless black green colored vest a little better by adding a handsome cut and sew design and quotation on the vest planning a house party or a reunion simply put on this vest and pair it with denim sneakers and youre ready to roll the vest is available in other color options choose your favourite one if you want to look for more options then do check out our entire collection of sayitloud for women t-shirts vest joggers track pants boxers shorts and jackets sayitloud fashion lifestyle workout vest sportswear gymwear vest vest trendy printed vest solid vest latest vest designer vest sayit loud say it loud slim fitvest regular fit vest cut and sew vest deal of the day round-neck vest v-neck vest casual vest regular vest branded vest sleeveless vest

 11. [Score: -0.189770] | PID: VESFGYJDBKHSE5H6
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest we made your favourite sleeveless navy blue maroon white colored vest a little better by adding a handsome cut and sew design on the vest planning a house party or a reunion simply put on this vest and pair it with denim sneakers and youre ready to roll the vest is available in other color options choose your favourite one if you want to look for more options then do check out our entire collection of sayitloud for women t-shirts vest joggers track pants boxers shorts and jackets sayitloud fashion lifestyle workout vest sportswear gymwear vest vest trendy printed vest solid vest latest vest designer vest sayit loud say it loud slim fitvest regular fit vest cut and sew vest deal of the day round-neck vest v-neck vest casual vest regular vest branded vest sleeveless vest

 12. [Score: -0.190065] | PID: VESFX3ZFJEKDYZPW
     Title: free authority women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: look trendy and feel comfortable with this character printed sleevless vest featuring the simpsons crafted out of cotton which is biowashed for smooth feel and befriend to skin this featuring can be worn for any occasion a casual day at work or for a fun filled weekend or leisure wear

 13. [Score: -0.190652] | PID: VESFGYJCX9TZN6ZC
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest we made your favourite sleeveless charcoal yellow grey colored vest a little better by adding a handsome cut and sew design on the vest planning a house party or a reunion simply put on this vest and pair it with denim sneakers and youre ready to roll the vest is available in other color options choose your favourite one if you want to look for more options then do check out our entire collection of sayitloud for women t-shirts vest joggers track pants boxers shorts and jackets sayitloud fashion lifestyle workout vest sportswear gymwear vest vest trendy printed vest solid vest latest vest designer vest sayit loud say it loud slim fitvest regular fit vest cut and sew vest deal of the day round-neck vest v-neck vest casual vest regular vest branded vest sleeveless vest

 14. [Score: -0.190809] | PID: VESFGYJCB6BVH8G4
     Title: sayitloud men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest with a quotation victory we made your favourite sleeveless black green colored vest a little better by adding a handsome cut and sew design and quotation on the vest planning a house party or a reunion simply put on this vest and pair it with denim sneakers and youre ready to roll the vest is available in other color options choose your favourite one if you want to look for more options then do check out our entire collection of sayitloud for men t-shirts vest joggers track pants boxers shorts and jackets sayitloud fashion lifestyle workout vest sportswear gymwear vest vest trendy printed vest solid vest latest vest designer vest sayit loud say it loud slim fitvest regular fit vest cut and sew vest deal of the day round-neck vest v-neck vest casual vest regular vest branded vest sleeveless vest

 15. [Score: -0.190891] | PID: VESFGYJCGXYEZYHT
     Title: sayitloud men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest with a quotation victory we made your favourite sleeveless steel grey black colored vest a little better by adding a handsome cut and sew design and quotation on the vest planning a house party or a reunion simply put on this vest and pair it with denim sneakers and youre ready to roll the vest is available in other color options choose your favourite one if you want to look for more options then do check out our entire collection of sayitloud for men t-shirts vest joggers track pants boxers shorts and jackets sayitloud fashion lifestyle workout vest sportswear gymwear vest vest trendy printed vest solid vest latest vest designer vest sayit loud say it loud slim fitvest regular fit vest cut and sew vest deal of the day round-neck vest v-neck vest casual vest regular vest branded vest sleeveless vest

 16. [Score: -0.190963] | PID: VESFGYJCADUGKVGG
     Title: sayitloud men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest with a quotation xxix we made your favourite sleeveless parry red grey colored vest a little better by adding a handsome cut and sew design and quotation on the vest planning a house party or a reunion simply put on this vest and pair it with denim sneakers and youre ready to roll the vest is available in other color options choose your favourite one if you want to look for more options then do check out our entire collection of sayitloud for men t-shirts vest joggers track pants boxers shorts and jackets sayitloud fashion lifestyle workout vest sportswear gymwear vest vest trendy printed vest solid vest latest vest designer vest sayit loud say it loud slim fitvest regular fit vest cut and sew vest deal of the day round-neck vest v-neck vest casual vest regular vest branded vest sleeveless vest

 17. [Score: -0.191112] | PID: VESFGYJDTX7WJBT7
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest we made your favourite sleeveless black green colored vest a little better by adding a handsome cut and sew design on the vest planning a house party or a reunion simply put on this vest and pair it with denim sneakers and youre ready to roll the vest is available in other color options choose your favourite one if you want to look for more options then do check out our entire collection of sayitloud for women t-shirts vest joggers track pants boxers shorts and jackets sayitloud fashion lifestyle workout vest sportswear gymwear vest vest trendy printed vest solid vest latest vest designer vest sayit loud say it loud slim fitvest regular fit vest cut and sew vest deal of the day round-neck vest v-neck vest casual vest regular vest branded vest sleeveless vest

 18. [Score: -0.192758] | PID: VESFGYJCY74MT7YW
     Title: sayitloud men vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest with a quotation boston we made your favourite sleeveless steel grey grey colored vest a little better by adding a handsome cut and sew design and quotation on the vest planning a house party or a reunion simply put on this vest and pair it with denim sneakers and youre ready to roll the vest is available in other color options choose your favourite one if you want to look for more options then do check out our entire collection of sayitloud for men t-shirts vest joggers track pants boxers shorts and jackets sayitloud fashion lifestyle workout vest sportswear gymwear vest vest trendy printed vest solid vest latest vest designer vest sayit loud say it loud slim fitvest regular fit vest cut and sew vest deal of the day round-neck vest v-neck vest casual vest regular vest branded vest sleeveless vest

 19. [Score: -0.193167] | PID: VESFGYJDGBAHTK8A
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest with a quotation previous we made your favourite sleeveless navy blue white maroon colored vest a little better by adding a handsome cut and sew design and quotation on the vest planning a house party or a reunion simply put on this vest and pair it with denim sneakers and youre ready to roll the vest is available in other color options choose your favourite one if you want to look for more options then do check out our entire collection of sayitloud for women t-shirts vest joggers track pants boxers shorts and jackets sayitloud fashion lifestyle workout vest sportswear gymwear vest vest trendy printed vest solid vest latest vest designer vest sayit loud say it loud slim fitvest regular fit vest cut and sew vest deal of the day round-neck vest v-neck vest casual vest regular vest branded vest sleeveless vest     

 20. [Score: -0.194381] | PID: VESFGYJCU2PM9HXR
     Title: sayitloud women vest
     Brand:  | Category: clothing and accessories | Sub-category: innerwear and swimwear        
     Description: the world is boring without a little twist and a twist is what we have added to your favourite round-neck cotton vest with a quotation xxix we made your favourite sleeveless navy blue red colored vest a little better by adding a handsome cut and sew design and quotation on the vest planning a house party or a reunion simply put on this vest and pair it with denim sneakers and youre ready to roll the vest is available in other color options choose your favourite one if you want to look for more options then do check out our entire collection of sayitloud for women t-shirts vest joggers track pants boxers shorts and jackets sayitloud fashion lifestyle workout vest sportswear gymwear vest vest trendy printed vest solid vest latest vest designer vest sayit loud say it loud slim fitvest regular fit vest cut and sew vest deal of the day round-neck vest v-neck vest casual vest regular vest branded vest sleeveless vest

... and 39 more results (not shown)
================================================================================





***AI Use*: AI tools were used to help design the general structure of some functions. However, the code logic, analysis approach, test queries, and relevance judgments were fully developed and verified by the team. All AI-generated parts were carefully reviewed, corrected, and adjusted to meet the project requirements. The insights, interpretations, and conclusions presented in this report are entirely the team’s own work.**
