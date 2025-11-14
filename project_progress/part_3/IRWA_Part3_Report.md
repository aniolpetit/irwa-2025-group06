# Part 3: Ranking & Filtering

## 1. Ranking Methods Implementation

### 1.1 TF-IDF + Cosine Similarity

For the TF-IDF + cosine similarity ranking, we reused the implementation from Part 2 of the project (`TFIDFRanker` class). The ranking follows the classical approach:

- **TF (Term Frequency)**: Uses logarithmic weighting: `w_tf = 1 + log₂(freq)`
- **IDF (Inverse Document Frequency)**: `w_idf = log₂(N / df)` where N is the total number of documents and df is the document frequency
- **TF-IDF Weight**: `w = w_tf × w_idf`
- **Document Length Normalization**: `length(d) = sqrt(Σ w_i²)`
- **Cosine Similarity Score**: `score(d, q) = (query_vector · doc_vector) / length(d)`

The implementation performs conjunctive query filtering (AND) to find all documents containing all query terms, then ranks them using the cosine similarity between the TF-IDF-weighted query and document vectors.

### 1.2 BM25

BM25 is a probabilistic ranking function that addresses some limitations of TF-IDF, particularly its tendency to over-penalize long documents and its linear term frequency saturation.

**BM25 Formula:**
```
score(d, q) = [Σ idf(t)] × (tf(t,d) × (k1 + 1)) / (tf(t,d) + k1 × (1 - b + b × |d|/avgdl))
```

Where:
- `idf(t) = log((N - df(t) + 0.5) / (df(t) + 0.5))`
- `tf(t,d)` is the term frequency of term t in document d
- `|d|` is the document length in tokens
- `avgdl` is the average document length
- `k1 = 1.2` (TREC value, controls term frequency saturation)
- `b = 0.75` (TREC value, controls length normalization)


### 1.2.1 Comparison: TF-IDF vs BM25

The main differences between TF-IDF and BM25 lie in how they handle document length and term frequency:

**Length Normalization:**
- **TF-IDF**: Cosine normalization can over-penalize longer documents, often favoring shorter documents even when longer ones contain more relevant information.
- **BM25**: Tunable length normalization (`b = 0.75`) provides more balanced treatment, allowing longer documents with detailed descriptions to rank appropriately.

**Term Frequency Saturation:**
- **TF-IDF**: Logarithmic scaling still allows very high term frequencies to dominate scores.
- **BM25**: Non-linear saturation function prevents excessive dominance from repeated terms, making it more robust to natural term repetition in well-written documents.

**Practical Impact:**
In our experiments, BM25 tends to rank documents with detailed product descriptions higher, while TF-IDF favors documents with minimal or no descriptions. For example, for the query "ecko unl shirt", BM25's top results include documents with full product descriptions, whereas TF-IDF's top results often have "N/A" descriptions.

**Pros and Cons:**

**TF-IDF:**
- **Pros**: Simple to implement and interpret; well-established baseline; works well for collections with uniform document lengths; favors concise, focused documents.
- **Cons**: Can over-penalize longer documents; logarithmic TF scaling still allows high frequencies to dominate; less robust to natural term repetition in detailed descriptions.

**BM25:**
- **Pros**: Better length normalization with tunable parameters; non-linear term frequency saturation prevents keyword stuffing; more robust to natural term repetition; better suited for heterogeneous document collections; probabilistic foundation.
- **Cons**: Requires parameter tuning (`k1` and `b`); more complex than TF-IDF; like TF-IDF, lacks semantic understanding and treats terms as independent.

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

