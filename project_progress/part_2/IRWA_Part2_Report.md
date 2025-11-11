
# Part 1: Indexing

## Defining Test Queries

- **Query**: `ecko unl shirt` (very popular, not specific)
- **Query**: `ecko unl men shirt` round neck (very popular, very specific)
- **Query**: `women polo cotton` (very popular, very specific)
- **Query**: `casual clothes slim fit` (not popular, moderately specific)
- **Query**: `biowash innerwear` (not popular, not specific)

We decided to select the queries based on their **popularity** and **specificity**. We included queries with high popularity to evaluate whether our retrieval engine can correctly narrow down and rank the most relevant documents among a large pool of potential matches. Conversely, we added low-popularity queries to test the engine’s ability to retrieve the few documents that contain these rarer terms.

Furthermore, we applied the same reasoning to specificity: by including both highly specific and more general queries, we can assess whether the engine handles different levels of query precision effectively. Highly specific queries help us test if the engine can accurately match detailed user intent, while broader ones allow us to evaluate how well it manages ambiguity and relevance ranking across a wider range of possible results.

# Part 2: Evaluation

## Ground Truth Definition and System Evaluation

**a. For the test queries you defined in Part 1, Step 2 during indexing, assign a binary relevance label to each document: 1 if the document is relevant to the query, or 0 if it is not.**

As seen in Exercise 1.3, the conjunctive queries produced highly variable result sizes, from 0 up to more than 1300 documents, consistent with their different popularity and specificity levels. For simplicity purposes, we will be labelling only the top 10 retrieved documents per each query.

- **Query**: `ecko unl shirt`
    1. men slim fit printed casual shirt **1**
    2. men slim fit printed casual shirt **1**
    3. women slim fit printed casual shirt **1**
    4. solid women round neck white t shirt **0**
    5. solid women round neck white t shirt **0**
    6. men slim fit solid casual shirt **1**
    7. men slim fit solid casual shirt **1**
    8. solid women round neck white t shirt **0**
    9. women slim fit solid casual shirt **1**
    10. men slim fit solid casual shirt **1**

    Documents labeled as 0 were excluded because a t-shirt is considered a different clothing category from a shirt, and therefore not relevant to the query intent (further discussed in Section 3c).

- **Query**: `ecko unl men shirt round neck`
    1. printed men round neck black t shirt **0** (again, t-shirt)
    2. printed men round neck black t shirt **0**
    3. printed men round neck black t shirt **0**
    4. printed men round neck black t shirt **0**
    5. solid men round neck white t shirt **0**
    6. printed men round neck white t shirt **0**
    7. solid men round neck multicolor t shirt **0**
    8. printed men round neck white t shirt **0**
    9. solid men round neck multicolor t shirt **0**
    10. solid men round neck blue t shirt **0**

- **Query**: `women polo cotton`

    1. solid women polo neck black t shirt **1**
    2. solid women polo neck black t shirt **1**
    3. solid women polo neck black t shirt **1**
    4. solid women polo neck white t shirt **1**
    5. solid women polo neck white t shirt **1**
    6. printed women polo neck black blue t shirt **1**
    7. solid women polo neck multicolor t shirt pack of **1**
    8. printed women polo neck blue t shirt **1**
    9. solid women polo neck black t shirt **1**
    10. solid women polo neck white t shirt **1**

    Although this information is not visible in the output above,  inspecting the documents by their PIDs (retrieved in `tfidf_ranking.py`) in the `fashion_products_dataset.json` reveals that the Fabric field for each ranked document is either “Cotton Blend” or “Pure Cotton”. Therefore, these documents are considered relevant to the query.

- **Query**: `casual clothes slim fit`

    No documents found matching all query terms.

- **Query**: `biowash innerwear`
    1. free authority men vest **1**
    2. free authority women vest **1**
    3. free authority women vest **1**
    4. free authority women vest **1**
    5. free authority men vest **1**
    6. printed men boxer pack of **1**
    7. sayitloud women vest **1**
    8. sayitloud women vest **1**
    9. sayitloud women vest **1**
    10. sayitloud men vest **1** 

*Note*: All relevance assessments were performed by carefully reviewing every available field in the products dataset. Decisions were based on information such as description, fabric, and other attributes to ensure that the final relevance judgments were as accurate and consistent as possible.

**b. Comment on each of the evaluation metrics, stating how they differ, and which information gives each of them. Analyze your results.**

The evaluation metrics used provide different perspectives on the performance of the retrieval system. Precision@K measures how many of the top K retrieved documents are relevant, showing the accuracy of the system’s highest-ranked results. Recall@K, on the other hand, reflects how many of all relevant documents in the corpus were retrieved, emphasizing completeness. These two often balance each other, i.e., high precision can mean lower recall and vice versa, so both are needed to understand overall effectiveness.

Average Precision@K combines precision and ranking quality, rewarding systems that return relevant documents earlier in the list. F1-Score@K provides a single measure that balances precision and recall equally. Mean Average Precision (MAP) extends this concept across all queries, summarizing the overall retrieval consistency. Mean Reciprocal Rank (MRR) focuses on how early the first relevant document appears, offering insight into how quickly a user might find something useful. Normalized Discounted Cumulative Gain (NDCG) further emphasizes ranking order, giving higher value to relevant documents that appear near the top.

Analyzing the obtained results, the system shows relatively low values across most metrics. Precision and F1 scores are near zero for small K values, indicating that the top-ranked documents often fail to include relevant results. Recall@K reaches high values only at large K, meaning that while relevant documents exist in the corpus, they are not ranked highly enough. The low Average Precision and MAP confirm this pattern: relevant documents tend to appear deep in the ranking, diminishing the perceived usefulness of the search results. The NDCG values follow the same trend, starting very low at small K and increasing only when many documents are considered, suggesting weak ranking discrimination. Finally, the modest MRR indicates that the first relevant document appears relatively far down the list. Altogether, these results suggest that while the system is capable of retrieving relevant documents eventually, its current TF-IDF ranking lacks effectiveness in promoting them to the top positions, pointing to a need for more refined weighting, query expansion, or improved preprocessing to enhance retrieval quality.

**c. Analyze the current search system and identify its main problems or limitations. For each issue you find, propose possible ways to resolve it. Consider aspects such as retrieval accuracy, ranking quality, handling of different field types, query formulation, and indexing strategies.**

The current search system, while functional, has several limitations that restrict its effectiveness in delivering relevant and well-ranked results. One of the main issues lies in its retrieval model, which is based on a simple conjunctive query approach using an inverted index. This means that only documents containing all the query terms are retrieved. Although this ensures precision, it sacrifices recall (many partially relevant documents are ignored simply because they lack one of the query terms, which could be expressed with a synonym). A more flexible retrieval strategy, such as a vector-space model or probabilistic ranking like BM25, would solve this by weighting terms by importance and allowing partial matches, producing a smoother balance between precision and recall.

The current system’s ranking quality remains limited. Although the system incorporates TF-IDF scoring with document length normalisation to improve ranking quality, it still faces limitations in capturing semantic relationships and contextual relevance. TF-IDF treats terms as independent and does not account for phrase meaning, synonymy, or field-specific importance. As a result, some documents with high term frequency may appear highly ranked even if they are contextually less relevant. Moving toward more advanced models such as BM25, embedding-based retrieval, or field-weighted TF-IDF could improve ranking quality by better reflecting semantic importance and user intent.

A related issue emerges from the preprocessing pipeline's handling of hyphenated terms. During text cleaning, hyphens are removed and replaced with spaces (e.g., "t-shirt" becomes "t shirt"), which is then tokenized into separate tokens ["t", "shirt"]. This causes semantic confusion when searching: a query for "shirt" incorrectly matches documents containing "t-shirt" because both share the token "shirt", despite representing fundamentally different product types (a formal shirt versus a casual t-shirt). This problem affects both retrieval accuracy and ranking quality, as the system cannot distinguish between these semantically distinct concepts. This issue could be addressed by preserving hyphenated terms as single tokens during preprocessing (treating "t-shirt" as one unit rather than splitting it), or by employing models that capture term dependencies and phrase-level semantics, such as n-gram indexing, phrase queries, or embedding models that naturally handle multi-word expressions without assuming term independence. A more straightforward solution would be to identify cases similar to the "t-shirt" example and explicitly handle them through predefined rules in our retrieval strategy.

Another structural weakness comes from the uniform treatment of fields. As described in the previous deliverable, product data contains several fields (e.g., title, brand, description), each contributing differently to relevance. The current index merges tokens without distinguishing their source, so a term appearing in a long description has the same weight as one in the title. Introducing field weighting or multi-field indexing (for example, giving higher scores to title matches) would make the retrieval more semantically meaningful and better aligned with user intent.

The query formulation process also lacks sophistication. The system performs literal term matching, ignoring morphological variations, synonyms, or spelling errors. As a result, users must phrase their queries precisely, or the engine will miss relevant results. Incorporating query expansion (using synonyms or related terms) and spell correction would make the engine more robust and user-friendly.

In summary, the system’s main limitations stem from a rigid retrieval model, unweighted ranking, undifferentiated field handling, and simplistic query processing. Addressing these issues through advanced ranking algorithms, semantic indexing, and enhanced query understanding would transform it into a more accurate, flexible, and intelligent search engine.

***AI Use*: AI tools were used to help design the general structure of some functions. However, the code logic, analysis approach, test queries, and relevance judgments were fully developed and verified by the team. All AI-generated parts were carefully reviewed, corrected, and adjusted to meet the project requirements. The insights, interpretations, and conclusions presented in this report are entirely the team’s own work.**