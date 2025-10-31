
# Queries

- ecko unl shirt --> high popularity low specific
- ecko unlmen shirt round neck --> high popularity and very specific
- women polo cotton --> mid popularity and mid specific
- casual clothes slim fit --> low popularity and mid specific
- biowash innerwear --> low popularity low specific

We decided to select the queries based on their **popularity** and **specificity**. We included queries with high popularity to evaluate whether our retrieval engine can correctly narrow down and rank the most relevant documents among a large pool of potential matches. On the other hand, we added low-popularity queries to test the engine’s ability to retrieve the few documents that contain these rarer terms.
Furthermore, we applied the same reasoning to specificity: by including both highly specific and more general queries, we can assess whether the engine handles different levels of query precision effectively. Highly specific queries help us test if the engine can accurately match detailed user intent, while broader ones allow us to evaluate how well it manages ambiguity and relevance ranking across a wider range of possible results.

# Part 2 (Evaluation) Exercise 3

## a. For the test queries you defined in Part 1, Step 2 during indexing, assign a binary relevance label to each document: 1 if the document is relevant to the query, or 0 if it is not.

- Full Sleeve Printed Women Sweatshirt
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Full Sleeve Striped Women Sweatshirt
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Full Sleeve Printed Women Sweatshirt
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Full Sleeve Graphic Print Women Sweatshirt
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Full Sleeve Solid Women Sweatshirt
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Full Sleeve Graphic Print Women Sweatshirt
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Typography Women V Neck Multicolor T-Shirt  (Pack of 2)
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Full Sleeve Printed Women Sweatshirt
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Full Sleeve Solid Women Sweatshirt
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Full Sleeve Graphic Print Women Sweatshirt
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Full Sleeve Graphic Print Women Sweatshirt
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Typography Women V Neck Multicolor T-Shirt  (Pack of 2)
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Full Sleeve Printed Women Sweatshirt
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Printed Women V Neck Multicolor T-Shirt  (Pack of 2)
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Full Sleeve Color Block Women Sweatshirt
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Full Sleeve Graphic Print Women Sweatshirt
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Full Sleeve Graphic Print Women Sweatshirt
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Full Sleeve Solid Women Sweatshirt
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Full Sleeve Solid Women Sweatshirt
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Superhero Women Round Neck Multicolor T-Shirt  (Pack of 2)
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Slim Men Dark Blue Jeans
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Tapered Fit Men Blue Jeans
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Super Skinny Women Blue Jeans
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Slim Women Blue Jeans
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Slim Women Dark Blue Jeans
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Slim Men Blue Jeans
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Tapered Fit Men Blue Jeans
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)
- Super Skinny Men Blue Jeans
    - Query 1 (ecko unl shirt) (**0**)
    - Query 2 (ecko unlmen shirt round neck) (**0**)
    - Query 3 (women polo cotton) (**0**)
    - Query 4 (casual clothes slim fit) (**0**)
    - Query 5 (biowash innerwear) (**0**)

## b. Comment on each of the evaluation metrics, stating how they differ, and which information gives each of them. Analyze your results.

The evaluation metrics used provide different perspectives on the performance of the retrieval system. Precision@K measures how many of the top K retrieved documents are relevant, showing the accuracy of the system’s highest-ranked results. Recall@K, on the other hand, reflects how many of all relevant documents in the corpus were retrieved, emphasizing completeness. These two often balance each other—high precision can mean lower recall and vice versa—so both are needed to understand overall effectiveness.

Average Precision@K combines precision and ranking quality, rewarding systems that return relevant documents earlier in the list. The F1-Score@K provides a single measure that balances precision and recall equally. Mean Average Precision (MAP) extends this concept across all queries, summarizing the overall retrieval consistency. Mean Reciprocal Rank (MRR) focuses on how early the first relevant document appears, offering insight into how quickly a user might find something useful. Normalized Discounted Cumulative Gain (NDCG) further emphasizes ranking order, giving higher value to relevant documents that appear near the top.

Analyzing the results obtained, the system shows relatively low values across most metrics. Precision and F1 scores are near zero for small K values, indicating that the top-ranked documents often fail to include relevant results. Recall@K reaches high values (even up to 1.0) only at large K, meaning that while relevant documents exist in the corpus, they are not ranked highly enough. The low Average Precision and MAP confirm this pattern: relevant documents tend to appear deep in the ranking, diminishing the perceived usefulness of the search results. The NDCG values follow the same trend, starting very low at small K and increasing only when many documents are considered, suggesting weak ranking discrimination. Finally, the modest MRR indicates that the first relevant document appears relatively far down the list. Altogether, these results suggest that while the system is capable of retrieving relevant documents eventually, its current TF-IDF ranking lacks effectiveness in promoting them to the top positions, pointing to a need for more refined weighting, query expansion, or improved preprocessing to enhance retrieval quality.

## c. Analyze the current search system and identify its main problems or limitations. For each issue you find, propose possible ways to resolve it. Consider aspects such as retrieval accuracy, ranking quality, handling of different field types, query formulation, and indexing strategies.

The current search system, while functional, has several limitations that restrict its effectiveness in delivering relevant and well-ranked results. One of the main issues lies in its retrieval model, which is based on a simple conjunctive query approach using an inverted index. This means that only documents containing all the query terms are retrieved. Although this ensures precision, it sacrifices recall (many partially relevant documents are ignored simply because they lack one of the query terms, which might be expressed with a synonim for instance). A more flexible retrieval strategy, such as a vector-space model or probabilistic ranking like BM25, would address this by weighting terms by importance and allowing partial matches, producing a smoother balance between precision and recall.

The ranking quality is also limited by now. Although the system incorporates TF-IDF scoring to improve ranking quality, it still faces limitations in capturing semantic relationships and contextual relevance. TF-IDF treats terms as independent and doesn’t account for phrase meaning, synonymy, or field-specific importance. As a result, some documents with high term frequency may appear highly ranked even if they’re contextually less relevant. Moving toward more advanced models such as BM25, embedding-based retrieval, or field-weighted TF-IDF could improve ranking quality by better reflecting semantic importance and user intent.

Another structural weakness comes from the uniform treatment of fields. As we defined in the previous deliver, product data contains several fields (e.g., title, brand, description), each contributing differently to relevance. The current index merges tokens without distinguishing their source, so a term appearing in a long description has the same weight as one in the title. Introducing field weighting or multi-field indexing (for example, giving higher scores to title matches) would make the retrieval more semantically meaningful and better aligned with user intent.

The query formulation process also lacks sophistication. The system performs literal term matching, ignoring morphological variations, synonyms, or spelling errors. As a result, users must phrase their queries precisely, or the engine will miss relevant results. Incorporating query expansion (using synonyms or related terms), stemming/lemmatization consistency, and spell correction would make the engine more robust and user-friendly.

In summary, the system’s main limitations stem from a rigid retrieval model, unweighted ranking, undifferentiated field handling, and simplistic query processing. Addressing these issues through advanced ranking algorithms, semantic indexing, and enhanced query understanding would transform it into a more accurate, flexible, and intelligent search engine.
