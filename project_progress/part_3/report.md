
# Part 3: Ranking & Filtering

## Our Score

For our custom ranking function, we designed a composite scoring model that extends the traditional TF-IDF approach with additional weighting factors reflecting both textual relevance and product-specific attributes. First, all weights are normalized to sum to 1 to maintain balance across components. Textual relevance is enhanced by incorporating term proximity, giving slightly higher scores to documents where query terms appear close to each other, and field-based weighting, where occurrences in the title, brand, or description receive different importance levels (e.g., matches in the title contribute more to relevance). Beyond text-based factors, we integrate numeric attributes to capture user-oriented relevance: products that are in stock are prioritized, and those with higher average ratings receive additional weight. Finally, we apply length normalization to penalize overly long descriptions that might artificially inflate TF-IDF scores. This combination allows our ranking function to account for both linguistic relevance and product quality, leading to more meaningful and user-centric search results.

**TO DO**: Justify why we have done that specific implementation of the query term proximity in the document.
