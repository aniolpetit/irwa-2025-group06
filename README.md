# IRWA Final Project 2025 - Group 06

## Team Members
- **Jan Aguiló**
- **Adrià Cortés** 
- **Tània Pazos**
- **Aniol Petit**

---

## Part 1: Text Processing and Exploratory Data Analysis

### What is Part 1?

Part 1 focuses on preparing a fashion products dataset for information retrieval tasks. It includes:

1. **Data Preprocessing**: Cleaning text, tokenization, stopword removal, stemming
2. **Exploratory Data Analysis**: Statistical analysis and visualizations of the dataset

### How to Run Part 1

#### Prerequisites
1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Download NLTK data (run once):
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

#### Step 1: Data Preprocessing
```bash
cd project_progress/part_1
jupyter notebook IRWA_Part1_Preparation.ipynb
```
- Run all cells sequentially
- This processes the raw dataset and creates `data/processed_corpus.json`

#### Step 2: Exploratory Data Analysis
```bash
jupyter notebook IRWA_Part1_EDA.ipynb
```
- Run all cells sequentially  
- This generates visualizations in the `figures/` folder

### Output Files
- **`data/processed_corpus.json`**: Preprocessed dataset ready for search
- **`figures/`**: 23 visualization files (histograms, bar charts, word clouds, etc.)

For detailed documentation, see `project_progress/part_1/IRWA_Part1_Report.md`, also attached in Aula Global's task.

---

## Part 2: Information Retrieval and Evaluation

### What is Part 2?

Part 2 implements an information retrieval system using:
1. **Inverted Index**: Conjunctive query processing (AND operations)
2. **TF-IDF Ranking**: Logarithmic TF-IDF scoring with cosine similarity
3. **Evaluation Metrics**: Precision@K, Recall@K, F1@K, Average Precision@K, NDCG@K, MAP, and MRR

### How to Run Part 2

#### Prerequisites
1. Complete Part 1 to generate `project_progress/part_1/data/processed_corpus.json`
2. Ensure all dependencies from Part 1 are installed

**Note:** Part 2 uses only Python standard library modules (`math`, `typing`, `collections`, `json`, `csv`, `array`). No additional dependencies beyond Part 1 are required since Part 2 scripts import only standard library modules and local project files.

#### Step 1: Build and Inspect the Inverted Index

```bash
cd project_progress/part_2
python inverted_index.py
```

**What it does:**
- Loads the processed corpus from Part 1
- Builds an inverted index with term positions for TF-IDF calculation
- Displays vocabulary statistics and sample index entries

**Note:** Debug output is shown only when running this file directly. When imported by other files, verbose output is suppressed by default.

#### Step 2: Run TF-IDF Ranking on Test Queries

```bash
python tfidf_ranking.py
```

**What it does:**
- Loads the corpus and builds the inverted index
- Executes 5 predefined test queries:
- For each query:
  - Performs conjunctive query (AND operation) to find candidate documents
  - Ranks documents using TF-IDF with cosine similarity
  - Displays top 20 ranked results with:

#### Step 3: Evaluate Retrieval System with Validation Data

```bash
python evaluate_validation.py
```

**What it does:**
- Loads validation labels from `../../data/validation_labels.csv`
- Uses 20 pre-retrieved documents per query as the retrieval set
- Computes evaluation metrics by comparing ranked results to ground truth labels:

**Note:** The validation file contains 20 documents per query with binary relevance labels (0=non-relevant, 1=relevant). The evaluation assumes these 20 documents represent the complete retrieval set for each query.

For further details on implementation decisions, see `project_progress/part_2/IRWA_Part2_Report.md`, also attached in Aula Global's task.

---

## Part 3: Advanced Ranking Experiments

### What is Part 3?

Part 3 extends the retrieval pipeline with additional ranking strategies so you can compare different scoring functions side by side:

1. **BM25 Ranking** (`bm25_search.py`)
2. **TF-IDF + Cosine (refined interface)** (`tfidf_cosine_search.py`)
3. **Custom Hybrid Ranker** (`custom_search.py`)
4. **Word2Vec + Cosine Similarity** (`word2vec_cosine_search.py`)

Each script loads the processed corpus and prints the top-20 results for the five benchmark queries used throughout the project.

### How to Run Part 3

#### Prerequisites
1. Create and activate your virtual environment (recommended):
   ```bash
   python -m venv irwa_venv
   source irwa_venv/bin/activate
   ```
2. Install Python dependencies (note the pinned `gensim==4.4.0`, which ships wheels for every modern Python version):
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```
   You can double‑check the installation with:
   ```bash
   python <<'PY'
   import gensim
   print('gensim version:', gensim.__version__)
   PY
   ```
   The output should be `gensim version: 4.4.0`. Earlier releases (e.g., 4.3.x) are *not* compatible with Python ≥3.12.

3. Make sure Part 1 has generated `project_progress/part_1/data/processed_corpus.json`, since every ranker relies on it.

#### Step 1: Run BM25 Baseline
```bash
cd project_progress/part_3
python bm25_search.py
```
- Adjust hyperparameters by editing the call to `run_bm25_for_queries(...)` (e.g., change `k1`, `b`, or `top_k`).

#### Step 2: Run TF-IDF Cosine (Search Wrapper)
```bash
python tfidf_cosine_search.py
```
- Mirrors the Part 2 ranker but reuses the search UI from Part 3.
- Tune the number of returned documents via `top_k`.

#### Step 3: Run Custom Hybrid Ranker
```bash
python custom_search.py
```
- Combines TF-IDF with field weights, proximity, rating signals, and stock penalties.
- Inside `CustomRanker`, tweak constants such as `FIELD_WEIGHTS`, `proximity_weight`, or `rating_weight` to experiment.

#### Step 4: Run Word2Vec + Cosine Ranking
```bash
python word2vec_cosine_search.py
```
- Uses pre-trained word embeddings (averaged) and cosine similarity.
- Key parameters in `run_word2vec_cosine_for_queries`:
  - `model_name`: set to `"glove-wiki-gigaword-100"` by default (fast to download). Switch to `"word2vec-google-news-300"` if you already have the model locally.
  - `model_path`: optional path to a local `.kv`/`.bin`/`.txt` embedding file.
  - `top_k`: number of ranked documents to display.
- If you provide a local model file, the ranker tries `KeyedVectors.load`, then `load_word2vec_format(binary=True/False)` automatically.

### Running Custom Queries or Integrating in Notebooks
Every script exposes a `run_*_for_queries` helper function. You can:
- Import the desired function in your own module or notebook,
- Pass a custom list of queries (`List[str]`),
- Supply a different `corpus_path`, or alter the ranking parameters programmatically.

### Summary of Options
- **Change the active ranker** by running the corresponding script.
- **Modify the default query set** in the `if __name__ == "__main__":` block of each file.
- **Control ranking parameters** via function arguments (`top_k`, `model_name`, `model_path`, BM25’s `k1`/`b`, etc.).
- **Ensure the correct gensim version** is installed (`4.4.0`) to avoid import/build errors regardless of your Python release.

## Part 4: RAG, User Interface, and Web Analytics

### What is Part 4?

Part 4 packages all previous components into a single Flask product search portal. It delivers:

1. A polished search/results/details experience wired to the production corpus.
2. A ranking selector that toggles TF-IDF, BM25, Word2Vec, and a custom hybrid ranker at runtime.
3. Retrieval-Augmented Generation (Groq or OpenAI) that summarizes the best match plus an alternative.
4. Full-funnel analytics covering sessions, missions, queries, clicks, dwell time, geo/device segments, and dashboard visualizations.

### How to Run Part 4

#### Prerequisites
1. Run Part 1 so `project_progress/part_1/data/processed_corpus.json` exists.
2. Make sure all dependencies are installed dependencies (ideally inside `irwa_venv`). For more reference refer to Part 3, when the step by step is explained.
3. Create a `.env` file in the repository root. At minimum set:
   ```
   SECRET_KEY = "afgsreg86sr897b6st8b76va8er76fcs6g8d7"
   DEBUG = True
   SESSION_COOKIE_NAME = "IRWA_SEARCH_ENGINE"
   DATA_FILE_PATH = "data/fashion_products_dataset.json"
   ```
   Optional RAG settings (only needed if you want AI summaries):
   ```
   GROQ_API_KEY=sk-...
   GROQ_MODEL=llama-3.1-8b-instant
   OPENAI_API_KEY=sk-...
   OPENAI_MODEL=gpt-4o-mini
   LLM_PROVIDER=groq   # or openai
   ```
   We used our own API keys in this part, make sure to create yours if you want this functionnality available

#### Step 1: Start the Flask server
```bash
python web_app.py
```
The app listens on `http://localhost:8088` (override host/port in `web_app.py` if necessary).

#### Step 2: Run searches and switch ranking algorithms
1. Visit `/` to load the search page.
2. Enter any query and use the “Ranking method” drop-down to pick **TF-IDF (cosine)**, **BM25**, **Word2Vec (cosine)**, or **Custom hybrid**.
3. Submit the form. The chosen method is stored in the session, displayed on the results page, and reused by document details/back-navigation flows.

#### Step 3: Inspect results, AI summaries, and product details
- Result cards highlight query terms, show product metadata, and expose the AI box (if credentials are available). Without keys, the UI shows a friendly “RAG unavailable” notice.
- Click a title to open `/doc_details?pid=...&search_id=...`, review the two-column layout, and use **Back to Results** to re-run the same query with the preserved ranking method.
- The stats table (`/stats`) lists top-clicked PIDs, while `/dashboard` renders KPI cards plus Altair charts for sessions, devices, dwell, status codes, brands, and price buckets.

#### Step 4: (Optional) Customize the RAG provider
- Set `LLM_PROVIDER=groq` or `openai` to prefer one client. The app automatically falls back to whichever API key is available.
- Adjust `GROQ_MODEL` / `OPENAI_MODEL` to experiment with different LLMs without touching code.
- The generated summary follows a “Best product / Why / Alternative” structure enforced by `myapp/generation/rag.py`.

#### Step 5: Reproducing the analytics demo

To populate every widget in a fresh environment, run the following manual test:

1. Start the Flask app, open two browser contexts (normal + incognito), and consent to the geo prompt in one of them.
2. In the first window, submit at least four queries (include a nonsense query to trigger zero-result counts), open the top three documents per query, wait a few seconds, then return to the results page so dwell events fire. Repeat one query twice to populate “Top Queries”.
3. In the second window, decline the geo prompt, run two different queries, and open at least one result per query; leave one detail tab open for ≈10 seconds to create longer dwell samples.
4. Visit `/dashboard` and `/stats` from both windows to log request traffic. Optionally hit a missing route to record a 404, so the status breakdown chart shows more than HTTP 200s.

Following those steps yields non-zero values everywhere: KPI cards update, missions show as “search journey”, the geo/device/OS counters differentiate “Unknown” from real locations, price/brand charts render with slices/bars, and the dwell histogram displays returning-time buckets.