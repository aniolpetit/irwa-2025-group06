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