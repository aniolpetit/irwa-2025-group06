# IRWA Final Project 2025 - Group 06

## Team Members
- **Jan Aguiló**
- **Adrià Cortés** 
- **Tània Pazos**
- **Aniol Petit**

---

## Part 1: Text Processing and Exploratory Data Analysis

### What is Part 1?

Part 1 focuses on preparing a fashion products dataset (28,080 records) for information retrieval tasks. It includes:

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

### Key Functions Available
- **Preprocessing**: `clean_text()`, `tokenize_text()`, `remove_stopwords()`, `stem_tokens()`
- **Analysis**: `text_stats()`, `numeric_summary()`, `categorical_summary()`
- **Visualization**: `plot_numeric_hist()`, `generate_wordcloud()`, `plot_categorical_bar()`

For detailed documentation, see `project_progress/part_1/IRWA_Part1_Report.md`, also attached in Aula Global's task.