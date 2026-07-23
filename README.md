# Project | Business Case: Automated Customer Reviews

<br>

## Project Goal

This project aims to develop a product review system powered by NLP models that aggregate customer feedback from different sources. The key tasks include classifying reviews, clustering product categories, and using generative AI to summarize reviews into recommendation articles.

<br>

---

## MODEL TRACKING SPREADSHEET

| Model                            | Feature extraction         | Training model                    | Accuracy | Test set | Notes                                                                                     |
|-----------------------------------|-----------------------------|------------------------------------|----------|----------|---------------------------------------------------------------------------------------------|
| 1. Sentiment Analysis            | Hugging Face tokenizer      | `distilbert-base-uncased` (fine-tuned) | 90.8%   | 3,295 reviews | Macro F1 0.848; weakest on the minority `neutral` class (F1 0.717), strongest on `positive` (F1 0.957) |
| 2. Product Category Clustering   | TF‑IDF (word n-grams)       | KMeans + rule-based mapping        | n/a (unsupervised) | — | Hybrid approach: KMeans clusters on cleaned product names, refined with `primaryCategories` into 7 final meta-categories |
| 3. Product Review Summarization  | Prompted generation         | `meta-llama/Llama-3.2-3B-Instruct` | n/a (generative) | — | Evaluated qualitatively (readability/faithfulness of generated summaries), not via classification accuracy |
| 4. Product Review Summarization                                |    Hugging Face tokenizer                           | `distilbert-base-uncased`                                     |          |          |       the new transformers did not provide the summarization task                                                                                        | 
| 5.                                |                              |                                     |          |          |                                                                                               |
| 6.                                |                              |                                     |          |          |                                                                                               |

<br>

---

## Deployment

### Application hosted on Vercel

<center>
<table>
  <td><img src="../00_Archive/Images/SentimentReviewLAB_Negative.jpg" height="420" alt="SentimentReviewLAB_Negative"></td>
  <td> &nbsp &nbsp </td>
  <td><img src="../00_Archive/Images/SentimentReviewLAB_Neutral.jpg" height="420" alt="SentimentReviewLAB_Neutral"></td>
</table>
</center>

---

## Live Demo

🌐 **Vercel Hosting API and Application**

:white_check_mark: Application Frontend: 
https://p3nlp-customers-reviews.vercel.app/

:white_check_mark: API Health endpoint: 
https://p3nlp-customers-reviews.vercel.app/api/health

:white_check_mark: API Prediction endpoint: 
https://p3nlp-customers-reviews.vercel.app/api/predict

<br>

---

## Problem Statement

With thousands of reviews available across multiple platforms, manually analyzing them is inefficient. This project seeks to automate the process using NLP models to extract insights and provide users with valuable product recommendations.

<br>

---

## Datasets

**Primary Dataset - Consumer Reviews of Amazon Products** :
- https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products/data
- This is a list of over 34,000 consumer reviews for Amazon products like the Kindle, Fire TV Stick, and more provided by Datafiniti's Product Database. The dataset includes basic product information, rating, review text, and more for each product.

     - File: `original_kaggle_data_download_archive.zip`
     - containing 3 csv files, with data to be to be merged

<br>

---

## Main Tasks

### 1. Build a model for Sentiment Analysis

- **Goal**: Classify the textual content of a review as **positive**, **neutral**, or **negative**.
- **Labeling**: Star ratings were mapped to sentiment classes (1–2 ★ → negative, 3 ★ → neutral, 4–5 ★ → positive). This mapping produced a heavily imbalanced dataset (~92% positive, ~4% neutral, ~4% negative), so class imbalance was accounted for during training and evaluation (macro F1 alongside accuracy).
- **Approach**: Fine-tuned `distilbert-base-uncased` from Hugging Face Transformers on the cleaned reviews, using the tokenizer to convert review text into padded input token sequences.
- **Result**: 90.8% accuracy / 0.848 macro F1 on the held-out test set (3,295 reviews). See the [Model Tracking Spreadsheet](#model-tracking-spreadsheet) above for the full breakdown by class.

<br>

---

### 2. Categorization with K-Means Clustering

- **Goal**: Group the catalog's 106 unique cleaned product names into a small set of business-friendly categories, without relying on manual labeling.
- **Approach**: A hybrid pipeline — TF-IDF vectorization + K-Means clustering on the cleaned product names, with the resulting clusters interpreted and refined using the (partially available but reliable) `primaryCategories` field, then mapped with explicit rules into final meta-categories. This keeps the data-driven discovery of clustering while producing labels that are consistent and easy to explain in a report.

**7 Final Meta-Categories:**

| cluster      | cluster_name       | Number of Products       |
|--------------|--------------------|--------------------------|
|        0. 	|E-Readers	 |       849          |
|        1.	|Others	         |       18360        |
|        2.	|Tablets for Kids|	5366            |
|        3.	|Chargers & Adapters|	354             |
|        4.	|Tablets with Alexa|	4628            |
|        5.	|Tablets	 |     18641            |
|        6.	|Smart Assistants|	4786            |



See [`03_Product_Category_Clustering/3_prodcat.ipynb`](03_Product_Category_Clustering/3_prodcat.ipynb) for the full workflow diagram and methodology writeup.

<br>

---

### 3. Summarization

- **Goal**: Turn raw customer reviews into short, readable recommendation-style write-ups per product category, so a shopper (or a store manager) can understand consensus opinion without reading hundreds of reviews.
- **Approach**: Prompted `meta-llama/Llama-3.2-3B-Instruct` with the reviews for each category (grouped by the 7 meta-categories from step 2) to generate, for each category: the top products with review counts and average ratings, what reviewers like about the top products ("what sets it apart"), the most common complaints, and a "worst product in the category" callout with the reason to avoid it.
- **Output**: See [`04_Product_Review_Summarization/category_summaries.md`](04_Product_Review_Summarization/category_summaries.md) for the generated summaries.


<br>


---



## Pipeline

```
Raw Kaggle exports (3 CSVs, 67,992 rows)
        │
        ▼
Merge + deduplicate  ───────────────►  59,743 unique reviews, 89 product IDs
        │
        ▼
Clean corrupted `name` field / treat `id` as unreliable
        │
        ▼
clean_reviews.csv  (single source of truth for all downstream tasks)
        │
        ├──────────────────────┬───────────────────────────┐
        ▼                      ▼                            ▼
 Sentiment Analysis     Product Category Clustering    (feeds into) Review Summarization
 (DistilBERT fine-tune) (TF-IDF + K-Means + rules)      (Llama-3.2-3B-Instruct, grouped by category)
        │                      │                            │
        └──────────────────────┴────────────────────────────┘
                                ▼
                    05_Deployment — FastAPI + ONNX Runtime
                    served on Vercel (sentiment prediction API)
```

<br>

--- 

### Text Preprocessing

- Merged the 3 raw Datafiniti export files and removed duplicate reviews (same product + review text + rating).
- Fixed a data quality issue where the `name` field concatenated two unrelated product names — kept only the first product name segment.
- Treated `id` as unreliable (the same `id` was found attached to unrelated products in ~11% of reviews) and used the cleaned product name as the trusted identifier instead.
- For clustering: lowercased text, removed punctuation and stopwords, applied stemming/lemmatization to the cleaned product names.
- For the transformer model: tokenized review text with the Hugging Face tokenizer matching `distilbert-base-uncased`, encoded into vocabulary IDs, and padded sequences to a uniform length.


<br>


### Feature Engineering

- **Sentiment labels**: star ratings mapped to 3 classes — 1–2 ★ → negative, 3 ★ → neutral, 4–5 ★ → positive.
- **Sentiment model input**: transformer token embeddings from the DistilBERT tokenizer (no manual feature engineering — the model learns representations directly from text).
- **Clustering features**: TF-IDF vectors (with n-grams) computed over the cleaned, deduplicated product names.

---

# Results

- **Sentiment Analysis** — 90.8% accuracy / 0.848 macro F1 on a held-out test set of 3,295 reviews. The model performs strongly on the majority `positive` class (F1 0.957) and reasonably on `negative` (F1 0.868), but is noticeably weaker on the minority `neutral` class (F1 0.717) — expected given `neutral` made up only ~4% of the training data.
- **Product Category Clustering** — 106 unique cleaned product names consolidated into 7 business-oriented meta-categories: Tablets (29,553 reviews), Accessories & Chargers (11,126), Smart Home & Speakers (7,665), E-Readers (4,631), and Non-Electronics (9 — too few reviews to draw firm conclusions on its own).
- **Review Summarization** — Category-level write-ups generated for each meta-category, highlighting top-rated products, recurring praise, common complaints, and the weakest product per category. See [`04_Product_Review_Summarization/category_summaries.md`](04_Product_Review_Summarization/category_summaries.md) for full examples.
- **Deployment** — The sentiment model was exported to ONNX and deployed as a FastAPI serverless function on Vercel, with the model weights hosted on the Hugging Face Hub and downloaded at cold start.


<br>


---

# Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/wtraquinas/project-3-brief-NLP-automated-customers-reviews.git
cd project-3-brief-NLP-automated-customers-reviews
```

<br>


### 2. Notebooks (data prep, modeling, clustering, summarization)

The core NLP work lives in Jupyter notebooks and was run on Google Colab / a local Jupyter environment. Install the core dependencies:

```bash
pip install pandas numpy scikit-learn transformers datasets accelerate evaluate torch
```

Then run the notebooks in order:

1. `01_DataPrep/dataprep2.ipynb` — merges the raw Kaggle CSVs, cleans, and outputs `clean_reviews.csv`
2. `02_Sentiment_Analysis/2_sentiment.ipynb` — fine-tunes `distilbert-base-uncased` on `clean_reviews.csv`
3. `03_Product_Category_Clustering/3_prodcat.ipynb` — TF-IDF + K-Means clustering into 7 meta-categories
4. `04_Product_Review_Summarization/4_summarization.ipynb` — prompts `meta-llama/Llama-3.2-3B-Instruct` to generate category summaries

The raw data is provided as `00_Data/original_kaggle_data_download_archive.zip` (containing the 3 source CSVs), so no separate Kaggle download is required.


<br>


### 3. Deployment app (sentiment prediction API)

```bash
cd "05_Deployment/Review Sentiment Lab v2"
pip install -r requirements.txt
export HF_MODEL_REPO=<your-hf-username>/<sentiment-model-onnx-repo>
uvicorn api.predict:app --reload --port 8000
```

Then test it locally:

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great battery life"}'
```

Deploy with `vercel --prod` once the Hugging Face Hub model repo is set up (see [`05_Deployment/Review Sentiment Lab v2/README.md`](05_Deployment/Review%20Sentiment%20Lab%20v2/README.md) for the full walkthrough, including pushing the model to the Hub).

<br>


---

# Project Structure

```
project-3-brief-NLP-automated-customers-reviews/
├── 00_Archive/                        # earlier notebook versions and working notes
├── 00_Data/
│   ├── original_kaggle_data_download_archive.zip   # 3 raw Datafiniti/Amazon CSVs
│   └── dataset_overview_summary.md    # data quality findings & merge summary
├── 01_DataPrep/
│   ├── dataprep2.ipynb                # merge, dedup, clean -> clean_reviews.csv
│   └── clean_reviews.csv              # cleaned dataset used by all downstream tasks
├── 02_Sentiment_Analysis/
│   ├── 2_sentiment.ipynb              # fine-tune distilbert-base-uncased
│   ├── 5_deploymodel_2_sentiment.ipynb# export/prep model for deployment (ONNX)
│   └── models/                        # (model weights hosted on Hugging Face Hub, not committed)
├── 03_Product_Category_Clustering/
│   ├── 3_prodcat.ipynb                # TF-IDF + K-Means clustering
│   ├── 3_prodcat-manualcat.ipynb      # rule-based category refinement
│   └── README.md                      # clustering methodology write-up
├── 04_Product_Review_Summarization/
│   ├── 4_summarization.ipynb          # Llama-3.2-3B-Instruct prompting
│   └── category_summaries.md          # generated per-category review summaries
├── 05_Deployment/
│   ├── Review Sentiment Lab v2/       # deployed FastAPI + ONNX sentiment API (Vercel)
│   │   ├── api/predict.py
│   │   ├── public/                    # static frontend (HTML/CSS/JS)
│   │   ├── push_to_hub.py             # pushes ONNX model to Hugging Face Hub
│   │   ├── requirements.txt
│   │   └── vercel.json
│   └── live review aggregator/        # planning doc for a fuller React/FastAPI aggregator app
└── README.md
```

<br>


---

# Tech Stack

**Data & Modeling**
- Python, Jupyter/Colab notebooks
- pandas, numpy — data merging and cleaning
- scikit-learn — TF-IDF vectorization, K-Means clustering, evaluation metrics
- Hugging Face `transformers`, `datasets`, `evaluate`, PyTorch — fine-tuning `distilbert-base-uncased` for sentiment classification
- `meta-llama/Llama-3.2-3B-Instruct` — prompted for category review summarization

**Deployment**
- FastAPI — serverless prediction API
- ONNX Runtime — fast CPU inference of the exported sentiment model
- Hugging Face Hub — model weight hosting, downloaded at cold start
- Vercel — static frontend hosting + serverless functions
- HTML / CSS / JavaScript — deployed frontend (`public/`)


<br>


---

# Future Improvements

- Fill in the remaining rows of the [Model Tracking Spreadsheet](#model-tracking-spreadsheet) as additional models/approaches are tried (e.g. comparing `bert-base-uncased` or `roberta-base` against the current DistilBERT baseline).
- Improve minority-class performance for the sentiment model, e.g. targeted oversampling, class-weighted loss, or a larger neutral-review sample.
- Expand the deployment beyond a single sentiment endpoint into a full **Live Review Aggregator**: a React frontend backed by FastAPI, with a database (SQLite → PostgreSQL) storing submitted reviews, category dashboards, cached category summaries, and charts (rating distribution, sentiment breakdown) — see [`05_Deployment/live review aggregator/README.md`](05_Deployment/live%20review%20aggregator/README.md) for the full roadmap.
- Polish the current sentiment API frontend: sentiment color badges, a confidence bar, example-review buttons, dark mode, response-time indicator, and interactive API docs.
- Automate the pipeline end-to-end (new review in → sentiment + category + rolling summary out) instead of running notebooks manually.


<br>


---

# Authors

**Antonio Traquinas** - https://github.com/wtraquinas/

and 

**David Tayebwa** - https://github.com/kerondavid-debug/

AI Engineering | Machine Learning | NLP


<br>


---

## Acknowledgements

Special Thanks to Luis Junco and Tejal Bhatti, for the inestimable assistance.

This project was developed as part of the **IronHack AI Engineering Bootcamp**, demonstrating the complete machine learning workflow from data preprocessing to deployment of an interactive web application.

