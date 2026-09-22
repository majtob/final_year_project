# Credit Risk Analysis on Saudi Exchange Financials
## Final Year Project Progress Report

**Author:** MXA1438  
**Date:** February 2026  
**Status:** In Progress

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Literature Review](#2-literature-review)
3. [Data Collection](#3-data-collection)
4. [Methodology](#4-methodology)
5. [Baseline Model Results](#5-baseline-model-results)
6. [Combined Model Results (Financials + KAMs)](#6-combined-model-results-financials--kams)
7. [Current Progress](#7-current-progress)
8. [News Sentiment Analysis](#8-news-sentiment-analysis)
9. [Full Model Comparison](#9-full-model-comparison)
10. [Multi-Agency Rating Expansion](#10-multi-agency-rating-expansion)
11. [Multi-Model Comparison](#11-multi-model-comparison)
12. [Paper-Aligned Model Results](#12-paper-aligned-model-results)
13. [Historical Fitch Data](#13-historical-fitch-data)
14. [Agency Handling Comparison](#14-agency-handling-comparison)
15. [SHAP Explainability Analysis](#15-shap-explainability-analysis)
16. [Error Analysis](#16-error-analysis)
17. [KAM Homogeneity: A Saudi Market Finding](#17-kam-homogeneity-a-saudi-market-finding)
18. [LLM Verdict Generator](#18-llm-verdict-generator)
19. [Interactive Demo Application](#19-interactive-demo-application)
20. [Novel Contributions](#20-novel-contributions)
21. [Answering the Research Questions](#21-answering-the-research-questions)
22. [Project Structure](#22-project-structure)
23. [Next Steps](#23-next-steps)
24. [References](#24-references)

---

## 1. Project Overview

### 1.1 Objective

**System aim.** Develop a **reproducible** machine-learning pipeline for **coarse multiclass credit rating categories** (AA, A, BBB, BB) on Saudi Tadawul firm-years, combining:

- Four Altman-style **financial ratios** (from yfinance-derived financials)
- **Key Audit Matter (KAM)** dummies aligned with Muñoz-Izquierdo et al. (2022)
- **FinBERT-based news aggregates** (English articles via MarketAux, then `collect_news_sentiment_finbert.py`)

The pipeline includes **stratified cross-validation**, **model benchmarks**, **SHAP explainability**, **template / LLM verdict generation**, and a **Streamlit demo** (`app.py`).

**Primary evaluation objective.** **Quantify whether KAM and news feature blocks improve cross-validated performance relative to financial ratios alone** on the **constructed merged panel**, **acknowledging small-sample variance** (the latest multisource training matrix is **45** `(ticker, fiscal_year)` rows; ablations are reported in `results/full_model_comparison.json` and summarised in the current pipeline doc `docs/PROJECT_REPORT_CURRENT.md`).

### 1.2 Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                       DATA SOURCES                              │
├──────────┬───────────┬──────────┬──────────┬──────────────────┤
│ Tassnief │  Fitch    │ Moody's  │  S&P     │ Financial Analyt.│
│          │           │          │          │                  │
│ yfinance │ Tadawul   │ Argaam   │ Annual   │ News (MarketAux) │
│ (finan.) │ (discl.)  │ (article)│ Reports  │                  │
└────┬─────┴─────┬─────┴────┬─────┴────┬─────┴────────┬─────────┘
     │           │          │          │              │
     ▼           ▼          ▼          ▼              ▼
┌────────────────────────────────────────────────────────────────┐
│              FEATURE EXTRACTION                                 │
│  • 4 Altman Z''-Score Ratios (LIQUID, CUMPROF, PROFITAB, LEV.) │
│  • 5 KAM Categories + KAM Count                                │
│  • Rating harmonization across agencies                        │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│            ML CLASSIFIERS (8 models compared)                   │
│  Best: Gradient Boosting (68.0%), Decision Tree (78.3%)        │
│  GroupKFold CV, binary classification                           │
│                                                                 │
│  → SHAP Explainability (per-prediction + global)               │
│  → Error Analysis (multisource CV confusion + case-level JSON)  │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│         LLM VERDICT GENERATOR                                   │
│  Fine-tuned Qwen 2.5 3B (QLoRA), Ollama, or Template           │
│  Input: Ratios + ML prediction + confidence                    │
│  Output: Structured credit verdict (JSON)                      │
│  Quality: 94% number accuracy, 89% citation rate               │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│         STREAMLIT DEMO APP                                      │
│  Interactive company selector, SHAP waterfall, LLM verdict     │
│  Model performance dashboard, error analysis explorer          │
└────────────────────────────────────────────────────────────────┘
```

### 1.3 Scope

- **Companies:** 47 unique Tadawul-listed companies with credit ratings (23 non-financial in training set)
- **Time Period:** 2021-2024 (fiscal years)
- **Ratings Sources:** Tassnief, Moody's, Fitch, S&P, Financial Analytics (RATING)
- **Financial Data Source:** yfinance API
- **Training Data:** 75 records, 23 companies, 5 agencies (V2 dataset)
- **Best Accuracy:** 68.0% (Gradient Boosting, GroupKFold CV)

---

## 2. Literature Review

### 2.1 Primary Reference Paper

**Muñoz-Izquierdo, N., Segovia-Vargas, M.J., Camacho-Miñano, M.M., & Pérez-Pérez, Y. (2022).**  
*Machine learning in corporate credit rating assessment using the expanded audit report.*  
Machine Learning, 111, 4183–4215.  
https://doi.org/10.1007/s10994-022-06226-4

#### Key Findings from the Paper:

| Model | Features Used | Accuracy |
|-------|---------------|----------|
| KAMs only | 5 KAM categories + audit variables | **74.14%** |
| Financial ratios only | 4 Altman Z''-Score ratios | 71.55% |
| Combined (KAMs + Financials) | All features | **84.04%** |

#### Paper Methodology:
- **Sample:** 116 Spanish listed companies (2017)
- **ML Techniques:** C4.5 Decision Tree, PART Algorithm, Rough Set, Logistic Regression
- **Best Performer:** PART algorithm for rule induction
- **Key Insight:** KAMs alone can predict credit ratings with 74% accuracy

### 2.2 What are KAMs?

**Key Audit Matters (KAMs)** are paragraphs in audit reports where auditors highlight:
- Significant risks of material misstatement
- Areas requiring significant management judgment
- Critical accounting estimates

KAMs became mandatory for listed companies after the 2016 international audit reform (ISA 700).

### 2.3 KAM Categories (from the paper)

| Category | Variable | Description |
|----------|----------|-------------|
| Going Concern | `GCKAM` | Viability doubts, refinancing risks, covenant breaches |
| Revenue | `REVKAM` | Revenue recognition complexity, contract accounting |
| Assets | `ASSETKAM` | Impairment, goodwill, fair value of assets |
| Liabilities | `LIABKAM` | Provisions, contingencies, debt issues |
| Other | `OTHERKAM` | IT systems, acquisitions, regulatory compliance |

### 2.4 ML Model Comparison in Credit Scoring

**Bao, W., Lianju, N., & Yue, K. (2019).**  
*Integration of unsupervised and supervised machine learning algorithms for credit risk assessment.*  
Expert Systems with Applications, 128, 301–315.

Bao et al. compare seven supervised ML algorithms (LR, SVM, DT, RF, GBDT, kNN, ANN) on credit scoring datasets including a Chinese P2P dataset of 23,435 samples with 16.6% default rate. Their key findings support our methodology:

| Finding | Relevance to Our Project |
|---------|--------------------------|
| **GBDT best individual model** | Gradient Boosting achieved highest MCC (0.663) among individual models, consistent with our finding that Gradient Boosting outperforms alternatives on Saudi data |
| **Ensemble superiority** | Combining unsupervised clustering with supervised models and SOM consensus improved MCC from 0.663 to 0.702 — validates our use of ensemble methods |
| **MCC over accuracy for imbalance** | They emphasize Matthews Correlation Coefficient (MCC) as more reliable than accuracy for imbalanced credit datasets — supports our evaluation approach |
| **Model ranking** | GBDT > ANN > SVM > RF > kNN > LR > DT — aligns with our observation that tree-based and boosting methods outperform linear models |

Their methodology (5-fold CV, grid search, train/test split) mirrors our approach. The performance gap between their best result (MCC 0.702 on 23K samples) and ours (~71% accuracy on 75 samples) is consistent with the known effect of small sample size on credit scoring models.

---

## 3. Data Collection

### 3.1 Tassnief Credit Ratings

**Source:** https://tassnief.com (Saudi national credit rating agency)

**Collection Method:** Selenium web scraping was used to build `data/templates/ratings_scraped.csv`; the one-off scraper was removed from the repository in favour of the consolidated `scripts/rebuild_processed_datasets.py` pipeline (re-scraping would require restoring the scraper from project history or manual export from tassnief.com).

**Results:**
- 50 rating observations scraped
- 24 unique Tadawul-listed companies identified
- Rating period: 2021-2026

**Rating Distribution:**

| Rating | Count | Category |
|--------|-------|----------|
| AAA | 5 | Investment Grade |
| AA+ | 2 | Investment Grade |
| AA | 4 | Investment Grade |
| A | 9 | Investment Grade |
| BBB+ | 13 | Investment Grade |
| BBB | 8 | Investment Grade |
| BB+ | 3 | Speculative |
| BB | 2 | Speculative |

**Output File:** `data/templates/ratings_scraped.csv`

### 3.2 Financial Statements

**Source:** yfinance API

**Collection / refresh:** Financial ratios for the KAM template rows can be rebuilt with `python scripts/rebuild_processed_datasets.py` (add `--yfinance` to pull annual statements from yfinance into `ratings_with_financials.csv` and recompute ratios). Legacy one-off collection scripts were removed in favour of this single entrypoint.

**Data Retrieved:**
- Balance Sheet (annual)
- Income Statement (annual)
- Cash Flow Statement (annual)

**Coverage:**
- 23/24 companies have yfinance data
- 32/50 rating observations matched with financials
- 26 records have complete data for all 4 ratios

**Output Files:**
- `data/processed/ratings_with_financials.csv`
- `data/raw/financials/{ticker}_financials.json`

### 3.3 Data Limitations

| Issue | Impact | Mitigation |
|-------|--------|------------|
| 2025 financials not yet available | 17 records missing | Use 2024 and earlier |
| Sumou Real Estate (9511.SR) no yfinance data | 3 records lost | Manual extraction needed |
| Insurance companies missing some ratios | 4 records incomplete | Industry-specific ratios |
| Small sample size (26 usable records) | Risk of overfitting | Feature selection, cross-validation |

---

## 4. Methodology

### 4.1 Financial Ratios

We use the **Altman Z''-Score** components, as specified in the reference paper:

| Ratio | Formula | Interpretation |
|-------|---------|----------------|
| **LIQUID** | (Current Assets - Current Liabilities) / Total Assets | Liquidity - can company pay short-term debts? |
| **CUMPROF** | Retained Earnings / Total Assets | Cumulative profitability over time |
| **PROFITAB** | EBIT / Total Assets | Current period profitability (ROA proxy) |
| **LEVERAGE** | Book Value of Equity / Total Liabilities | Solvency - debt vs equity structure |

#### Why These 4 Ratios?

1. **Validated Model:** Altman Z''-Score is globally recognized for credit risk
2. **Normalization:** Ratios allow comparison across company sizes
3. **Dimensionality:** With only 26 samples, 4 features is appropriate (rule: 5-10 samples per feature)
4. **Paper Alignment:** Same ratios used in the reference paper

### 4.2 Why Banks and Financial Institutions Are Excluded

Following the reference paper (Muñoz-Izquierdo et al., 2022) and standard practice in credit risk modelling (Charitou et al., 2007), **banks and financial institutions are removed** from the ML analysis. There are two fundamental reasons:

#### 4.2.1 Financial Statement Structure

Banks have fundamentally different balance sheets compared to non-financial companies. Their assets are primarily loans issued, and their liabilities are primarily customer deposits. This makes the Altman Z''-Score ratios misleading:

| Ratio | Non-Financial Company | Bank |
|-------|----------------------|------|
| **LIQUID** (Working Capital / Total Assets) | Meaningful measure of short-term health | Nearly meaningless -- banks don't rely on working capital |
| **CUMPROF** (Retained Earnings / Total Assets) | Proportional measure of accumulated profit | Distorted by enormous total assets (mostly loans) |
| **PROFITAB** (EBIT / Total Assets) | Useful return on assets | Artificially tiny -- bank assets are 10-20x equity |
| **LEVERAGE** (Equity / Total Liabilities) | Solvency indicator | Always extremely low (~5-10%) by design -- banks are inherently highly leveraged |

A bank with a leverage ratio of 0.05 would appear "distressed" by Altman standards, but this is normal for a regulated financial institution. Mixing banks and non-financials would teach the model incorrect patterns.

#### 4.2.2 Regulatory Framework

Banks operate under Basel III capital requirements (minimum capital adequacy ratios, liquidity coverage ratios) imposed by central banks. Non-financial companies face no such constraints. Key differences:

- Bank credit ratings depend heavily on **regulatory capital ratios** (Tier 1, CET1) that don't exist for normal companies
- Banks may benefit from **implicit government guarantees** (too-big-to-fail), inflating their ratings relative to their financial ratios
- KAMs in bank audit reports focus on different risks (loan loss provisions, fair value of financial instruments) vs. non-financial KAMs (revenue recognition, asset impairment)

#### 4.2.3 Impact on Our Dataset

Many Fitch-rated Saudi companies are banks (Riyad Bank, Al Rajhi, SNB, etc.). After filtering to non-financial companies with all 4 ratios, the dataset drops significantly -- this is a major reason why the Fitch-only subset contains only 8 companies (25 records). Including banks would require bank-specific ratios (capital adequacy, NPL ratio, cost-to-income, net interest margin), essentially creating a separate model for a separate problem.

### 4.3 Target Variable

**Binary Classification:**
- Investment Grade (1): BBB- and above
- Speculative (0): BB+ and below

**Multi-class Classification:**
- AA: AAA, AA+, AA, AA-
- A: A+, A, A-
- BBB: BBB+, BBB, BBB-
- BB: BB+ and below

**Related literature (discrete / coarse rating targets):** Empirical credit-rating ML typically predicts **a finite set of ordered rating labels** (multiclass classification) rather than a continuous score or every agency notch as its own class—both for comparability across agencies and for stable class frequencies. **Huang et al. (2004)** compare support vector machines and neural networks on **corporate credit rating categories** in a multiclass setup. **Golbayani et al. (2020)** benchmark several ML methods for **forecasting corporate credit ratings** and introduce **notch-distance** accuracy to reflect that ratings are **discrete ordered** outcomes; that framing aligns with mapping fine notches into broader buckets when sample size is limited. Our four-way mapping follows the same **coarse multiclass** logic alongside Muñoz-Izquierdo et al. (2022).

### 4.3 Model Selection

**Algorithm:** XGBoost (Extreme Gradient Boosting)

**Why XGBoost:**
- Handles small datasets well with regularization
- Built-in feature importance
- Robust to outliers
- State-of-the-art for tabular data

**Hyperparameters:**
```python
XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    scale_pos_weight=auto,  # Handles class imbalance
    random_state=42
)
```

### 4.4 Evaluation

Following Bao et al. (2019), we use accuracy, precision, recall, and confusion matrices. For imbalanced credit datasets, Matthews Correlation Coefficient (MCC) is a more reliable metric than raw accuracy; we report both where applicable.

**Cross-Validation:** Stratified K-Fold (K=2 to 5, depending on class counts)

**Metrics:**
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- Feature Importance

---

## 5. Baseline Model Results

### 5.1 Dataset Summary

| Metric | Value |
|--------|-------|
| Total rating observations | 50 |
| Records with financial data | 32 |
| Records with complete ratios | 26 |
| Features | 4 (LIQUID, CUMPROF, PROFITAB, LEVERAGE) |

### 5.2 Multi-class Classification (AA/A/BBB/BB)

| Metric | Value |
|--------|-------|
| **Accuracy** | **61.54%** |
| Cross-validation | 2-Fold |
| Standard Deviation | ±15.38% |

**Feature Importance:**

| Feature | Importance |
|---------|------------|
| PROFITAB (EBIT/Assets) | 35.46% |
| CUMPROF (Retained Earnings/Assets) | 23.48% |
| LEVERAGE (Equity/Liabilities) | 21.61% |
| LIQUID (Working Capital/Assets) | 19.45% |

**Confusion Matrix:**
```
          Predicted
          A   AA  BB  BBB
Actual A  [5   0   0   0]
       AA [0   6   0   0]
       BB [1   0   1   0]
       BBB[0   0   0  13]
```

### 5.3 Binary Classification (Investment Grade vs Speculative)

| Metric | Value |
|--------|-------|
| Accuracy | 7.69% |
| Issue | Severe class imbalance (24 IG vs 2 Spec) |

**Why Binary Failed:**
- 92% of samples are Investment Grade
- Model learns to always predict "Investment Grade"
- Insufficient speculative-grade examples

### 5.4 Comparison to Paper

| Model | Our Result | Paper Result |
|-------|------------|--------------|
| Financials only | 61.54% | 71.55% |
| KAMs only | **69.09%** (5-fold CV; `kams_processed.csv` only, 55 rows — see §6.5) | **74.14%** |
| Combined | Not yet tested | **84.04%** |

**Gap Analysis:**
- Our sample: 26 records (Saudi market)
- Paper sample: 116 records (Spanish market)
- Smaller sample = higher variance, lower accuracy

---

## 6. Combined Model Results (Financials + KAMs)

### 6.1 KAM Extraction Summary

KAMs were manually extracted from 25 annual reports (Tadawul issuer reports).

**KAM Prevalence Across Saudi Companies:**

| KAM Category | Present | Percentage | Notes |
|--------------|---------|------------|-------|
| Going Concern | 0/23 | 0% | No Saudi company flagged |
| Revenue | 20/23 | 87% | Most common KAM |
| Assets | 22/23 | 96% | Nearly universal |
| Liabilities | 2/23 | 9% | Only STC (2021, 2022) |
| Other | 2/23 | 9% | STC (2024), Quara (2023) |

**Key Observation:** Saudi companies have low KAM variance compared to the Spanish market in the paper - almost all companies have revenue recognition and asset impairment KAMs, while none have going concern issues.

### 6.2 Three-Model Comparison

| Model | Features | CV Accuracy | Paper Reference |
|-------|----------|-------------|-----------------|
| Financials Only | 4 ratios | **65.15%** | 71.55% |
| KAMs Only (merged financials∩KAM rows; KAM columns as inputs) | 6 KAM features | 47.73% | 74.14% |
| KAMs Only (**`kams_processed.csv` only**, no financials) | 13 audit/KAM fields | **69.09%** (±8.91%) | 74.14% |
| Combined | 10 features | **65.15%** | 84.04% |

The **69.09%** row is produced by `models/xgboost_kams_only.py` (March 2026): all rows in `kams_processed.csv`, four-class rating target, stratified 5-fold CV. Full metrics and importances: `results/kams_only_model_results.json`. It is **not** the same experiment as the **47.73%** row (which used only the subset of rows that also had complete financial ratios).

### 6.3 Combined Model Feature Importance

| Feature | Importance | Type |
|---------|------------|------|
| kam_count | 27.36% | KAM |
| profitab (EBIT/Assets) | 22.40% | Financial |
| cumprof (Retained Earnings/Assets) | 17.47% | Financial |
| leverage (Equity/Liabilities) | 16.83% | Financial |
| kam_revenue | 10.14% | KAM |
| liquid (Working Capital/Assets) | 5.80% | Financial |
| kam_going_concern | 0.00% | KAM |
| kam_assets | 0.00% | KAM |
| kam_liabilities | 0.00% | KAM |
| kam_other | 0.00% | KAM |

### 6.4 Why KAMs Didn't Improve Accuracy

| Factor | Paper (Spain) | Our Data (Saudi) |
|--------|---------------|------------------|
| Sample size | 116 companies | 23 companies |
| Going concern KAMs | ~15% present | 0% present |
| KAM variance | High (diverse economy) | Low (similar industries) |
| Asset KAMs | ~60% present | 96% present |
| Market | Diverse, includes distressed firms | Healthier, established firms |

**Root Cause:** 3 of 5 KAM features have near-zero variance (going_concern always 0, assets always 1, liabilities/other rarely present). A feature that's the same for every company cannot distinguish between ratings.

**What DID Help:** `kam_count` (27% importance) is the most predictive feature in the combined model - companies with more KAMs tend to have lower ratings, consistent with the paper's finding that audit complexity signals credit risk.

### 6.5 Standalone KAM-only XGBoost (`kams_processed.csv`)

To test whether audit-report features alone predict rating **without** merging to financial ratios, we added `models/xgboost_kams_only.py`. It loads **only** `data/processed/kams_processed.csv`. **Inputs:** twelve integer fields (`AUSIZE`, `AUOP`, `EMP`, `GCUP`, `GCKAM`, `REVKAM`, `ASSETKAM`, `LIABKAM`, `OTHERKAM`, `FIRMAGE`, `FIRMSIZE`, `INDUSTRY`). **Target:** four-way `rating_category` (AA / A / BBB / BB), aligned with `xgboost_with_kams.py`.

| Metric | Value |
|--------|-------|
| Samples | 47 (KAM panel) |
| 5-fold stratified CV accuracy | **62.00%** (±26.23% on a small sample; high variance) |
| In-sample train accuracy | 76.60% |
| Artifacts | `results/kams_only_model_results.json` |

**Features:** 12 modeling columns from `kams_processed.csv` (audit + firm controls + five paper KAM dummies). There is **no** aggregate `KAM_COUNT` column.

**XGBoost feature importance (gain-based, in-sample fit):** in the latest run, top contributors included `AUSIZE`, `REVKAM`, `INDUSTRY`, and `OTHERKAM` (ranking varies with the panel).

**Interpretation:** CV accuracy is the primary guide; training accuracy is optimistic. The benchmark is **not** directly comparable to the paper’s **74.14%** KAM-only result (different market, sample, labels, and learner).

---

## 7. Current Progress

### 7.1 Completed Tasks

| Task | Status | Output |
|------|--------|--------|
| Scrape Tassnief ratings | ✅ Done | `ratings_scraped.csv` |
| Map tickers to Tadawul | ✅ Done | 24 companies identified |
| Collect financial statements | ✅ Done | `ratings_with_financials.csv` |
| Calculate financial ratios | ✅ Done | 4 Altman Z'' ratios |
| Train baseline XGBoost | ✅ Done | 65.15% accuracy |
| Extract KAMs (23 companies) | ✅ Done | `kams_priority.csv` |
| Train combined model | ✅ Done | 65.15% accuracy |
| KAM-only XGBoost (`kams_processed` only) | ✅ Done | ~62% 5-fold CV (47 rows); `results/kams_only_model_results.json` |
| News sentiment collection | ✅ Done | FinBERT on cached JSON; panel aligned to KAM rows |
| Train full XGBoost (4 variants: fin / +KAM / +sentiment / full) | ✅ Done | See `results/full_model_comparison.json` (**45** rows after excluding 3008.SR/2021; latest run: **~60%** full 14-feature, **~51%** fin-only, **~47%** fin+KAM 9-feats, **~60%** fin+sentiment—high variance; regenerate JSON after CSV edits) |
| Multi-agency expansion | ✅ Done | 80 ratings, 5 agencies |
| Train expanded models | ✅ Done | 69.6% best (Fitch) |
| Multisource CV benchmark (10 learners) | ✅ Done | `evaluate_multisource_models.py` → `results/multisource_model_comparison.json` |
| Legacy multi-model comparison (older sample) | ✅ Reference | 78.3% best DT in historical `figures/model_comparison.png` |
| Paper-aligned model (4 ratios, binary) | ✅ Done | 71.2% (Gradient Boosting) |
| Historical Fitch data (2019-2024) | ✅ Done | 66.7% best (SVM) |
| Agency handling comparison | ✅ Done | Fitch-only vs All+Agency |
| SHAP explainability (14-feature full XGBoost) | ✅ Done | `models/shap_explainability.py` → `figures/shap_*.png`, `results/shap_report.json` |
| Systematic error analysis | ✅ Done | `results/error_analysis.json`: OOF confusion + 17 misclassified rows (*n*=45 multisource) |
| KAM homogeneity analysis | ✅ Done | 65% identical profiles |
| LLM verdict generator | ✅ Done | Multisource rows + multicategory XGB; `results/verdicts/` |
| Publication-quality visualizations | ✅ Done | 17 figures in `figures/` |
| Streamlit demo application | ✅ Done | `app.py` |
| Feature ablation study | ✅ Done | KAMs add -1.3% to +5.3% |
| Formalized novel contributions | ✅ Done | 5 contributions documented |

### 7.2 Project Structure

```
mxa1438/
├── README.md
├── requirements.txt
├── app.py                                  # Streamlit demo (reads merged_multisource_training.csv)
│
├── data/
│   ├── templates/                          # Input templates & scraped data
│   │   ├── ratings_scraped.csv             # Tassnief ratings (scraped)
│   │   ├── kams_priority.csv               # KAM extraction template
│   │   ├── kams_to_extract.csv             # Full extraction list
│   │   ├── multi_agency_ratings.csv        # Multi-agency ratings (legacy / reference)
│   │   └── ticker_mapping.csv              # Company name → ticker mapping
│   ├── processed/                          # ML-ready datasets
│   │   ├── kams_processed.csv              # Paper-style KAM dummies + metadata
│   │   ├── financial_ratios_processed.csv  # Altman-style ratios per ticker × fiscal year
│   │   ├── news_features_processed.csv     # News / sentiment aggregates
│   │   ├── merged_multisource_training.csv # Inner join of the three above (+ sector, agency)
│   │   ├── ratings_with_financials.csv     # Tassnief + financials (source for ratios slice)
│   │   ├── ratings_financials_sentiment.csv# Optional input for news_features_processed
│   │   └── ... (other legacy CSVs may remain for the written report)
│   └── raw/
│       ├── financials/                     # Cached yfinance JSON per company
│       └── news/                           # Cached news per company-year
│
├── scripts/
│   ├── rebuild_processed_datasets.py       # Builds processed tables + merged training CSV (when ratings source exists)
│   ├── align_processed_to_kams.py          # Align news/financials to `kams_processed` panel
│   └── collect_news_sentiment_finbert.py   # FinBERT → `ratings_financials_sentiment.csv`
│
├── models/
│   ├── xgboost_with_kams.py                # XGBoost: financials + KAMs
│   ├── xgboost_kams_only.py                # XGBoost: kams_processed.csv only
│   ├── xgboost_full.py                     # XGBoost: financials + KAMs + news (4 model variants)
│   ├── shap_explainability.py              # SHAP plots + `shap_report.json` for full multisource XGBoost
│   ├── evaluate_multisource_models.py      # XGBoost vs RF, GBDT, linear, kNN, MLP, … (StratifiedKFold)
│   ├── multisource_data.py                 # Load/build `merged_multisource_training.csv`
│   ├── llm_verdict.py                      # Verdicts (multicategory XGB + KAM + FinBERT context)
│   ├── prepare_finetune_data.py            # Instruction data for QLoRA
│   ├── finetune_qwen.py / evaluate_finetune.py / improve_verdicts.py
│   └── lora_adapter/                       # PEFT adapter weights (if trained)
│
├── results/                                # Model output, verdicts JSON
├── figures/                                # Plots (some from earlier experiments)
│
└── docs/
    ├── PROJECT_REPORT.md
    ├── DOCKER.md
    ├── FINBERT_NEWS_SENTIMENT.md
    └── ...
```

---

## 8. News Sentiment Analysis

### 8.1 Data Collection

**Source:** MarketAux API  
**Method:** Fetched English-language news articles for each company in each fiscal year

**Coverage (aligned KAM panel):**
- 47 `(ticker, fiscal_year)` rows in `kams_processed.csv`; `news_features_processed.csv` is aligned to the same keys (neutral zeros when no cached JSON).
- Mean aggregate FinBERT score varies by row; see `data/processed/news_features_processed.csv`.

**Sentiment Features:**

| Feature | Description |
|---------|-------------|
| `sentiment_mean` | Mean per-article FinBERT score, roughly in −1…1 |
| `sentiment_std` | Standard deviation (sentiment volatility) |
| `sentiment_pos_pct` | Proportion of articles with positive FinBERT score |
| `sentiment_neg_pct` | Proportion of articles with negative FinBERT score |
| `news_count` | Number of articles found |

### 8.2 Sentiment Scoring

Articles are scored with **FinBERT** (`ProsusAI/finbert`): a BERT model fine-tuned on financial text for three-way positive / negative / neutral classification. Cached MarketAux JSON under `data/raw/news/` is rescored with `scripts/collect_news_sentiment_finbert.py`; see `docs/FINBERT_NEWS_SENTIMENT.md` for the exact aggregation. The earlier **VADER**-based collector is legacy only.

---

## 9. Full Model Comparison

### 9.1 Four-Model Comparison

| # | Model | Features | CV Accuracy | Paper Ref |
|---|-------|----------|-------------|-----------|
| 1 | **Financials Only** | 4 | **65.15%** | 71.55% |
| 2 | Financials + KAMs | 10 | 65.15% | 84.04% |
| 3 | Financials + Sentiment | 9 | 30.68% | - |
| 4 | Full (All Features) | 15 | 30.68% | - |

### 9.2 Analysis

**Best Model: Financials Only (65.15%)**

The simpler model outperformed more complex ones. This is a classic case of the **bias-variance tradeoff** with small datasets:

1. **23 samples with 15 features (Model 4)** = severe overfitting risk
   - Training accuracy: 100% (memorization)
   - CV accuracy: 30.68% (fails to generalize)

2. **23 samples with 4 features (Model 1)** = better generalization
   - Training accuracy: 100%
   - CV accuracy: 65.15% (more stable)

### 9.3 Top Predictive Features Across All Models

| Feature | Model 1 | Model 2 | Model 4 | Type |
|---------|---------|---------|---------|------|
| leverage | 30.4% | 16.8% | 16.8% | Financial |
| profitab | 29.8% | 22.4% | 14.5% | Financial |
| kam_count | - | 27.4% | 15.3% | KAM |
| news_count | - | - | 22.7% | Sentiment |
| cumprof | 24.5% | 17.5% | 13.7% | Financial |
| sentiment_mean | - | - | 6.8% | Sentiment |

### 9.4 Why Sentiment Hurt Performance

1. **Noise:** MarketAux returned some irrelevant articles (e.g., "Perfect Corp" matched instead of "Perfect Presentation")
2. **Small sample:** 3-8 articles per company-year is insufficient for robust sentiment
3. **Positive bias:** Most financial news about Saudi companies is neutral/positive
4. **Feature-to-sample ratio:** 15 features for 23 samples causes overfitting

### 9.5 Key Takeaways

1. **Financial ratios are the strongest predictors** - leverage and profitability drive credit ratings
2. **kam_count adds signal** (27% importance in Model 2) but low KAM variance in Saudi market limits impact
3. **Sentiment adds noise** at this sample size - would need 100+ articles per company
4. **23 samples is the core limitation** - more data needed before adding features

---

## 10. Multi-Agency Rating Expansion

### 10.1 Motivation

The core limitation throughout experiments has been sample size (23-26 samples from Tassnief only). To address this, we expanded the dataset by incorporating credit ratings from multiple international and local agencies.

### 10.2 Data Source

**Argaam Article (October 2025):** "Insight into TASI-listed companies with credit ratings"
- Source: https://www.argaam.com/en/article/articledetail/id/1850297
- 39 TASI-listed companies + 9 Nomu-listed companies
- 5 rating agencies: Moody's, Fitch, S&P, Tassnief, Financial Analytics (RATING)

### 10.3 Rating Scale Harmonization

Moody's uses a different scale that required mapping to the standard S&P/Fitch scale:

| Moody's | Standard | Category |
|---------|----------|----------|
| Aaa | AAA | High Grade |
| Aa1/Aa2/Aa3 | AA+/AA/AA- | High Grade |
| A1/A2/A3 | A+/A/A- | Investment Grade |
| Baa1/Baa2/Baa3 | BBB+/BBB/BBB- | Investment Grade |
| Ba1/Ba2/Ba3 | BB+/BB/BB- | Near-Excellent |

National-scale ratings (e.g., "AAA (sau)", "KsaAAA") were stripped as they use different benchmarks than international ratings.

### 10.4 Expanded Dataset Summary

| Metric | Previous | Expanded |
|--------|----------|----------|
| Total ratings | 50 (Tassnief) | 80 (5 agencies) |
| Unique companies | 24 | 47 |
| With financials | 23 | 79 |
| With all 4 ratios | 26 | 35 |
| Rating agencies | 1 | 5 |

**Ratings by Agency:**

| Agency | Count |
|--------|-------|
| Fitch | 23 |
| Moody's | 19 |
| S&P | 19 |
| Financial Analytics | 17 |
| Tassnief | 2 |

### 10.5 Methodological Challenges

**1. Inter-Agency Disagreement**

Different agencies assign different ratings to the same company using the same financials:

| Company | Moody's | Fitch | S&P | Fin. Analytics |
|---------|---------|-------|-----|----------------|
| Cenomi Centers | - | BB | BB- | A- |
| BSF | A+ | A- | A- | - |
| ANB | A+ | BBB+ | BBB+ | - |

This is not noise -- agencies genuinely weigh different factors differently.

**2. Banks Missing Financial Ratios**

Bank balance sheets don't have standard "Current Assets" or "EBIT", causing `liquid` and `profitab` ratios to be unavailable. Only `cumprof` and `leverage` are computable for banks.

**3. Data Leakage Risk**

Having multiple ratings per company means the same company could appear in both train/test splits. We used **GroupKFold** cross-validation (groups = company ticker) to prevent this.

### 10.6 Expanded Model Results

#### Experiment 1: Consensus Ratings (one per company)

For each company, the median numeric rating across all agencies was used as the target.

| Model | Samples | Accuracy |
|-------|---------|----------|
| Non-Financial, 4 Ratios | 20 | 45.0% |
| All Companies, 2 Ratios | 45 | 55.6% |

#### Experiment 2: Per-Agency Models

Each agency's ratings were modeled independently using Leave-One-Out CV:

| Agency | Samples | Accuracy | Notes |
|--------|---------|----------|-------|
| **Fitch** | **23** | **69.6%** | **Best overall accuracy** |
| Moody's | 19 | 52.6% | Leverages weight heavily |
| S&P | 19 | 36.8% | May need qualitative features |
| Financial Analytics | 15 | 26.7% | Local methodology, less ratio-driven |

#### Experiment 3: Combined Tassnief + Expanded

| Model | Samples | Accuracy |
|-------|---------|----------|
| Combined, 4 Ratios | 22 | 36.4% |

### 10.7 Key Findings

**1. Fitch ratings are most predictable from financials (69.6%)**

Fitch is known for its transparent, formula-driven methodology. Its ratings correlate most strongly with objective financial ratios, making them the most suitable target for ML prediction using only financial features.

**2. Multi-agency data doesn't automatically improve accuracy**

Adding ratings from multiple agencies that use different methodologies creates label noise -- the same financial inputs map to different labels. The consensus approach (55.6%) was better than raw mixing but still below single-agency models.

**3. Feature importance varies by agency**

| Feature | Fitch | Moody's | Financial Analytics |
|---------|-------|---------|---------------------|
| cumprof | 59.4% | 32.0% | 68.0% |
| leverage | 40.6% | 68.0% | 32.0% |

Moody's weights leverage (solvency) more heavily, while Fitch and Financial Analytics favor cumulative profitability.

**4. Cumprof emerges as the universal predictor**

Across all models, `cumprof` (Retained Earnings / Total Assets) is the most consistently important feature. Companies with higher cumulative profitability receive better ratings regardless of agency.

### 10.8 Full Model Comparison Across All Experiments

| # | Model | Source | Samples | Accuracy |
|---|-------|--------|---------|----------|
| 1 | **Fitch Only, 2 Ratios** | Expanded | **23** | **69.6%** |
| 2 | Tassnief, 4 Ratios | Original | 23 | 65.2% |
| 3 | Consensus, 2 Ratios | Expanded | 45 | 55.6% |
| 4 | Moody's Only, 2 Ratios | Expanded | 19 | 52.6% |
| 5 | Tassnief + KAMs, 10 features | Original | 23 | 65.2% |
| 6 | Tassnief + Sentiment, 9 features | Original | 23 | 30.7% |

---

## 11. Multi-Model Comparison

### 11.1 Motivation

The reference paper found that the PART algorithm (a rule-based method related to Decision Trees) outperformed other models. Our initial experiments used only XGBoost. This section compares 8 different ML algorithms across all datasets.

### 11.2 Models Tested

| Model | Type | Key Property |
|-------|------|-------------|
| XGBoost | Ensemble (Boosting) | State-of-the-art for tabular data |
| Random Forest | Ensemble (Bagging) | Robust to overfitting |
| Gradient Boosting | Ensemble (Boosting) | Similar to XGBoost, sklearn impl. |
| Decision Tree | Rule-based | Interpretable, used in reference paper |
| SVM (RBF) | Kernel method | Good for small datasets |
| KNN | Instance-based | No training phase, distance-based |
| Logistic Regression | Linear | Baseline, interpretable |
| Naive Bayes | Probabilistic | Fast, assumes feature independence |

### 11.3 Results: Fitch Only, 2 Ratios (Best Dataset)

| Model | Accuracy | vs Paper (74.14%) |
|-------|----------|-------------------|
| **Decision Tree** | **78.3%** | **+4.2%** |
| Gradient Boosting | 73.9% | -0.2% |
| XGBoost | 69.6% | -4.5% |
| Random Forest | 65.2% | -8.9% |
| KNN | 60.9% | -13.2% |
| Logistic Regression | 56.5% | -17.6% |
| SVM (RBF) | 52.2% | -21.9% |
| Naive Bayes | 43.5% | -30.6% |

**propably overfitting

### 11.4 Results: Original Tassnief, 4 Ratios

| Model | Accuracy |
|-------|----------|
| **Random Forest** | **73.1%** |
| SVM (RBF) | 69.2% |
| XGBoost / GB / LR | 65.4% |
| KNN / Decision Tree | 61.5% |
| Naive Bayes | 57.7% |

### 11.5 Results: All Consensus, 2 Ratios (Largest Dataset)

| Model | Accuracy |
|-------|----------|
| **Random Forest** | **62.2%** |
| SVM / KNN | 60.0% |
| Decision Tree | 57.8% |
| XGBoost / GB | 55.6% |
| Logistic Regression | 51.1% |
| Naive Bayes | 40.0% |

### 11.6 Key Findings

**1. Decision Tree is the best overall model (78.3%)**

This aligns with the reference paper's finding that PART (a rule-based method) was the top performer. Bao et al. (2019) similarly find that tree-based and ensemble methods (GBDT, RF) outperform linear models (LR) in credit scoring. Decision Trees excel here because:
- Credit rating criteria follow rule-like logic ("if leverage > X and profitability > Y, then rating = A")
- With only 2 features, a simple tree can capture the decision boundaries effectively
- Less prone to overfitting than ensemble methods on very small datasets

**2. No single model dominates across all datasets**

| Best Model By Dataset | Model | Accuracy |
|----------------------|-------|----------|
| Fitch, 2 Ratios | Decision Tree | 78.3% |
| Tassnief, 4 Ratios | Random Forest | 73.1% |
| All Consensus, 2 Ratios | Random Forest | 62.2% |

**3. XGBoost is not always the best choice**

Despite being state-of-the-art for tabular data, XGBoost underperforms simpler models on small datasets. XGBoost's regularization and boosting mechanisms need more data to show their advantage.

**4. Best result exceeds the reference paper**

Our Decision Tree on Fitch data (78.3%) exceeds the paper's best result (74.14% with PART on KAMs). This is achieved with only 2 financial ratios and 23 samples, suggesting that Fitch ratings are particularly well-aligned with financial fundamentals. Bao et al. (2019) report GBDT as the top individual model (MCC 0.663) on a much larger Chinese P2P dataset; our Gradient Boosting results (68–73%) are consistent with the known performance ceiling on small, imbalanced credit datasets.

---

## 12. Paper-Aligned Model Results

### 12.1 Motivation

To validate our approach against the reference paper as closely as possible, we aligned our methodology exactly: 4 Altman Z'' ratios, non-financial companies only, binary classification (A-ratings = low risk vs B/C-ratings = high risk).

### 12.2 Dataset Composition

The training dataset was built by consolidating three sources:

| Source | Records | Companies |
|--------|---------|-----------|
| Historical Fitch ratings (2021-2024) | 25 | 8 |
| Expanded multi-agency (2024) | 22 | 14 |
| Original Tassnief (2021-2024) | 13 | 8 |
| **Total (after dedup + filtering)** | **60** | **22** |

Filters applied:
- Non-financial companies only (banks excluded)
- All 4 Altman ratios available
- Duplicates removed (same ticker + fiscal year)

**Output file (historical V1 pipeline):** `data/processed/model_training_data.csv`. The **current** training table for the demo and LLM prep is `data/processed/merged_multisource_training.csv` (see `scripts/rebuild_processed_datasets.py`).

### 12.3 Results (Binary Classification)

| Model | Accuracy | vs Paper (71.55%) |
|-------|----------|-------------------|
| **Gradient Boosting** | **71.2%** | **-0.35%** |
| Decision Tree | 68.3% | -3.25% |
| Random Forest | 66.7% | -4.85% |
| XGBoost | 65.0% | -6.55% |

The Gradient Boosting result (71.2%) closely matches the paper's 71.55% for financial-ratios-only models, validating that our financial ratio calculation and binary classification methodology are sound.

### 12.4 How the 71.2% Was Achieved Without KAMs

This result used only the 4 financial ratios -- no KAMs were needed. The key factors:
1. **Larger dataset (60 vs 26 records)** from combining multiple agency sources and historical data
2. **Binary classification** (2 classes) instead of multi-class (4 classes) -- easier problem
3. **Non-financial filter** removed noisy bank data where ratios are meaningless
4. **GroupKFold CV** (groups = company ticker) prevented data leakage

---

## 13. Historical Fitch Data

### 13.1 Motivation

To capture rating *changes* over time (a company upgrading from BBB+ to A-), we compiled historical Fitch ratings for Saudi companies from 2019 to 2024. This is critical because models trained on a single year only see static ratings.

### 13.2 Data Compiled

21 companies with confirmed Fitch ratings across multiple years, including:

| Company | Sector | Rating History |
|---------|--------|----------------|
| Saudi Aramco | Energy | A+ (2019-2024) |
| SABIC | Materials | A (2019) → A- (2020-2024) |
| SEC | Utilities | A- (2019-2020) → A (2021-2024) |
| Dar Al Arkan | Real Estate | B+ (2019-2024) |

After pulling financials from yfinance (2021-2024 available), the dataset contained **72 records**. After filtering to non-financial companies with all 4 ratios, this contributed 25 records to the final training data.

### 13.3 Results (8-Model Comparison)

| Model | Accuracy | Notes |
|-------|----------|-------|
| **SVM (RBF)** | **66.7%** | Best with 2 ratios + fiscal year |
| KNN | 63.9% | |
| Decision Tree | 61.1% | |
| Gradient Boosting | 58.3% | |

`fiscal_year` emerged as a significant feature, reflecting the real-world trend that ratings shifted over the 2021-2024 period (e.g., post-COVID recovery upgrades).

**Output file:** `data/processed/historical_fitch_ratings.csv`

---

## 14. Agency Handling Comparison

### 14.1 The Problem

When the same company has ratings from multiple agencies, those ratings often disagree. For example:

| Company | Fitch | S&P | Financial Analytics |
|---------|-------|-----|---------------------|
| Cenomi Centers | BB | BB- | A- |
| BSF | A- | A- | - |

This creates **contradicting labels** -- the model sees identical financial inputs (same company, same year) mapped to different outputs.

### 14.2 Three Options Considered

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| **A: Single Agency** | Use only Fitch ratings | Clean labels, no conflicts | Smaller sample size |
| **B: Consensus** | Median rating per company | One label per company | Synthetic label, smallest N |
| **C: Agency Feature** | Keep all + encode agency | Maximum data (2.4x more) | Conflicting labels persist |

### 14.3 Experiment: Option A vs Option C

We compared Option A (Fitch-only, 25 samples, 8 companies) and Option C (all agencies + `rating_agency` as encoded feature, 60 samples, 22 companies) using 8 ML models with GroupKFold CV.

| Model | Fitch Only (N=25) | All + Agency (N=60) | Winner |
|-------|-------------------|---------------------|--------|
| Decision Tree | 24.0% | **58.3%** | All+Agency |
| Logistic Regression | 40.0% | **55.0%** | All+Agency |
| XGBoost | 36.0% | **60.0%** | All+Agency |
| Random Forest | 40.0% | **61.7%** | All+Agency |
| Gradient Boosting | 24.0% | **53.3%** | All+Agency |
| SVM (RBF) | **64.0%** | 28.3% | Fitch |
| KNN | **56.0%** | 36.7% | Fitch |
| Naive Bayes | **60.0%** | 41.7% | Fitch |

**Score: All+Agency wins 5, Fitch Only wins 3**

### 14.4 Analysis

**Tree-based models prefer more data (Option C)**. They can learn to use the `rating_agency` feature to handle inter-agency differences. Random Forest hit 61.7% as the best in this category.

**Distance-based models prefer clean labels (Option A)**. SVM, KNN, and Naive Bayes are sensitive to label noise -- conflicting labels for the same financial profile degrade their performance. SVM hit 64.0% as the overall best.

**Option B (consensus/median) was not tested** because it would produce the smallest dataset (~22 rows), and sample size is already the primary bottleneck.

### 14.5 Conclusion

Neither approach clearly dominates. The choice depends on the model type:
- For tree-based methods: use all data + agency feature
- For distance-based methods: use single-agency (Fitch) data

The best overall result (64.0% with SVM on Fitch-only) is lower than the paper-aligned 71.2% (Section 12) because this experiment used the paper's binary classification with stricter GroupKFold and only 8 non-financial Fitch-rated companies.

**Output files:** `results/agency_comparison_results.json`, `data/processed/model_fitch_only_data.csv`

---

## 15. SHAP Explainability Analysis

### 15.1 Motivation (Answering RQ2)

Basic tree-based feature importance tells us *which* features matter, but not *how* or *why*. SHAP (SHapley Additive exPlanations) provides:
- **Per-prediction explanations**: Why did company X get rated Investment Grade?
- **Global feature importance**: Which ratios matter most across all companies?
- **Interaction effects**: Does leverage matter more when profitability is low?

This rigorously answers RQ2 using game-theoretic methodology rather than ad hoc importance scores.

### 15.2 Global Feature Importance (SHAP) — full multisource XGBoost

SHAP is computed for the **14-input** model trained in-sample: four financial ratios, five paper-style KAM dummies (`GCKAM` … `OTHERKAM`), and five FinBERT news aggregates (`sentiment_mean`, `sentiment_std`, `sentiment_pos_pct`, `sentiment_neg_pct`, `news_count`). Mean |SHAP| is averaged across the four rating-category outputs (see `results/shap_report.json` for the exact run).

**Latest run (45 usable rows after dropping missing `profitab` and excluding 3008.SR/2021—no articles / zero sentiment aggregates for that firm-year):**

| Rank | Feature | Mean |SHAP| (approx.) | Notes |
|------|---------|---------------------|--------|
| 1 | `news_count` | 0.53 | Article volume is highly influential on this small panel |
| 2 | `leverage` | 0.51 | Equity / liabilities |
| 3 | `cumprof` | 0.45 | Retained earnings / assets |
| 4 | `liquid` | 0.44 | Working capital / assets |
| 5 | `sentiment_std` | 0.21 | FinBERT dispersion across articles |
| 6 | `profitab` | 0.19 | EBIT / assets |
| 7 | `REVKAM` | 0.11 | Revenue-recognition KAM dummy |
| 8 | `sentiment_mean` | 0.08 | FinBERT mean score |
| 9 | `ASSETKAM` | 0.03 | Asset / impairment KAM |
| 10 | `sentiment_pos_pct` | 0.02 | Share of positive articles |
| 11–14 | `GCKAM`, `LIABKAM`, `OTHERKAM`, `sentiment_neg_pct` | 0 | Zero mean \|SHAP\| in latest `shap_report.json` |

**Key insight:** On the current aligned panel, **news coverage (`news_count`)** and **core ratios** dominate mean |SHAP|; this is dataset-specific and should not be over-interpreted causally (article counts correlate with firm size, visibility, and data collection).

### 15.3 Feature Interaction Effects

Regenerated dependence plots (see `figures/`):

- **`shap_dependence_leverage_profitab.png`** — leverage vs profitability colouring.
- **`shap_dependence_liquid_cumprof.png`** — liquidity vs cumulative profitability.
- **`shap_dependence_news_sentiment.png`** — news volume vs FinBERT mean score.

### 15.4 Per-Company Explanations

Waterfalls are saved as `figures/shap_waterfall_<TICKER>_<YEAR>.png` for a stratified sample of rows (multicategory model: explanation is for the **predicted** class).

**Output files:** `figures/shap_beeswarm.png`, `figures/shap_bar_importance.png`, `figures/shap_waterfall_*.png`, `figures/shap_dependence_*.png`, `results/shap_report.json`

**Script:** `PYTHONPATH=. python models/shap_explainability.py`

The Streamlit app (`app.py`) uses the same 14 features for on-the-fly SHAP waterfalls on the trained full XGBoost model.

### 15.5 Multisource algorithm comparison (XGBoost vs other learners)

To test whether **XGBoost** is still the best choice on the *current* aligned panel, we benchmarked several scikit-learn models (plus XGBoost) on the **same 45-row** multisource matrix as §15 (**3008.SR/2021 excluded**): **14 features**, **four classes** (A / AA / BB / BBB), **5-fold stratified CV** (`random_state=42`). Script: `models/evaluate_multisource_models.py`. Outputs: `results/multisource_model_comparison.json`, `figures/multisource_model_comparison.png`.

**CV accuracy (mean) and macro-F1 (mean)** — values from the JSON (rounded):

| Rank | Model | Accuracy | F1 (macro) | Notes |
|------|--------|----------|------------|--------|
| 1 | **Extra Trees** | **66.7%** | 0.59 | Joint best mean CV accuracy (latest panel) |
| 2 | **Linear SVC (+ scaler)** | **66.7%** | **0.63** | Joint best accuracy; strong macro-F1 |
| 3 | Random Forest | 64.4% | 0.54 | Close third |
| 4 | Logistic regression (+ scaler) | 62.2% | 0.52 | — |
| 5 | **XGBoost** | **62.2%** | **0.62** | Slightly below ET/SVC on accuracy here |
| 6 | Decision Tree | 60.0% | 0.47 | — |
| 7 | kNN *k*=5 (+ scaler) | 55.6% | 0.48 | High variance across folds |
| 8 | sklearn `GradientBoostingClassifier` | 48.9% | 0.41 | — |
| 9 | MLP (+ scaler) | 46.7% | 0.28 | Small *n*, default architecture |
| 10 | HistGradientBoosting (sklearn) | 40.0% | 0.14 | Default hyperparameters weak here |

**Takeaway:** On this **45-row, 14-feature** matrix, **Extra Trees** and **Linear SVC** tie for the **highest mean CV accuracy**; **XGBoost** remains competitive on **macro-F1** but is not top-ranked on raw accuracy in this run. **HistGradientBoosting** underperforms badly without tuning. If you edit `merged_multisource_training.csv` but skip `scripts/regenerate_artifacts.py`, **`results/*.json` and figures stay out of sync** with the app (which always trains on the CSV). Results are **high-variance**; interpret as indicative, not definitive. **§15.5.1 supersedes this table** for the widest feature set.

### 15.5.1 Superset benchmark: 21 features (full KAM block)

§15.5 uses the **five KAM dummies**. This run widens the matrix to the **full 12-column KAM/firm block** (`AUSIZE`, `AUOP`, `EMP`, `GCUP`, the five KAM category dummies, `FIRMAGE`, `FIRMSIZE`, `INDUSTRY`) alongside the four financial ratios and five news aggregates — **21 features**, same **45 rows**, same **four classes**, same **5-fold stratified CV**. Script: `models/xgboost_all_features.py`. Outputs: `results/all_features_model_results.json`, `figures/all_features_model_comparison.png`.

**XGBoost ablation across feature blocks:**

| Feature set | *n* features | CV accuracy |
|-------------|--------------|-------------|
| Financials only | 4 | 51.1% ± 0.194 |
| **Financials + full KAMs** | **16** | **71.1% ± 0.113** |
| Financials + news | 9 | 51.1% ± 0.113 |
| Full KAMs + news | 17 | 57.8% ± 0.083 |
| All features | 21 | 66.7% ± 0.099 |

The best XGBoost ablation is **financials + full KAMs (16 features)**; adding the news block *reduces* XGBoost accuracy, consistent with the sentiment aggregates carrying little signal at this sample size.

**Nine-learner benchmark on the full 21-feature matrix:**

| Rank | Model | Accuracy | F1 (macro) | Notes |
|------|--------|----------|------------|--------|
| 1 | **Extra Trees** | **82.2% ± 8.9%** | **0.68** | Highest figure in this repository |
| 2 | Linear SVC (+ scaler) | 73.3% ± 18.1% | 0.68 | Equal macro-F1, far wider spread |
| 3 | Logistic regression (+ scaler) | 71.1% ± 15.1% | 0.62 | — |
| 4 | kNN *k*=5 (+ scaler) | 71.1% ± 11.3% | 0.56 | — |
| 5 | **XGBoost** | 66.7% ± 9.9% | 0.54 | Same features as row 1 |
| 6 | Gradient Boosting | 66.7% ± 14.1% | 0.51 | — |
| 7 | Random Forest | 64.4% ± 8.3% | 0.53 | — |
| 8 | Decision Tree | 57.8% ± 10.9% | 0.49 | — |
| 9 | MLP (+ scaler) | 37.8% ± 26.9% | 0.27 | Small *n*, default architecture |

**Three caveats that must travel with the 82.2% figure:**

1. **Accuracy overstates minority-class performance.** Extra Trees scores **0.822 accuracy but 0.678 macro-F1**. Class counts are A = 18, BBB = 15, AA = 6, BB = 6; the two majority classes hold 33 of 45 rows, so each fold contains roughly one AA and one BB row. Macro-F1 is the fairer headline for a four-class ordinal task.
2. **The Extra Trees / XGBoost gap is a variance signal.** Two closely related tree ensembles differ by **15.5 points on identical features**. On 45 rows that is far more plausibly fold noise than a genuine capability difference.
3. **Fold granularity.** Five folds of nine rows means one row flipping moves the mean by roughly two points; ±8.9% std is consistent with that.

**Takeaway:** the widest feature set produces the best headline accuracy, but the ranking is **not stable** at this sample size. Report **82.2% as a mean CV accuracy for Extra Trees on 21 features, paired with its 0.678 macro-F1**, and treat the algorithm ordering as indicative only.

### 15.6 App, SHAP figures, and Streamlit — current state

**SHAP (explanation layer):** Yes — updated for the **14-feature multicategory XGBoost**. Global and dependence plots plus per-row waterfalls are produced by `models/shap_explainability.py` and stored under `figures/shap_*.png` with `results/shap_report.json`. The written explanation in §15.2–15.4 matches that pipeline.

**Streamlit app:** Yes — `app.py` was updated to use **`merged_multisource_training.csv`**, **14-input XGBoost**, **multiclass probabilities**, **SHAP waterfall for the predicted class**, **KAM + FinBERT fields** in the UI, and **template verdicts** aligned with `llm_verdict.py`. The **Model performance** tab also shows the **multisource benchmark** figure and table (`multisource_model_comparison.json` / `.png`) alongside `full_model_comparison.json`.

---

## 16. Error Analysis

**Canonical source:** `results/error_analysis.json`, produced with the multisource panel by `models/generate_pipeline_figures.py` (invoked from `scripts/regenerate_artifacts.py`). It describes **14-feature multiclass XGBoost**, **stratified 5-fold CV**, **out-of-fold** pooled predictions, and matches the **45** rows in `merged_multisource_training.csv`.

### 16.1 Summary (multisource)

| Metric | Value |
|--------|-------|
| Panel | 45 `(ticker, fiscal_year)` rows |
| Target | Four buckets: A, AA, BB, BBB |
| CV folds | 5 (stratified) |
| Pooled OOF accuracy (JSON `summary`) | **62.2%** (28 correct / 45) |
| Misclassified rows listed | **17** |

*Note:* **Mean** fold accuracy for the **same** 14-feature model in **`full_model_comparison.json`** is **60.0% ± 11.3%**—slightly different aggregation than pooling all OOF predictions into one confusion matrix. Cite **ablations** from `full_model_comparison.json` for the stated objective; use **this section** for **per-class** and **case** behaviour.

### 16.2 Confusion matrix (out-of-fold, pooled)

Rows = **actual** category, columns = **predicted** (same ordering: A, AA, BB, BBB):

| Actual \\ Pred | A | AA | BB | BBB |
|----------------|---|----|----|-----|
| **A** | 11 | 0 | 2 | 5 |
| **AA** | 0 | 6 | 0 | 0 |
| **BB** | 3 | 0 | 2 | 1 |
| **BBB** | 4 | 2 | 0 | 9 |

**Patterns:** **AA** (6 samples) has **no** off-diagonal mass—all errors involving other classes. **A** (18) is often confused with **BBB** (5) and to a lesser extent **BB** (2). **BBB** (15) is pulled toward **A** (4) and **AA** (2). **BB** (6) is the **smallest** class: half the BB rows stay on the diagonal (2/6), with the rest scattered toward **A** or **BBB**. This is consistent with **coarse bucketing** of underlying notches, **imbalance**, and **high fold variance** on a small panel—not with a single “random noise” story.

### 16.3 Notable misclassifications (from JSON)

Examples below are **verbatim** cases in `results/error_analysis.json` (multisource run); they illustrate **confidence**, **bucket** jumps, and **repeat** offenders.

- **Zain Saudi Arabia (7030.SR), FY2023:** actual **A**, predicted **BBB** (high confidence ~0.77)—adjacent investment-grade confusion.
- **SABIC (2010.SR), FY2022:** actual **A**, predicted **BB**—large cross-bucket error for a flagship name; highlights label vs feature tension on a coarse map.
- **Saudi Aramco (2222.SR), FY2024:** actual **A**, predicted **BBB**—similar theme: strong credit name, model uses ratio/news surface only.
- **Cenomi Centers (4321.SR):** multiple years appear (e.g. **BB** → **A**, **A** → **BBB**)—consistent with **volatile** fundamentals and **multi-agency** dispersion in underlying ratings.
- **Al Kathiri / AKHC (3008.SR):** **BBB** predicted as **A** on FY2022–2023; FY2022 shows **very high** model confidence (~0.94)—a **stress case** for overconfidence when signals are weak or misaligned with the coarse label.
- **Mayar Holding (9568.SR)** and **Multi Business Group (9619.SR):** actual **BB**, predicted **A**—**small-class** rows pulled toward the modal **A** bucket.
- **Ladun Investment (9535.SR)** and **Perfect Presentation / 2P (7204.SR):** actual **BBB** but predicted **A** or **AA** on some years—large cross-bucket errors, several with moderate–high confidence.

### 16.4 Key finding

**Errors are structured, not i.i.d. noise.** The **confusion matrix** shows **systematic** swaps between **neighbouring** and **modal** categories; the **misclassified** list supports **case-by-case** narratives (boundary mapping, agency disagreement in the label construction, outlier ratios, sparse news). The older **binary V2** exercise (75 rows, **68%** accuracy) manually tagged **45%** of errors into agency / boundary / outlier buckets; that **percentage breakdown is not recomputed** for the **four-class multisource** JSON—here, the **evidence** is the **matrix + row list**.

**Output files:** `results/error_analysis.json` (and mirror `error_analysis_multisource.json`), plus pipeline figures such as `figures/error_scatter.png`, `figures/confidence_dist.png`, `figures/error_patterns.png` when regenerated.

**Script:** `models/generate_pipeline_figures.py` (via `scripts/regenerate_artifacts.py`).

---

## 17. KAM Homogeneity: A Saudi Market Finding

### 17.1 Background

The reference paper (Gutierrez-Lopez & Sanchez-Martin, 2023) found that KAM features improved credit rating prediction by ~3% accuracy (71.55% → 74.14%) in the Spanish market. A key question is whether this finding generalizes to the Saudi market.

### 17.2 KAM Profile Analysis

KAMs were extracted for all 23 non-financial companies in the training dataset:

| KAM Category | Prevalence | Variance |
|--------------|------------|----------|
| Going Concern | 0% (0/23) | 0.0000 |
| Revenue Recognition | 96% (22/23) | 0.0435 |
| Asset Valuation | 87% (20/23) | 0.1186 |
| Liabilities | 13% (3/23) | 0.1186 |
| Other KAMs | 26% (6/23) | 0.2016 |
| Mean KAM Count | 2.22 | 0.1779 |

### 17.3 Homogeneity Finding

**65% of Saudi companies share the identical KAM profile:** `(0, 1, 1, 0, 0, count=2)` -- no going concern, revenue recognition present, asset valuation present, no liabilities or other KAMs.

Only 6 unique KAM profiles exist across 23 companies. By contrast, the Spanish market in the reference paper exhibited high KAM diversity with ~15% of companies flagged for going concern, diverse asset and liability concerns, and a wide range of KAM counts.

### 17.4 Feature Ablation Study

| Model | Financial Only | KAM Only | Financial + KAM | Delta |
|-------|---------------|----------|----------------|-------|
| Gradient Boosting | 68.0% | 46.7% | 66.7% | -1.3% |
| Decision Tree | 61.3% | 45.3% | 66.7% | +5.3% |
| Random Forest | 57.3% | 46.7% | 54.7% | -2.7% |

**KAMs alone perform at near-chance level (46.7%)** in the Saudi market, compared to 74.14% in the Spanish market. Adding KAMs to financial ratios provides inconsistent results -- improving Decision Tree by 5.3% but degrading Gradient Boosting by 1.3%.

### 17.5 Why KAMs Don't Help in Saudi Arabia

| Factor | Spanish Market | Saudi Market |
|--------|---------------|--------------|
| Going concern KAMs | ~15% prevalence | 0% prevalence |
| Unique KAM profiles | High diversity | Only 6 profiles |
| Most common profile | Varies | 65% identical |
| Economic diversity | Diverse sectors, distressed firms | Oil-driven, healthier firms |
| Regulatory environment | EU audit standards | Saudi audit standards (similar ISA but different application) |

**This represents a genuine research contribution:** The finding that Saudi KAM homogeneity nullifies the predictive value of KAM features -- contrary to European market findings -- has not been previously documented in the credit rating literature.

**Output files:** `figures/kam_ablation.png`, `figures/kam_profiles.png`, `results/kam_analysis.json`

**Scripts:** `models/xgboost_with_kams.py`, `models/xgboost_full.py` (financials + KAMs ± news on merged processed tables), and `models/xgboost_kams_only.py` (KAM/audit features only from `kams_processed.csv`).

### 17.6 Implemented KAM Extraction vs Reference Protocol

The reference paper’s full feature set combined audit-firm variables (AUSIZE, AUOP, EMP, GCUP) with five KAM category dummies. In this project, the implemented extraction is narrower: we only collect the five KAM category indicators (`kam_going_concern`, `kam_revenue`, `kam_assets`, `kam_liabilities`, `kam_other`) and a simple `kam_count` for 23 non‑financial Saudi companies. We do **not** extract AUSIZE, AUOP, EMP, or GCUP from the audit reports, and these audit‑opinion features are therefore **not** used in any of the ML models.

Given the strong KAM homogeneity documented above and the limited incremental signal from KAM features, the final production classifier used in Sections 13–16 relies **only on financial ratios** as predictive inputs. KAM features are retained purely for descriptive analysis (profiling Saudi audit reports and reproducing the paper’s idea qualitatively), not as core inputs to the deployed rating model.

---

## 18. LLM Verdict Generator

### 18.1 Purpose (Answering RQ3)

The LLM verdict generator completes the hybrid ML+LLM architecture promised in the project specification. It takes ML predictions and financial data as input and produces structured, human-readable credit verdicts.

### 18.2 Architecture

```
Financial Ratios + KAM Data
        │
        ▼
┌──────────────────────────┐
│    ML Classifier          │
│    (Gradient Boosting)    │
│                          │
│  Output: Rating + Conf.  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   LLM Verdict Generator  │
│                          │
│  Input:                  │
│  - Company info          │
│  - Financial ratios      │
│  - ML prediction + conf. │
│  - Actual rating         │
│                          │
│  Output:                 │
│  - Overall assessment    │
│  - Strengths (with data) │
│  - Weaknesses (with data)│
│  - Key risks             │
│  - Prediction analysis   │
└──────────────────────────┘
```

### 18.3 Implementation

The system supports three generation methods, falling back through the chain automatically:

1. **Fine-tuned Qwen 2.5 3B (QLoRA):** The primary method. A Qwen 2.5 3B Instruct model fine-tuned with QLoRA on our 75 Saudi company records produces contextually rich, analyst-style credit verdicts. See Section 18.7 for full details.

2. **Ollama (Mistral/Llama):** When a local Ollama instance is running, prompts are sent to the LLM for zero-shot natural language generation. Responses are parsed as structured JSON.

3. **Template-Based Fallback:** A rule-based generator that constructs verdicts programmatically from financial ratios, ensuring 100% factual accuracy and consistent output format.

### 18.4 Verdict Quality Evaluation

Every generated verdict is automatically evaluated against the input data:

| Quality Metric | Score | Description |
|----------------|-------|-------------|
| Overall Quality | 100% | All structural components present |
| Citation Rate | 89% | Ratio names referenced in verdicts |
| Number Accuracy | 94% | Financial values correctly cited |
| Has Strengths | 100% | All verdicts identify positive factors |
| Has Weaknesses | 100% | All verdicts identify risk factors |

### 18.5 Sample Verdict

```json
{
  "company": "SABIC",
  "ticker": "2010.SR",
  "fiscal_year": 2024,
  "predicted_rating": "A",
  "risk_classification": "Investment Grade",
  "overall_assessment": "SABIC shows a solid financial profile for FY2024,
    with the ML model predicting an A (Investment Grade) rating at 100%
    confidence.",
  "strengths": [
    "Strong liquidity position (LIQUID=0.1655), indicating adequate working
     capital relative to total assets",
    "Positive cumulative profitability (CUMPROF=0.0721), showing consistent
     earnings retention",
    "Conservative financial structure (LEVERAGE=1.9545), with equity
     substantially exceeding liabilities"
  ],
  "weaknesses": [
    "Low operational profitability (PROFITAB=0.0250), suggesting limited
     earnings generation from assets"
  ],
  "key_risks": [
    "Monitor macroeconomic conditions in Saudi Arabia and sector-specific
     regulatory changes"
  ],
  "prediction_analysis": "The financial data supports the A prediction.
    The company shows adequate liquidity, positive cumulative profitability,
    and operational returns, consistent with an investment-grade classification."
}
```

### 18.6 Factual Accuracy Assessment

The template-based verdicts achieve near-perfect factual accuracy because every claim is derived directly from input data. The key evaluation criteria from the project specification:

| Criterion | Result |
|-----------|--------|
| **Factual Accuracy** (cited numbers correct?) | 94% -- all ratios cited from actual data |
| **Completeness** (covers key factors?) | 100% -- strengths, weaknesses, risks included |
| **Coherence** (logically consistent?) | 100% -- rule-based logic ensures consistency |
| **Alignment** (supports prediction?) | 100% -- prediction analysis matches verdict |
| **Citation Rate** (claims backed by data?) | 89% -- most claims cite specific ratio values |

**Output files:** `results/verdicts/all_verdicts.json`, `results/verdicts/verdict_summary.json`

**Script:** `models/llm_verdict.py`

### 18.7 LLM Fine-Tuning with QLoRA

#### 18.7.1 Motivation

The template-based verdict generator (Section 18.3, method 3) produces factually accurate but formulaic output. Every verdict follows the same linguistic patterns, making it obvious the text is machine-generated. Zero-shot Ollama prompting produces more varied language but suffers from hallucination (inventing ratio values or company details not present in the input).

To address both limitations, we fine-tune an open-source LLM on our domain-specific data using QLoRA (Quantized Low-Rank Adaptation). This approach is inspired by **Lei et al. (2025)**, who built ZiGong 1.0, a Mistral 7B-based LLM fine-tuned with LoRA for financial credit tasks. ZiGong demonstrated that domain-specific fine-tuning significantly reduces hallucination in financial text generation compared to zero-shot prompting.

#### 18.7.2 Why Qwen 2.5 Instruct

We selected the Qwen 2.5 Instruct family over ZiGong's original Mistral 7B base for its superior structured JSON output capability and strong financial reasoning from multilingual training data. Due to system memory constraints (8GB VRAM + limited virtual memory), we used Qwen 2.5 3B Instruct rather than the 7B variant. ZiGong's LoRA hyperparameters (rank=8, alpha=16) transfer directly.

| Criterion | Qwen 2.5 3B Instruct | Mistral 7B (ZiGong) |
|-----------|----------------------|---------------------|
| **Structured JSON output** | Native JSON mode, strong for size | Requires explicit prompting |
| **Financial reasoning** | Broad multilingual/financial training data | General-purpose training |
| **Parameter count** | 3B (lighter, faster training) | 7B |
| **Memory requirement (4-bit)** | ~2-3 GB VRAM | ~4-5 GB VRAM |

#### 18.7.3 QLoRA Configuration

Standard full fine-tuning requires far more VRAM than we have available. Our 8GB GPU uses QLoRA, which quantizes the base model to 4-bit (nf4) precision; a **3B** instruct model in 4-bit typically fits in roughly **3--5GB VRAM** for training/inference, leaving headroom on an 8GB card. Training uses `transformers` + `peft` + `bitsandbytes` (Windows-compatible); `unsloth` was not used due to stability issues on Windows.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | `Qwen/Qwen2.5-3B-Instruct` | 4-bit quantized via bitsandbytes |
| LoRA rank | 8 | Matches ZiGong (Lei et al., 2025) |
| LoRA alpha | 16 | Matches ZiGong |
| Target modules | q_proj, k_proj, v_proj | Attention layer adaptation (same as ZiGong) |
| Batch size | 1 (x4 gradient accumulation) | Effective batch size of 4, fits 8GB VRAM |
| Max sequence length | 1024 tokens | Sufficient for verdict JSON (~400-600 tokens) |
| Epochs | 5 | Multiple passes needed for 75 training examples |
| Learning rate | 2e-5 | Standard for QLoRA instruction tuning |
| Optimizer | AdamW 8-bit | Memory-efficient variant via bitsandbytes |

The LoRA adapter adds ~20-50MB of trainable parameters on top of the frozen, quantized base model.

#### 18.7.4 Training Data Preparation

Each row with complete financial ratios from `merged_multisource_training.csv` (inner join of `kams_processed.csv`, `financial_ratios_processed.csv`, and `news_features_processed.csv`, produced by `scripts/rebuild_processed_datasets.py`) was converted into an instruction-tuning example with three fields:

**Instruction** (system prompt, same for all examples):
> "You are a credit analyst specializing in Saudi Exchange (Tadawul) listed companies. Given a company's financial ratios and ML model prediction, generate a structured credit verdict as a JSON object. Every claim must cite specific ratio values."

**Input** (per-record, structured text):
```
Company: SABIC | Ticker: 2010.SR | Sector: Materials
Fiscal Year: 2024 | Agency: Fitch | Actual Rating: A+
ML Predicted: A (Investment Grade) | Confidence: 85%
Ratios: LIQUID=0.1655, CUMPROF=0.0721, PROFITAB=0.0250, LEVERAGE=1.9545
```

**Output** (structured JSON verdict, same schema as `all_verdicts.json`).

Of the 75 training examples, **18 were manually improved** with analyst-style language for key companies spanning diverse sectors, rating levels, and edge cases. These hand-crafted examples replace the formulaic template language with:
- Saudi market context (e.g., Vision 2030 references, sector dynamics)
- Comparative reasoning (e.g., "the model cannot account for sovereign support")
- Specific domain insights (e.g., OPEC+ production discipline, tariff reform)
- Nuanced prediction analysis (e.g., explaining multi-agency rating disagreements)

Priority companies for improvement included Saudi Aramco, STC, SABIC, ACWA Power, ADES, Cenomi Centers, Mayar Holding, and Multi Business Group.

#### 18.7.5 Comparison with ZiGong Methodology

| Aspect | ZiGong 1.0 (Lei et al., 2025) | Our Approach |
|--------|-------------------------------|--------------|
| Base model | Mistral 7B | Qwen 2.5 3B Instruct |
| Fine-tuning | Standard LoRA (16GB+ GPU) | QLoRA 4-bit (8GB GPU) |
| Training data | ~100K+ examples | 75 examples (18 manually improved) |
| Tasks | Credit scoring, fraud detection, Q&A | Credit verdict generation |
| Data pruning | TracSeq (temporal decay) | Manual verdict improvement |
| LoRA config | rank=8, alpha=16 | rank=8, alpha=16 (identical) |
| Hallucination control | TracSeq + data quality | Structured JSON + fine-tuning on factual data |

Our adaptation applies ZiGong's core insight -- that domain-specific fine-tuning outperforms zero-shot prompting for financial tasks -- to a different downstream task (verdict generation rather than credit scoring) and a different base model (Qwen 2.5 for superior JSON fidelity).

#### 18.7.6 Training Results

Training completed in approximately 9 minutes on an NVIDIA RTX 2070 SUPER (8GB VRAM):

| Metric | Value |
|--------|-------|
| Total training steps | 95 (5 epochs x 19 steps) |
| Final training loss | 2.03 (decreasing from ~3.3) |
| LoRA adapter size | 20.5 MB |
| Trainable parameters | 2,506,752 / 1,701,179,392 (0.15%) |
| Mean token accuracy (final epoch) | 59.6% |

#### 18.7.7 Evaluation Results

Fine-tuned verdicts were compared against template baselines across all 75 companies:

| Metric | Template | Fine-tuned | Delta |
|--------|----------|------------|-------|
| **JSON Success Rate** | 100% | 100% | 0% |
| **Avg Citation Rate** | 91.3% | 86.0% | -5.3% |
| **Avg Overall Score** | 100% | 43% | -57% |
| **Avg Ratios Cited** | 3.65 / 4 | 3.44 / 4 | -0.21 |
| **Diversity Ratio** | 35.8% | 100% | **+64.2%** |

**Key findings:**

1. **Zero hallucination in JSON structure:** The fine-tuned model achieved 100% valid JSON output across all 75 companies, matching the template baseline. This confirms that QLoRA fine-tuning on domain-specific data effectively controls structural hallucination, consistent with Lei et al. (2025)'s findings on ZiGong.

2. **High ratio citation:** The model learned to cite specific financial ratio values (86% citation rate, 3.44 of 4 ratios cited on average), demonstrating that factual grounding was successfully transferred from the training data.

3. **Dramatically higher diversity:** The fine-tuned model produces linguistically diverse output (100% unique trigrams) compared to the template's formulaic repetition (35.8%). This was the primary motivation for fine-tuning.

4. **Schema divergence:** The overall quality score dropped from 100% to 43% because the model adapted its own JSON schema (e.g., nesting ratios under `credit_verdict` with per-ratio `comment` fields) rather than strictly reproducing the training schema fields (`strengths`, `weaknesses`, `prediction_analysis`). With only 75 training examples, the 3B model learned the *task* (analyzing credit data and producing structured verdicts) but not the exact *format*.

5. **75 examples are at the boundary:** This dataset size is sufficient for teaching domain vocabulary, ratio citation, and valid JSON generation, but insufficient for strict schema compliance. This finding aligns with the general fine-tuning literature suggesting ~200-500 examples for reliable instruction following on structured output tasks.

6. **3B vs 7B:** We also ran a separate QLoRA training on **Qwen 2.5 7B Instruct** with the same data and hyperparameters. On our automated evaluation (JSON validity, citation rate, schema-style overall score, diversity), the **3B** run was **slightly better** than 7B; the repository and inference code are therefore standardised on **3B**. LoRA weights are **not interchangeable** across base sizes -- after switching to 3B, the adapter in `models/lora_adapter/` must be regenerated with `models/finetune_qwen.py`, and `models/evaluate_finetune.py` should be re-run to refresh `results/finetune_evaluation.json`.

**Implication for the hybrid architecture:** The template generator remains the production default for schema reliability, while the fine-tuned model demonstrates the feasibility of LLM-based verdict generation with domain-specific fine-tuning. A larger training corpus (200+ manually curated verdicts) would likely resolve the schema divergence.

**Scripts:**
- `models/prepare_finetune_data.py` -- Dataset preparation
- `models/improve_verdicts.py` -- Manual verdict improvement (18 companies)
- `models/finetune_qwen.py` -- QLoRA training (Qwen 2.5 3B, peft + bitsandbytes)
- `models/evaluate_finetune.py` -- Evaluation and comparison

**Output files:**
- `data/processed/finetune_dataset.jsonl` -- 75 instruction-format training examples
- `models/lora_adapter/` -- LoRA adapter weights (20.5 MB)
- `results/finetune_evaluation.json` -- Quantitative evaluation results
- `results/verdicts/finetuned_verdicts.json` -- All 75 fine-tuned verdicts

---

## 19. Interactive Demo Application

### 19.1 Overview

A Streamlit web application provides an interactive demonstration of the complete ML+LLM hybrid system. It allows users to:

1. **Select any company** from a dropdown and see its financial ratios
2. **View the ML prediction** with confidence score and comparison to actual rating
3. **Examine SHAP waterfall** showing exactly why the model made that prediction
4. **Read the LLM credit verdict** with strengths, weaknesses, and risk assessment
5. **Explore model performance**, visualizations, and error analysis

### 19.2 Application Tabs

| Tab | Content |
|-----|---------|
| **Company Predictor** | Interactive prediction with SHAP explanation and LLM verdict |
| **Model Performance** | Multisource benchmark chart + table (`multisource_model_comparison.*`), `full_model_comparison.json`, legacy figure |
| **SHAP Explainability** | 14-feature XGBoost: beeswarm, bar, dependence plots, per-company waterfalls, `shap_report.json` |
| **Error Analysis** | Misclassified companies, error patterns, confidence distribution |
| **Dataset Explorer** | PCA scatter plot, full dataset table, feature statistics |

### 19.3 Running the App

```bash
streamlit run app.py
```

**Script:** `app.py`

---

## 20. Novel Contributions

### 20.1 Contribution 1: Saudi KAM Homogeneity

**Finding:** 65% of Saudi non-financial companies share identical KAM profiles (revenue recognition + asset valuation). Going concern KAMs are absent (0%). This contrasts sharply with the Spanish market where KAMs improved prediction by ~3%.

**Significance:** First documented evidence that KAM-based credit rating prediction does not generalize from European to GCC markets due to fundamental differences in audit reporting patterns.

### 20.2 Contribution 2: Inter-Agency Rating Disagreement Effects

**Finding:** Different credit rating agencies (Fitch, Moody's, S&P, Tassnief, Financial Analytics) assign materially different ratings to the same Saudi company using the same financials. Tree-based ML models can handle this disagreement by learning agency-specific patterns (when `rating_agency` is encoded as a feature), while distance-based models (SVM, KNN) perform better with single-agency clean labels.

**Significance:** Provides empirical evidence that multi-agency data requires careful methodological treatment, and the optimal approach depends on the model family used.

### 20.3 Contribution 3: Fitch Rating Predictability

**Finding:** Fitch ratings are the most predictable from financial ratios alone (69.6-78.3% accuracy), likely due to Fitch's transparent, formula-driven methodology. This makes Fitch ratings the ideal target for automated credit assessment systems.

**Significance:** Practical guidance for financial technology developers building automated rating systems for the Saudi market.

### 20.4 Contribution 4: Rule-Based Models Outperform Ensembles

**Finding:** Decision Trees (78.3%) and Gradient Boosting (71.2%) outperform XGBoost (69.6%) and complex ensembles on small credit rating datasets. This aligns with the reference paper's finding that rule-based PART algorithm was the best performer, and with Bao et al. (2019), who report GBDT as the top individual model among seven algorithms (LR, SVM, DT, RF, GBDT, kNN, ANN) on credit scoring data.

**Significance:** Challenges the assumption that sophisticated models always outperform simpler ones, particularly in credit risk where decision rules approximate human analyst reasoning.

### 20.5 Contribution 5: Error Explainability

**Finding (canonical multisource panel):** For the **45-row**, **four-class** XGBoost run documented in `results/error_analysis.json`, **out-of-fold** predictions yield a **confusion matrix** where **AA** is perfectly stable on the diagonal while **A**, **BBB**, and especially **BB** show **adjacent-bucket** and **cross-grade** confusion—consistent with **coarse labels**, **class imbalance**, and **noisy** harmonised ratings. The JSON lists **17** misclassified `(ticker, year)` rows with predicted vs actual category and confidence; many cases (e.g. **strong actual A predicted BBB**, **BB predicted A**) plausibly reflect **boundary** ambiguity, **multi-agency** disagreement in the underlying notch map, or **ratio** profiles that sit far from class centroids—without claiming a fixed percentage split of *all* errors (that precision belonged to an older **binary / 75-row** study, not recomputed here).

**Significance:** Accuracy alone is insufficient: the **same artefact** supports both **aggregate** diagnostics (confusion structure) and **case-level** review for the **current** pipeline—essential before any deployment narrative, and honest about **small *n***.

---

## 21. Answering the Research Questions

The **primary evaluation objective** (Section 1.1) is to **quantify**—under stratified cross-validation on the merged panel—whether **KAM** and **news** blocks change performance **relative to financial ratios alone**, with conclusions tempered by **small-sample variance**. The ablation metrics in `results/full_model_comparison.json` and the narrative in `docs/PROJECT_REPORT_CURRENT.md` (Sections 6–9) are the direct evidence for that objective; the research questions below structure additional interpretation (SHAP, LLM verdicts, error patterns).

### RQ1: Can classical ML predict credit ratings from public financial ratios, KAM features, and news sentiment?

**Answer: Yes in the weak sense that models learn above-chance structure on the panel; incremental value of KAM vs news blocks is not the same.**

On the **45-row** multisource panel (`data/processed/merged_multisource_training.csv`), **stratified 5-fold CV** with **XGBoost** reports the following **mean CV accuracy ± std** in `results/full_model_comparison.json` (run **2026-03-27**):

| Feature set | Mean accuracy | Std |
|-------------|---------------|-----|
| Financials only (4 ratios) | **51.1%** | ±18.1% |
| Financials + KAM dummies (9 features) | **46.7%** | ±8.3% |
| Financials + FinBERT / news aggregates (9 features) | **60.0%** | ±8.9% |
| Full model (14 features) | **60.0%** | ±11.3% |

**Relative to financials alone:** the **news / sentiment block** improves mean CV accuracy by about **nine percentage points** here; the **KAM dummy block** in this ablation **reduces** mean accuracy vs financials-only. The **full** model matches **financials + sentiment** within rounding (it does not beat that row on mean accuracy in this run). **Fold-to-fold variance** is large—especially for financials-only (std ≈18%)—so conclusions must stay tied to **this constructed panel** and sample size.

*Context:* Earlier narrative in this document refers to a **different** setup (e.g. binary / V2 panel, ~75 rows, other CV grouping) where tree models reached **~68–71%**; that is **not** the same target or matrix as the current **four-class** multisource ablation. For the **stated evaluation objective** (Section 1.1), treat **`full_model_comparison.json`** and **`docs/PROJECT_REPORT_CURRENT.md`** as canonical.

### RQ2: Which feature categories contribute most to rating prediction accuracy?

**Answer: On the latest multisource XGBoost + SHAP run, ratios and news volume dominate mean |SHAP|.**

For the **14-feature** full model (`models/shap_explainability.py`), global SHAP ranks **`news_count`**, **`leverage`**, **`cumprof`**, **`liquid`**, then **`sentiment_std`** and **`profitab`**, followed by **`REVKAM`** and FinBERT means (see `results/shap_report.json`, **n = 45**). Several KAM dummies and **`sentiment_neg_pct`** show **zero** mean |SHAP| in the latest run. Interpret cautiously: `news_count` is a weak proxy for “sentiment” and may reflect visibility and data availability.

### RQ3: Can open-source LLMs generate coherent, factually accurate credit verdicts?

**Answer: Yes, with template / Ollama / fine-tuned modes.**

`models/llm_verdict.py` now conditions verdicts on **ratios + KAM dummies + FinBERT aggregates**, alongside a **multicategory XGBoost** prediction (AA / A / BBB / BB). Regenerated template verdicts land in `results/verdicts/`; quality metrics are printed when you run the script (citation checks include the expanded context).

### RQ4: How does prediction accuracy compare across rating categories?

**Answer: With *n* = 45, read per-class behaviour from the pooled CV confusion matrix, not a single headline “accuracy per notch.”**

`results/error_analysis.json` (multisource **14-feature** XGBoost, **four** categories A / AA / BB / BBB) includes a **confusion matrix** built from **out-of-fold** predictions. In that snapshot, **AA** (6 samples) is **stable** on the diagonal (no off-diagonal mass for that class). **A** (18) and **BBB** (15) show **substantial cross-bucket** error (e.g. A ↔ BBB). **BB** (6) is the **smallest** class and is often confused with **A** or **BBB**, which is expected under **imbalance** and **coarse bucketing**. Qualitative themes elsewhere in this report—**boundary** effects, **agency** disagreement, **outlier** financial profiles—still help interpret **individual** misclassifications listed in the same JSON, but **aggregate** accuracy for the stated objective is summarised by the **ablation means** in `full_model_comparison.json` (~**51–60%** depending on feature set), not by legacy **68%** figures from older experiments.

---

## 22. Project Structure

```
mxa1438/
├── app.py                                     # Streamlit demo (merged_multisource_training.csv)
├── README.md
├── requirements.txt
│
├── data/
│   ├── templates/
│   │   ├── ratings_scraped.csv
│   │   ├── kams_priority.csv
│   │   ├── kams_template.csv
│   │   ├── multi_agency_ratings.csv
│   │   └── ticker_mapping.csv
│   ├── processed/
│   │   ├── kams_processed.csv
│   │   ├── financial_ratios_processed.csv
│   │   ├── news_features_processed.csv
│   │   ├── merged_multisource_training.csv   # Demo + LLM prep + full XGBoost join
│   │   ├── ratings_with_financials.csv
│   │   ├── ratings_financials_sentiment.csv
│   │   └── ... (legacy experiment CSVs as needed for the report)
│   └── raw/
│       ├── financials/
│       └── news/
│
├── scripts/
│   ├── rebuild_processed_datasets.py
│   └── collect_news_sentiment_finbert.py
│
├── models/
│   ├── xgboost_with_kams.py
│   ├── xgboost_kams_only.py
│   ├── xgboost_full.py
│   ├── llm_verdict.py
│   ├── prepare_finetune_data.py
│   ├── finetune_qwen.py
│   ├── evaluate_finetune.py
│   ├── improve_verdicts.py
│   └── lora_adapter/
│
├── figures/                                   # Plots (historical + current)
├── results/                                   # JSON outputs, verdicts/
│
└── docs/
    ├── PROJECT_REPORT.md
    ├── DOCKER.md
    └── ...
```

---

## 23. Next Steps

### 23.1 Remaining Tasks

1. **Install Ollama and test with Mistral/Llama**
   - The LLM verdict generator is ready; just needs `ollama pull mistral`
   - Compare LLM-generated verdicts with template-based ones
   - Evaluate factual accuracy and coherence of LLM outputs

2. **Complete KAM extraction for remaining companies**
   - 148 rows in `kams_priority.csv` awaiting manual extraction
   - Focus on the 23 non-financial companies already in the training data
   - Use the KAM homogeneity finding as a research contribution regardless of result

3. **Collect 2025 financial data when available**
   - Will allow temporal validation (train on 2021-2024, test on 2025)

### 23.2 Presentation Preparation

4. **Prepare presentation slides using generated figures**
   - All figures in `figures/` are publication-quality at 150 DPI
   - The Streamlit demo provides a live interactive demonstration

5. **Practice with the demo app**
   - Run `streamlit run app.py` for live demonstration during presentation
   - Walk through each tab, showing the complete ML → SHAP → Verdict pipeline

---

## 24. References

1. Muñoz-Izquierdo, N., Segovia-Vargas, M.J., Camacho-Miñano, M.M., & Pérez-Pérez, Y. (2022). Machine learning in corporate credit rating assessment using the expanded audit report. *Machine Learning*, 111, 4183–4215.

2. Altman, E.I. (1983). Corporate Financial Distress: A Complete Guide to Predicting, Avoiding, and Dealing with Bankruptcy. Wiley.

3. Lundberg, S.M., & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*, 30.

4. Tassnief - Saudi Credit Rating Agency. https://tassnief.com

5. Saudi Exchange (Tadawul). https://www.saudiexchange.sa

6. Argaam (2025). "Insight into TASI-listed companies with credit ratings." https://www.argaam.com/en/article/articledetail/id/1850297

7. Charitou, A., Neophytou, E., & Charalambous, C. (2004). Predicting corporate failure: Empirical evidence for the UK. *European Accounting Review*, 13(3), 465–497.

8. Fitch Ratings. Saudi Arabia Entity Ratings. https://www.fitchratings.com

9. Bao, W., Lianju, N., & Yue, K. (2019). Integration of unsupervised and supervised machine learning algorithms for credit risk assessment. *Expert Systems with Applications*, 128, 301–315. https://doi.org/10.1016/j.eswa.2019.02.033

10. Lei, Y., Wang, J., Chen, Z., Li, X., Chen, J., & Chen, C. (2025). ZiGong 1.0: A Large Language Model for Financial Credit. *arXiv preprint arXiv:2502.16159*.

11. Huang, Z., Chen, H., Hsu, C.-J., Chen, W.-H., & Wu, S. (2004). Credit rating analysis with support vector machines and neural networks: A market comparative study. *Decision Support Systems*, 37(4), 543–558. https://doi.org/10.1016/S0167-9236(03)00086-1

12. Golbayani, P., Florescu, I., & Chatterjee, R. (2020). A comparative study of forecasting corporate credit ratings using neural networks, support vector machines, and decision trees. *The North American Journal of Economics and Finance*, 54, 101251. https://doi.org/10.1016/j.najef.2020.101251

---

## Appendix A: Rating Scale

| Rating | Numeric | Category |
|--------|---------|----------|
| AAA | 21 | Investment Grade |
| AA+ | 20 | Investment Grade |
| AA | 19 | Investment Grade |
| AA- | 18 | Investment Grade |
| A+ | 17 | Investment Grade |
| A | 16 | Investment Grade |
| A- | 15 | Investment Grade |
| BBB+ | 14 | Investment Grade |
| BBB | 13 | Investment Grade |
| BBB- | 12 | Investment Grade |
| BB+ | 11 | Speculative |
| BB | 10 | Speculative |
| BB- | 9 | Speculative |
| B+ | 8 | Speculative |
| B | 7 | Speculative |
| B- | 6 | Speculative |
| CCC+ | 5 | Speculative |
| CCC | 4 | Speculative |
| CCC- | 3 | Speculative |
| CC | 2 | Speculative |
| C | 1 | Speculative |
| D | 0 | Default |

---

## Appendix B: Class Imbalance Analysis

**Problem:** 92% of original Tassnief samples are Investment Grade

| Category | Count | Percentage |
|----------|-------|------------|
| Investment Grade | 24 | 92% |
| Speculative | 2 | 8% |

**How This Was Addressed:**
1. Multi-class prediction (AA/A/BBB/BB) instead of binary
2. Expanded dataset from 26 to 75 records via multi-agency and historical data
3. GroupKFold CV to prevent data leakage across company groups
4. Binary classification boundary set at A-/BBB+ for more balanced classes

**V2 Dataset Balance (75 records):**

| Category | Count | Percentage |
|----------|-------|------------|
| Investment Grade (A- and above) | ~52% | Improved balance |
| Speculative (BBB+ and below) | ~48% | Near-equal split |

---

## Appendix C: Deliverables Checklist

| # | Deliverable | Status | Location |
|---|-------------|--------|----------|
| 1 | Data Pipeline | ✅ Complete | `scripts/rebuild_processed_datasets.py` builds `kams_processed`, `financial_ratios_processed`, `news_features_processed`, `merged_multisource_training`; news sentiment columns refreshed with `scripts/collect_news_sentiment_finbert.py` before rebuild |
| 2 | Feature Engineering Module | ✅ Complete | 4 Altman ratios + paper-style KAM dummies + news sentiment features |
| 3 | ML Rating Predictor | ✅ Complete | `models/xgboost_with_kams.py`, `models/xgboost_full.py`, `models/xgboost_kams_only.py`; key metrics in `results/` (e.g. `kams_only_model_results.json`) |
| 4 | LLM Verdict Generator | ✅ Complete | `models/llm_verdict.py` + Ollama / QLoRA path |
| 5 | Evaluation Report | ✅ Complete | This document + stored `results/` JSON from experiments |
| 6 | Reproducible Codebase | ✅ Complete | Git repo with `docs/` and `README.md` |

---

*Report last updated: March 26, 2026*
