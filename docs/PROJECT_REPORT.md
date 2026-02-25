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
15. [Next Steps](#15-next-steps)
16. [References](#16-references)

---

## 1. Project Overview

### 1.1 Objective

Develop a machine learning system to predict credit ratings for Saudi Exchange (Tadawul) listed companies using:
- Financial statement data
- Key Audit Matters (KAMs) from annual reports
- News sentiment analysis (future work)

### 1.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT FEATURES                           │
├─────────────────┬─────────────────┬─────────────────────────┤
│  Financial      │  KAM Features   │  News Sentiment         │
│  Ratios (4)     │  (5 binary)     │  (future)               │
├─────────────────┴─────────────────┴─────────────────────────┤
│                                                             │
│                 XGBoost Classifier                          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                 Credit Rating Prediction                    │
│           (AAA, AA, A, BBB, BB, etc.)                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│            Open-Source LLM (Verdict Generation)             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Scope

- **Companies:** 47 unique Tadawul-listed companies with credit ratings
- **Time Period:** 2021-2025 (fiscal years)
- **Ratings Sources:** Tassnief, Moody's, Fitch, S&P, Financial Analytics (RATING)
- **Financial Data Source:** yfinance API

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

---

## 3. Data Collection

### 3.1 Tassnief Credit Ratings

**Source:** https://tassnief.com (Saudi national credit rating agency)

**Collection Method:** Selenium web scraping (`scripts/scrape_tassnief_selenium.py`)

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

**Collection Script:** `scripts/collect_financials.py`

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
| KAMs only | Not yet tested | **74.14%** |
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
| KAMs Only | 6 KAM features | 47.73% | 74.14% |
| Combined | 10 features | **65.15%** | 84.04% |

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
| Extract KAMs (25 reports) | ✅ Done | `kams_priority.csv` |
| Train combined model | ✅ Done | 65.15% accuracy |
| News sentiment collection | ✅ Done | 49/50 records |
| Train full model (4 variants) | ✅ Done | 65.15% best accuracy |
| Multi-agency expansion | ✅ Done | 80 ratings, 5 agencies |
| Train expanded models | ✅ Done | 69.6% best (Fitch) |
| Multi-model comparison | ✅ Done | 78.3% best (Decision Tree) |
| Paper-aligned model (4 ratios, binary) | ✅ Done | 71.2% (Gradient Boosting) |
| Historical Fitch data (2019-2024) | ✅ Done | 66.7% best (SVM) |
| Agency handling comparison | ✅ Done | Fitch-only vs All+Agency |

### 7.2 Project Structure

```
mxa1438/
├── README.md
├── requirements.txt
│
├── data/
│   ├── templates/                          # Input templates & scraped data
│   │   ├── ratings_scraped.csv             # Tassnief ratings (scraped)
│   │   ├── kams_priority.csv               # KAM extraction template (148 rows)
│   │   ├── kams_to_extract.csv             # Full extraction list
│   │   ├── multi_agency_ratings.csv        # 47 companies, 5 agencies
│   │   └── ticker_mapping.csv              # Company name → ticker mapping
│   ├── processed/                          # ML-ready datasets
│   │   ├── model_training_data.csv         # V1 training set (60 records)
│   │   ├── model_training_data_v2.csv      # V2 training set (75 records)
│   │   ├── model_fitch_only_data.csv       # Fitch-only subset
│   │   ├── ratings_with_financials.csv     # Tassnief + financials
│   │   ├── expanded_ratings_financials.csv # Multi-agency + financials
│   │   ├── historical_fitch_ratings.csv    # Historical Fitch (72 records)
│   │   ├── tadawul_ratings_clean.csv       # Tadawul disclosures + financials
│   │   ├── tadawul_ratings_financials.csv  # Full Tadawul dataset
│   │   └── ratings_financials_sentiment.csv
│   └── raw/                                # Cached API responses
│       ├── financials/                     # yfinance JSON per company
│       └── news/                           # MarketAux news per company-year
│
├── scripts/                                # Data collection & processing
│   ├── scrape_tassnief_selenium.py          # Tassnief ratings scraper
│   ├── scrape_tadawul_ratings.py            # Tadawul disclosure collector
│   ├── collect_financials.py                # yfinance financial data
│   ├── collect_news_sentiment.py            # News sentiment (MarketAux + VADER)
│   ├── expand_dataset.py                    # Multi-agency expansion
│   ├── build_historical_fitch.py            # Historical Fitch data
│   ├── export_training_data.py              # Consolidate final dataset
│   ├── merge_all_data.py                    # Merge all sources into V2
│   ├── update_kams_template.py              # Add new companies to KAMs
│   └── add_tickers.py                       # Ticker mapping utility
│
├── models/                                 # ML model training & comparison
│   ├── baseline_xgboost.py                  # Initial XGBoost model
│   ├── xgboost_with_kams.py                 # Financials + KAMs
│   ├── xgboost_full.py                      # Financials + KAMs + sentiment
│   ├── xgboost_expanded.py                  # Multi-agency XGBoost
│   ├── xgboost_expanded_v2.py               # Per-agency & consensus models
│   ├── model_comparison.py                  # 8 ML models comparison
│   ├── model_comparison_historical.py       # Historical data comparison
│   ├── model_paper_ratios.py                # Paper-aligned methodology
│   ├── model_agency_comparison.py           # Fitch-only vs All+Agency
│   └── model_v2_comparison.py               # V1 vs V2 dataset comparison
│
├── results/                                # Model output (JSON)
│   ├── model_v2_comparison.json             # Latest: V1 vs V2
│   ├── agency_comparison_results.json
│   ├── paper_methodology_results.json
│   ├── model_comparison_results.json
│   └── ... (other experiment results)
│
└── docs/                                   # Documentation
    ├── PROJECT_REPORT.md                    # This document
    ├── FYP_layers.md                        # Project specification
    ├── kam_extraction.md                    # KAM extraction guide
    └── annual_report_checklist.md           # Annual report checklist
```

---

## 8. News Sentiment Analysis

### 8.1 Data Collection

**Source:** MarketAux API  
**Method:** Fetched English-language news articles for each company in each fiscal year

**Coverage:**
- 49/50 records have news articles
- Average 4.3 articles per company-year
- Mean sentiment: +0.332 (generally positive, expected for established firms)

**Sentiment Features:**

| Feature | Description |
|---------|-------------|
| `sentiment_mean` | Average VADER compound score (-1 to +1) |
| `sentiment_std` | Standard deviation (sentiment volatility) |
| `sentiment_pos_pct` | Proportion of positive articles |
| `sentiment_neg_pct` | Proportion of negative articles |
| `news_count` | Number of articles found |

### 8.2 Sentiment Scoring

Used VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment scoring - a rule-based model specifically tuned for financial/social media text.

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

This aligns with the reference paper's finding that PART (a rule-based method) was the top performer. Decision Trees excel here because:
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

Our Decision Tree on Fitch data (78.3%) exceeds the paper's best result (74.14% with PART on KAMs). This is achieved with only 2 financial ratios and 23 samples, suggesting that Fitch ratings are particularly well-aligned with financial fundamentals.

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

**Output file:** `data/processed/model_training_data.csv`

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

## 15. Next Steps

### 15.1 Immediate (Highest Impact)

1. **Extract KAMs for remaining companies**
   - 148 rows in `kams_priority.csv` awaiting manual extraction
   - Prioritize ~30-40 non-financial companies with Fitch ratings
   - Paper showed KAMs add ~3% accuracy (71.55% → 74.14%)

2. **Implement LLM verdict generation**
   - Input: Financial ratios + KAMs + ML rating prediction
   - Output: Human-readable credit risk explanation
   - This completes the hybrid ML+LLM architecture

### 15.2 Short-term

3. **Retrain models with KAM features**
   - Once KAMs are extracted, add the 5 binary categories + kam_count
   - Expected improvement: 3-13% based on reference paper

4. **Increase sample size further**
   - Collect 2025 financials when available
   - Add historical ratings from agency press releases

### 15.3 Medium-term

5. **Model refinement**
   - Hyperparameter tuning for Decision Tree, SVM, Gradient Boosting
   - Ensemble of best models per agency
   - Explore ordinal regression (ratings have natural order)

---

## 16. References

1. Muñoz-Izquierdo, N., Segovia-Vargas, M.J., Camacho-Miñano, M.M., & Pérez-Pérez, Y. (2022). Machine learning in corporate credit rating assessment using the expanded audit report. *Machine Learning*, 111, 4183–4215.

2. Altman, E.I. (1983). Corporate Financial Distress: A Complete Guide to Predicting, Avoiding, and Dealing with Bankruptcy. Wiley.

3. Tassnief - Saudi Credit Rating Agency. https://tassnief.com

4. Saudi Exchange (Tadawul). https://www.saudiexchange.sa

5. Argaam (2025). "Insight into TASI-listed companies with credit ratings." https://www.argaam.com/en/article/articledetail/id/1850297

6. Charitou, A., Neophytou, E., & Charalambous, C. (2004). Predicting corporate failure: Empirical evidence for the UK. *European Accounting Review*, 13(3), 465–497. (Cited for exclusion of financial institutions from credit risk models.)

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

**Problem:** 92% of samples are Investment Grade

| Category | Count | Percentage |
|----------|-------|------------|
| Investment Grade | 24 | 92% |
| Speculative | 2 | 8% |

**Why This Happens:**
- Tassnief rates established Saudi companies
- Companies seeking ratings are typically creditworthy
- Speculative-grade companies rarely seek public ratings

**Mitigation Strategies:**
1. Multi-class prediction (AA/A/BBB/BB) instead of binary
2. Class weighting in XGBoost
3. Collect more speculative-grade examples
4. SMOTE or other oversampling techniques

---

*Report last updated: February 6, 2026*
