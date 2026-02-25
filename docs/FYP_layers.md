# Credit Rating Prediction for Saudi Exchange Companies (2021–2024)
## Classical ML + LLM Verdict Generation

---

## Definition Layer

### Aim
Evaluate whether classical machine learning models can predict credit ratings for Saudi listed companies using multi-source public data (financials, KAMs, news), with open-source LLMs providing qualitative verdict explanations.

### Scope
- **Companies**: Saudi Exchange (Tadawul) main-market firms with Tassnief credit ratings
- **Period**: Fiscal years 2021–2024
- **Data**: Public filings (English), news, Key Audit Matters (KAMs), Tassnief ratings
- **Exclusions**: Non-public data sources (banks, terminals, paid APIs)

### Research Questions

1. **RQ1**: Can a classical ML model (XGBoost/Random Forest) predict Tassnief credit ratings from public financial ratios, KAM features, and news sentiment?

2. **RQ2**: Which feature categories (financials, KAMs, news) contribute most to rating prediction accuracy?

3. **RQ3**: Can open-source LLMs generate coherent, factually accurate credit verdicts that explain the predicted ratings?

4. **RQ4**: How does model prediction accuracy compare across rating categories (investment grade vs speculative)?

### Deliverables

1. **Data Pipeline**: Automated collection and preprocessing of financials, KAMs, news, and Tassnief ratings
2. **Feature Engineering Module**: Extraction of financial ratios, KAM features, and news sentiment features
3. **ML Rating Predictor**: Trained classifier predicting Tassnief rating categories
4. **LLM Verdict Generator**: Open-source LLM system producing structured credit verdicts
5. **Evaluation Report**: Prediction accuracy metrics + verdict quality assessment
6. **Reproducible Codebase**: Complete documentation and reproducible pipeline

### Constraints

- Public data only; no paid financial terminals or private credit data
- Open-source LLMs only (Mistral, Llama, Falcon) for verdict generation
- Fixed temporal coverage (2021–2024)
- All verdicts must cite specific data points from input features
- Ethical compliance: no personal data, respect robots.txt for scraping

---

## Data and Infrastructure Layer

### Data Sources

| Source | Type | Purpose | Collection Method |
|--------|------|---------|-------------------|
| **Tadawul Filings** | PDFs, HTML, XLS | Financial statements, balance sheet, income statement | Web scraping (requests, BeautifulSoup) |
| **Key Audit Matters** | Text (from audit reports) | Risk indicators from auditor's perspective | Extracted from annual reports |
| **News** | Text | Sentiment, events, market perception | NewsAPI / MarketAux API |
| **Market Data** | Numeric | Price, volume, market cap | yfinance |
| **Tassnief Ratings** | Categorical | Ground truth labels | tassnief.com (manual/scraping) |

### Data Inventory Schema

```json
{
  "companies": [
    {
      "company_name": "Company Name",
      "ticker": "XXXX.SR",
      "sector": "Banking",
      "tassnief_rated": true
    }
  ],
  "filings": [...],
  "ratings": [
    {
      "ticker": "XXXX.SR",
      "rating_date": "2023-06-15",
      "rating": "A+",
      "outlook": "Stable",
      "rating_type": "issuer",
      "source_url": "https://tassnief.com/..."
    }
  ],
  "news": [
    {
      "ticker": "XXXX.SR",
      "headline": "...",
      "source": "...",
      "published_date": "2023-03-10",
      "sentiment": null
    }
  ]
}
```

### Collection Pipeline

1. **Financials**: Download Tadawul filings → parse tables → extract line items
2. **KAMs**: Extract auditor's report section → identify KAM paragraphs → categorize
3. **News**: Query NewsAPI/MarketAux by company name/ticker → store articles
4. **Market Data**: yfinance API → daily/quarterly prices
5. **Tassnief Ratings**: Collect from tassnief.com → map to company-period

### Preprocessing

| Data Type | Processing Steps |
|-----------|------------------|
| **Financials** | Table parsing (pdfplumber), numeric normalization, ratio calculation |
| **KAMs** | Text extraction, categorization (going concern, impairment, revenue recognition, etc.) |
| **News** | Sentiment analysis, keyword extraction, event tagging |
| **Ratings** | Ordinal encoding, rating bucket mapping |

### Schema Design

**Company-Period Record** (one row per company per fiscal year):
- Identifiers: company_name, ticker, fiscal_year
- Financial ratios: current_ratio, debt_equity, interest_coverage, ROA, ROE, net_margin, etc.
- KAM features: kam_count, kam_going_concern, kam_impairment, kam_revenue_recognition, etc.
- News features: news_count, avg_sentiment, negative_event_count, positive_event_count
- Market features: avg_price, volatility, market_cap
- Target: tassnief_rating, rating_bucket (investment_grade / speculative)

### Infrastructure

- **Storage**: SQLite/DuckDB for structured data; filesystem for raw filings
- **Processing**: Python (pandas, scikit-learn, XGBoost)
- **LLM Inference**: Ollama / HuggingFace Transformers for local open-source models
- **Environment**: Conda environment with pinned dependencies
- **Logging**: loguru for pipeline logs

### Data Quality

- Schema validation with Pandera/Pydantic
- Missing value handling rules
- Temporal alignment checks (rating date vs fiscal year)
- News date range validation (within fiscal year)

---

## Modeling Layer

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
├─────────────┬─────────────┬─────────────┬─────────────────────┤
│  Financials │    KAMs     │    News     │  Tassnief Ratings   │
│  (Tadawul)  │  (Filings)  │ (NewsAPI)   │  (Ground Truth)     │
└──────┬──────┴──────┬──────┴──────┬──────┴──────────┬──────────┘
       │             │             │                 │
       ▼             ▼             ▼                 │
┌─────────────────────────────────────────────┐     │
│           FEATURE EXTRACTION                │     │
│  • Financial ratios (15-20 features)        │     │
│  • KAM categorical features (5-10 features) │     │
│  • News sentiment features (5-10 features)  │     │
└──────────────────┬──────────────────────────┘     │
                   │                                 │
                   ▼                                 │
┌─────────────────────────────────────────────┐     │
│         FEATURE VECTOR                      │     │
│  [financial_ratios | kam_features |         │     │
│   news_features | market_features]          │     │
│  ~30-50 features per company-period         │     │
└──────────────────┬──────────────────────────┘     │
                   │                                 │
                   ▼                                 │
┌─────────────────────────────────────────────┐     │
│         ML CLASSIFIER                       │     │
│  • XGBoost / Random Forest                  │     │
│  • Target: Tassnief rating category         │◄────┘
│  • Output: Predicted rating + probabilities │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│         LLM VERDICT GENERATOR               │
│  • Input: Features + ML prediction          │
│  • Model: Mistral / Llama (open-source)     │
│  • Output: Structured credit verdict        │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│              EVALUATION                      │
│  • ML: Accuracy, F1, ordinal correlation   │
│  • Verdict: Factual accuracy, coherence    │
└─────────────────────────────────────────────┘
```

### Feature Engineering

#### Financial Ratios (from Tadawul filings)
| Feature | Formula | Category |
|---------|---------|----------|
| current_ratio | Current Assets / Current Liabilities | Liquidity |
| quick_ratio | (Current Assets - Inventory) / Current Liabilities | Liquidity |
| debt_to_equity | Total Debt / Total Equity | Leverage |
| debt_to_assets | Total Debt / Total Assets | Leverage |
| interest_coverage | EBIT / Interest Expense | Coverage |
| debt_service_coverage | Operating Cash Flow / Total Debt Service | Coverage |
| return_on_assets | Net Income / Total Assets | Profitability |
| return_on_equity | Net Income / Total Equity | Profitability |
| net_margin | Net Income / Revenue | Profitability |
| operating_margin | Operating Income / Revenue | Profitability |
| asset_turnover | Revenue / Total Assets | Efficiency |
| revenue_growth | (Revenue_t - Revenue_t-1) / Revenue_t-1 | Growth |

#### KAM Features (from audit reports)
| Feature | Type | Description |
|---------|------|-------------|
| kam_count | Integer | Total number of KAMs |
| kam_going_concern | Binary | Going concern issue flagged |
| kam_impairment | Binary | Asset impairment KAM present |
| kam_revenue_recognition | Binary | Revenue recognition KAM present |
| kam_related_party | Binary | Related party transactions KAM |
| kam_litigation | Binary | Litigation/contingencies KAM |
| kam_severity_score | Float | Weighted severity (0-1) |

#### News Features (from NewsAPI/MarketAux)
| Feature | Type | Description |
|---------|------|-------------|
| news_count | Integer | Total articles in fiscal year |
| avg_sentiment | Float | Average sentiment score (-1 to 1) |
| sentiment_std | Float | Sentiment volatility |
| negative_news_ratio | Float | % negative articles |
| positive_news_ratio | Float | % positive articles |
| event_lawsuit | Integer | Lawsuit/legal mentions count |
| event_expansion | Integer | Expansion/growth mentions count |
| event_regulatory | Integer | Regulatory issue mentions count |

### ML Model

**Primary Model**: XGBoost Classifier
- Handles mixed feature types well
- Provides feature importance
- Robust to missing values
- Good performance on tabular data

**Baseline Models** (for comparison):
- Logistic Regression (interpretable baseline)
- Random Forest (ensemble alternative)

**Target Variable**:
- Multi-class: Full Tassnief scale (AAA, AA+, AA, AA-, A+, A, A-, BBB+, ...)
- Binary: Investment Grade (BBB- and above) vs Speculative (BB+ and below)

**Training Setup**:
- Train/validation/test split by time (avoid leakage)
- Cross-validation for hyperparameter tuning
- Class weighting for imbalanced ratings

### LLM Verdict Generator

**Purpose**: Generate human-readable credit assessment explaining the ML prediction

**Input to LLM**:
```
Company: {company_name} ({ticker})
Period: FY {fiscal_year}

=== FINANCIAL RATIOS ===
Current Ratio: {current_ratio}
Debt/Equity: {debt_equity}
Interest Coverage: {interest_coverage}
ROA: {roa}%
Net Margin: {net_margin}%
...

=== KEY AUDIT MATTERS ===
{kam_summary}

=== NEWS SUMMARY ===
{news_summary}

=== ML PREDICTION ===
Predicted Rating: {predicted_rating}
Confidence: {confidence}%
Actual Tassnief Rating: {actual_rating} (if available)

=== TASK ===
Provide a structured credit verdict as JSON with:
1. overall_assessment: 1-2 sentence summary
2. strengths: list of positive factors with supporting data
3. weaknesses: list of risk factors with supporting data
4. risk_factors: key items to monitor
5. prediction_analysis: does the data support the prediction?
```

**Output Schema**:
```json
{
  "company": "...",
  "period": "FY 2023",
  "overall_assessment": "...",
  "strengths": ["Strong liquidity (CR: 1.8)", "..."],
  "weaknesses": ["Elevated leverage (D/E: 2.1)", "..."],
  "risk_factors": ["Debt refinancing in 2024", "..."],
  "prediction_analysis": "The BBB+ prediction aligns with..."
}
```

**Models**: Mistral 7B, Llama 3 8B (via Ollama or HuggingFace)

---

## Evaluation Layer

### ML Model Evaluation

| Metric | Description | Target |
|--------|-------------|--------|
| **Accuracy** | % correct predictions | >60% (multi-class) |
| **Macro F1** | Average F1 across classes | >0.5 |
| **Weighted F1** | Class-weighted F1 | >0.6 |
| **Spearman Correlation** | Ordinal ranking correlation | >0.7 |
| **Investment Grade F1** | Binary classification F1 | >0.75 |
| **Confusion Matrix** | Error analysis by rating | - |

### Feature Importance Analysis

- SHAP values for global feature importance
- Per-category analysis (financials vs KAMs vs news)
- Ablation study: model performance with each feature category removed

### LLM Verdict Evaluation

| Metric | Description | Method |
|--------|-------------|--------|
| **Factual Accuracy** | Are cited numbers correct? | Automated check against input |
| **Completeness** | Does verdict cover key factors? | Rubric-based scoring |
| **Coherence** | Is reasoning logically consistent? | Manual review |
| **Alignment** | Does verdict support prediction? | Automated + manual |
| **Citation Rate** | % claims with data backing | Automated parsing |

### Comparison Framework

1. **ML vs Tassnief**: How well does the model predict actual ratings?
2. **Feature Ablation**: Which features matter most?
3. **Verdict Quality**: Are LLM explanations useful and accurate?
4. **Error Analysis**: Where and why does the model fail?

---

## Project Management Layer

### Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 1: Data Collection** | - | Financials, KAMs, news, ratings collected |
| **Phase 2: Feature Engineering** | - | Feature extraction pipeline, clean dataset |
| **Phase 3: ML Development** | - | Trained classifier, baseline evaluation |
| **Phase 4: LLM Integration** | - | Verdict generator, prompt templates |
| **Phase 5: Evaluation** | - | Full evaluation report, analysis |
| **Phase 6: Documentation** | - | Final report, reproducible codebase |

### Milestones

- [ ] Data inventory populated with all companies and sources
- [ ] Raw data collected (filings, news, ratings)
- [ ] Feature extraction pipeline operational
- [ ] ML model trained and evaluated
- [ ] LLM verdict generator producing valid outputs
- [ ] Evaluation report completed
- [ ] Final repository and documentation delivered

### Risk Management

| Risk | Mitigation |
|------|------------|
| Limited Tassnief coverage | Focus on rated companies only; use rating buckets |
| News data sparsity | Aggregate by company-year; use multiple news sources |
| Class imbalance in ratings | Class weighting, SMOTE, focus on binary classification |
| KAM extraction errors | Manual validation of extraction rules |
| LLM hallucination in verdicts | Schema enforcement, factual accuracy checks |

### Version Control

- Git for code and documentation
- DVC for data versioning (optional)
- Tagged releases for milestones
- Experiment logging for model runs

---

## Appendix: Rating Scale Reference

### Tassnief Rating Scale

| Rating | Category | Description |
|--------|----------|-------------|
| AAA | Investment Grade | Highest credit quality |
| AA+, AA, AA- | Investment Grade | Very high credit quality |
| A+, A, A- | Investment Grade | High credit quality |
| BBB+, BBB, BBB- | Investment Grade | Good credit quality |
| BB+, BB, BB- | Speculative | Speculative |
| B+, B, B- | Speculative | Highly speculative |
| CCC+, CCC, CCC- | Speculative | Substantial risk |
| CC, C | Speculative | Extremely speculative |
| D | Default | In default |

### Rating Bucket Mapping

For binary classification:
- **Investment Grade**: AAA, AA+, AA, AA-, A+, A, A-, BBB+, BBB, BBB-
- **Speculative**: BB+ and below
