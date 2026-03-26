# Credit Risk Modelling: Current Multisource Pipeline (Saudi Tadawul)

**Author:** MXA1438  
**Report file:** `docs/PROJECT_REPORT_CURRENT.md`  
**Scope:** This document describes **only** the pipeline, data files, code, and **numerical results** that exist in the repository **as of the timestamps embedded in the cited JSON artifacts**. It does **not** restate older experiments, removed scripts, or paths that no longer appear in the tree. For a longer historical narrative, see `docs/PROJECT_REPORT.md` (may reference retired files).

**Primary results sources (ground truth for this report):**

- `results/full_model_comparison.json`
- `results/combined_model_results.json`
- `results/kams_only_model_results.json`
- `results/multisource_model_comparison.json`
- `results/shap_report.json`
- `results/error_analysis.json`
- `results/verdicts/verdict_summary.json`
- `results/pipeline_figures_meta.json`

---

## Table of contents

1. [Summary](#1-summary)  
2. [Literature review and cited foundations](#2-literature-review-and-cited-foundations)  
3. [Data in this repository](#3-data-in-this-repository)  
4. [Target and features](#4-target-and-features)  
5. [Methods and code](#5-methods-and-code)  
6. [Results: XGBoost feature ablations](#6-results-xgboost-feature-ablations)  
7. [Results: financials + full KAM block on the merged sample](#7-results-financials--full-kam-block-on-the-merged-sample)  
8. [Results: KAM-only model on the full KAM panel](#8-results-kam-only-model-on-the-full-kam-panel)  
9. [Results: multisource classifier benchmark](#9-results-multisource-classifier-benchmark)  
10. [SHAP explainability](#10-shap-explainability)  
11. [Error analysis and diagnostic figures](#11-error-analysis-and-diagnostic-figures)  
12. [LLM verdict layer](#12-llm-verdict-layer)  
13. [Figures present in `figures/`](#13-figures-present-in-figures)  
14. [Reproducing outputs](#14-reproducing-outputs)  
15. [Repository layout (essential paths)](#15-repository-layout-essential-paths)  
16. [References](#16-references)

---

## 1. Summary

The project predicts **four-way rating categories** (AA, A, BBB, BB) for Tadawul-listed firms using a **merged multisource table**: four Altman-style financial ratios, five Key Audit Matter (KAM) dummies aligned with Muñoz-Izquierdo et al. (2022), and five FinBERT-based news aggregates. The **main modelling sample** after dropping rows with missing `profitab` contains **46** `(ticker, fiscal_year)` rows (`results/full_model_comparison.json`). The **best in-repo XGBoost configuration** on that sample reaches **about 71.8%** stratified 5-fold cross-validated accuracy when financials, KAM dummies, and news features are combined. A **broader algorithm comparison** on the same matrix ranks **XGBoost first** among ten sklearn-style baselines (`results/multisource_model_comparison.json`). **SHAP** analysis and **template LLM verdicts** are wired to this same panel.

---

## 2. Literature review and cited foundations

This section carries over the **substantive text** from `docs/PROJECT_REPORT.md` that motivates the design and ties it to the bibliography (Section 16). **Empirical results** for the current pipeline are still reported only in Sections 6–12 and in the JSON files listed in the header—not in the tables below, which summarise the **reference paper** or **prior literature**.

### 2.1 Primary reference: Muñoz-Izquierdo et al. (2022)

**Muñoz-Izquierdo, N., Segovia-Vargas, M.J., Camacho-Miñano, M.M., & Pérez-Pérez, Y. (2022).** *Machine learning in corporate credit rating assessment using the expanded audit report.* *Machine Learning*, 111, 4183–4215. https://doi.org/10.1007/s10994-022-06226-4

**Key findings reported in that paper:**

| Model | Features used | Accuracy (paper) |
|-------|---------------|------------------|
| KAMs only | 5 KAM categories + audit variables | **74.14%** |
| Financial ratios only | 4 Altman Z''-Score ratios | 71.55% |
| Combined (KAMs + financials) | All features | **84.04%** |

**Paper methodology (summary):** Sample of 116 Spanish listed companies (2017); ML techniques included C4.5 Decision Tree, PART, Rough Set, and Logistic Regression; PART performed strongly for rule induction. **Key insight:** KAM disclosures alone carry substantial information for rating prediction.

Our Saudi multisource replication uses the **same style of inputs** (four Altman-style ratios + paper-style KAM dummies + extensions for news) but a **different market, period, and sample size**; **our** accuracies appear in `results/*.json`, not in the table above.

### 2.2 What are Key Audit Matters (KAMs)?

**Key Audit Matters (KAMs)** are paragraphs in audit reports where auditors highlight:

- Significant risks of material misstatement  
- Areas requiring significant management judgment  
- Critical accounting estimates  

KAMs became mandatory for many listed companies after the 2016 international audit reform (ISA 700).

### 2.3 KAM categories (mapping to features)

Following Muñoz-Izquierdo et al. (2022), we map categories to binary indicators used in modelling:

| Category | Variable | Description |
|----------|----------|-------------|
| Going concern | `GCKAM` | Viability doubts, refinancing risks, covenant breaches |
| Revenue | `REVKAM` | Revenue recognition complexity, contract accounting |
| Assets | `ASSETKAM` | Impairment, goodwill, fair value of assets |
| Liabilities | `LIABKAM` | Provisions, contingencies, debt issues |
| Other | `OTHERKAM` | IT systems, acquisitions, regulatory compliance |

### 2.4 Credit scoring and algorithm comparisons: Bao et al. (2019)

**Bao, W., Lianju, N., & Yue, K. (2019).** *Integration of unsupervised and supervised machine learning algorithms for credit risk assessment.* *Expert Systems with Applications*, 128, 301–315. https://doi.org/10.1016/j.eswa.2019.02.033

Bao et al. compare seven supervised ML algorithms (LR, SVM, DT, RF, GBDT, kNN, ANN) on credit-scoring datasets, including a Chinese P2P dataset of 23,435 samples with 16.6% default rate. Findings that inform our methodology:

| Finding in Bao et al. | Relevance here |
|------------------------|----------------|
| **GBDT best individual model** | Gradient boosting family (we use XGBoost and benchmark sklearn gradient boosting) achieved strong MCC among individual models in their study. |
| **Ensemble / hybrid designs** | Combining unsupervised structure with supervised learners improved MCC in their setup—motivation for rich feature sets rather than a single raw signal. |
| **MCC vs accuracy under imbalance** | They stress Matthews Correlation Coefficient where class balance is uneven; we report accuracy and F1 from stratified CV on our small multiclass panel (`results/multisource_model_comparison.json`). |
| **Model ranking (their data)** | GBDT > ANN > SVM > RF > kNN > LR > DT — broadly consistent with tree/boosting methods outperforming naive linear baselines on tabular credit-style data. |

Their use of k-fold CV, search, and hold-out evaluation parallels our use of **stratified k-fold CV** on the merged multisource matrix. The gap between their MCC on large samples and our **~72% accuracy on 46 rows** is expected from **small-sample variance** and a different task (multiclass rating categories vs default prediction).

### 2.5 Financial ratios: Altman Z''-style components (Altman, 1983)

**Altman, E.I. (1983).** *Corporate Financial Distress: A Complete Guide to Predicting, Avoiding, and Dealing with Bankruptcy.* Wiley.

We use the **four ratio definitions** aligned with Muñoz-Izquierdo et al. (2022) (Altman Z''-score style), implemented in the pipeline as `liquid`, `cumprof`, `profitab`, `leverage`:

| Ratio | Formula (conceptual) | Interpretation |
|-------|----------------------|----------------|
| **LIQUID** | (Current assets − Current liabilities) / Total assets | Short-term liquidity |
| **CUMPROF** | Retained earnings / Total assets | Cumulative profitability |
| **PROFITAB** | EBIT / Total assets | Return on assets (operating) |
| **LEVERAGE** | Book equity / Total liabilities | Capital structure / solvency |

**Why these four:** (1) Altman-style ratios are standard in distress and rating modelling; (2) scale-free comparison across firms; (3) **direct comparability** with the reference paper’s financial block; (4) parsimony under a **small** panel—see Sections 6–9 for empirical performance, not fixed rules such as “samples per feature” here.

### 2.6 Excluding banks and financial firms (Charitou et al., 2004)

**Charitou, A., Neophytou, E., & Charalambous, C. (2004).** Predicting corporate failure: Empirical evidence for the UK. *European Accounting Review*, 13(3), 465–497.

Following Muñoz-Izquierdo et al. (2022) and common practice in **non-financial** credit modelling, **banks and financial institutions** are treated separately from industrial firms when interpreting ratios:

**Financial statement structure.** Banks’ balance sheets are dominated by loans and deposits. The four ratios above are **misleading** if applied naively across sectors—for example, **LEVERAGE** (equity / liabilities) is intrinsically low for banks by design, and **PROFITAB** (EBIT / assets) can be tiny because assets are gross loan books.

**Regulatory environment.** Bank ratings depend on **capital adequacy** and supervisory frameworks (e.g. Basel) that do not apply to non-financial issuers; audit KAMs in banks also emphasise different risks (loan loss allowances, fair value of instruments) than typical industrial filers.

The **current processed training file** (`merged_multisource_training.csv`) reflects the project’s non-financial alignment choices; empirical counts are in the JSON summaries, not re-derived here.

### 2.7 Market and rating-agency sources (web references)

- **Tassnief** (Saudi Credit Rating Agency) — https://tassnief.com — primary **ground-truth** rating source for the Tadawul-focused panel; template extracts appear under `data/templates/ratings_scraped.csv`.  
- **Saudi Exchange (Tadawul)** — https://www.saudiexchange.sa — listing, disclosures, and issuer identifiers.  
- **Argaam** (2025), “Insight into TASI-listed companies with credit ratings” — https://www.argaam.com/en/article/articledetail/id/1850297 — market commentary and coverage context.  
- **Fitch Ratings** (Saudi entity ratings) — https://www.fitchratings.com — international agency perspective where used for comparison or narrative in the wider project; the **current** multisource metrics in this report are anchored on the merged processed CSVs, not on a separate Fitch-only table.

### 2.8 Explaining predictions: SHAP (Lundberg & Lee, 2017)

**Lundberg, S.M., & Lee, S.I. (2017).** A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*, 30.

Tree-based **gain importance** shows *which* features are used often in splits; **SHAP (SHapley Additive exPlanations)** adds:

- **Local** explanations (why this firm-year received a given predicted class)  
- **Global** attribution (average impact of each feature across observations)  
- **Interaction** views (e.g. dependence plots coloured by a second feature)  

We implement SHAP for the **14-feature** full XGBoost model in `models/shap_explainability.py`; outputs are summarised in Section 10 and `results/shap_report.json`.

### 2.9 Financial LLMs and QLoRA: Lei et al. (2025) / ZiGong

**Lei, Y., Wang, J., Chen, Z., Li, X., Chen, J., & Chen, C. (2025).** ZiGong 1.0: A Large Language Model for Financial Credit. *arXiv* arXiv:2502.16159.

**Template and zero-shot limits.** Rule-based verdicts are factually grounded but repetitive; general LLM prompting adds fluency but risks **hallucinated** numbers unless controlled.

**ZiGong insight.** Lei et al. show that **domain-specific fine-tuning** (LoRA on a financial credit task) can improve reliability of financial text generation relative to pure zero-shot use.

**Our adaptation.** We use **QLoRA** on **Qwen2.5-3B-Instruct** (structured JSON, smaller VRAM footprint than a 7B Mistral base). LoRA hyperparameters (e.g. rank 8, alpha 16) are chosen in the spirit of their reported setup. Instruction data are built from **`merged_multisource_training.csv`** and related fields via `models/prepare_finetune_data.py`; the adapter lives under `models/lora_adapter/`. Section 12 and `results/verdicts/verdict_summary.json` describe the **template** verdict run used in the current artefact refresh; fine-tuned outputs may appear in `results/verdicts/finetuned_verdicts.json` as a separate run.

---

## 3. Data in this repository

**Processed modelling inputs (CSV):**

| File | Role |
|------|------|
| `data/processed/kams_processed.csv` | KAM dummies and audit/firm controls; defines the KAM-aligned panel. |
| `data/processed/financials_processed.csv` | Financial ratios (`liquid`, `cumprof`, `profitab`, `leverage`) keyed to ticker × year. |
| `data/processed/news_features_processed.csv` | FinBERT aggregates and `news_count`. |
| `data/processed/merged_multisource_training.csv` | Inner-join style training table used by the app and multisource scripts (46 usable rows after ratio drops in the current pipeline). |
| `data/processed/ratings_financials_sentiment.csv` | Bridge / seed file for news pipeline alignment (see `docs/FINBERT_NEWS_SENTIMENT.md`). |

**Templates and reference tables under `data/templates/`** (including `ratings_scraped.csv`, `kams_priority.csv`, `ticker_mapping.csv`, `multi_agency_ratings.csv`, etc.) support collection and alignment; the **numerical results below** are tied to the **processed** files above, not to every template row.

**Raw inputs** under `data/raw/financials/` and `data/raw/news/` back the processed CSVs.

---

## 4. Target and features

- **Target:** `rating_category` with classes **A, AA, BB, BBB** (distribution in `results/full_model_comparison.json`: A 18, AA 6, BB 6, BBB 16 on the 46-row sample).  
- **Financial ratios (4):** `liquid`, `cumprof`, `profitab`, `leverage`.  
- **KAM dummies (5):** `GCKAM`, `REVKAM`, `ASSETKAM`, `LIABKAM`, `OTHERKAM`.  
- **News / FinBERT aggregates (5):** `sentiment_mean`, `sentiment_std`, `sentiment_pos_pct`, `sentiment_neg_pct`, `news_count`.  

The **full multisource vector** has **14** features (`results/multisource_model_comparison.json`).

**KAM-only XGBoost** (`results/kams_only_model_results.json`) uses **twelve** columns from `kams_processed.csv`: the five KAM dummies plus `AUSIZE`, `AUOP`, `EMP`, `GCUP`, `FIRMAGE`, `FIRMSIZE`, `INDUSTRY` on **47** rows (no financial merge).

---

## 5. Methods and code

- **Stratified 5-fold cross-validation**, `random_state=42` where specified in the multisource benchmark JSON.  
- **XGBoost** multiclass models with `eval_metric` / sampling consistent with `models/xgboost_full.py`, `models/xgboost_with_kams.py`, and `models/xgboost_kams_only.py`.  
- **SHAP:** `models/shap_explainability.py` → `results/shap_report.json` and SHAP PNGs.  
- **Benchmark:** `models/evaluate_multisource_models.py` → `results/multisource_model_comparison.json`, `figures/multisource_model_comparison.png`.  
- **Diagnostic plots:** `models/generate_pipeline_figures.py` → multiple `figures/*.png` and `results/error_analysis.json`.  
- **One-shot regeneration:** `scripts/regenerate_artifacts.py` (runs XGBoost scripts, benchmark, SHAP, pipeline figures, and optional template verdicts).  
- **Streamlit demo:** `app.py`.  
- **FinBERT documentation:** `docs/FINBERT_NEWS_SENTIMENT.md`.  

---

## 6. Results: XGBoost feature ablations

From `results/full_model_comparison.json` (generated `2026-03-26`, **n = 46**):

| Model variant | Mean CV accuracy | Std |
|---------------|------------------|-----|
| Financials only (4 ratios) | 56.89% | 0.098 |
| Financials + KAM dummies (9 features) | 65.11% | 0.086 |
| Financials + sentiment block (9 features) | 61.33% | 0.128 |
| **Full (14 features)** | **71.78%** | **0.087** |

Adding KAMs and news both improve on financials-only on this panel; the **full 14-feature** model is best among these four.

---

## 7. Results: financials + full KAM block on the merged sample

`results/combined_model_results.json` compares **financials-only** vs **financials + twelve KAM/audit columns** on the **same 46 rows** as the merged financial/KAM join (see `financials_source`: `financials_processed.csv`).

| Model | Mean CV accuracy | Std |
|-------|------------------|-----|
| Financials only | 63.33% | 0.097 |
| KAM-only on merged rows (12 feats) | 69.78% | 0.099 |
| **Combined (16 features)** | **78.22%** | **0.100** |

The JSON also records **paper reference** accuracies (0.7155 and 0.8404) for side-by-side comparison only.

---

## 8. Results: KAM-only model on the full KAM panel

`results/kams_only_model_results.json` (**n = 47**, all rows in `kams_processed.csv`):

- **Mean 5-fold CV accuracy:** **62.00%** (std **13.12%**).  
- **In-sample train accuracy:** 76.60%.  
- **Top gain-based importances (as stored):** `AUSIZE` 0.316, `REVKAM` 0.283, `INDUSTRY` 0.141, `OTHERKAM` 0.107, `FIRMAGE` 0.082, `ASSETKAM` 0.064.  

---

## 9. Results: multisource classifier benchmark

From `results/multisource_model_comparison.json` (**46 × 14**, same classes as above), ranked by mean CV accuracy:

| Rank | Model | Accuracy (mean ± std) | F1 macro (mean ± std) |
|------|--------|------------------------|------------------------|
| 1 | XGBoost | 71.78% ± 8.65% | 0.661 ± 0.112 |
| 2 | Random forest | 66.89% ± 12.58% | 0.533 ± 0.126 |
| 3 | Extra trees | 65.11% ± 11.10% | 0.553 ± 0.069 |
| 4 | sklearn `GradientBoostingClassifier` | 58.44% ± 9.10% | 0.515 ± 0.129 |
| 5 | Linear SVC (scaled pipeline) | 56.67% ± 8.89% | 0.587 ± 0.142 |
| 6 | kNN (k=5, scaled) | 54.67% ± 16.43% | 0.428 ± 0.149 |
| 7 | Decision tree | 54.22% ± 8.73% | 0.468 ± 0.156 |
| 8 | Logistic regression (scaled) | 52.00% ± 11.81% | 0.490 ± 0.176 |
| 9 | MLP (scaled) | 41.33% ± 4.35% | 0.181 ± 0.056 |
| 10 | `HistGradientBoostingClassifier` | 39.33% ± 6.35% | 0.140 ± 0.017 |

**Figure:** `figures/multisource_model_comparison.png`.

---

## 10. SHAP explainability

`results/shap_report.json` summarises **mean |SHAP|** for the **14-input** full XGBoost model (**n = 46**). Global ranking (highest first):

| Rank | Feature | Mean \|SHAP\| |
|------|---------|----------------|
| 1 | `news_count` | 0.537 |
| 2 | `cumprof` | 0.482 |
| 3 | `leverage` | 0.453 |
| 4 | `liquid` | 0.415 |
| 5 | `profitab` | 0.294 |
| 6 | `sentiment_std` | 0.173 |
| 7 | `sentiment_mean` | 0.144 |
| 8 | `REVKAM` | 0.107 |
| 9 | `ASSETKAM` | 0.034 |
| 10–14 | Other sentiment shares and KAM dummies | ≤ 0.032 |

**Outputs:** `figures/shap_beeswarm.png`, `figures/shap_bar_importance.png`, `figures/shap_dependence_*.png`, `figures/shap_waterfall_*.png` (per-row samples), plus the JSON above.

Interpretation is **associational**: high `news_count` may proxy for firm size, media coverage, or data collection intensity, not necessarily a causal driver of ratings.

---

## 11. Error analysis and diagnostic figures

`results/error_analysis.json` is generated for **multisource XGBoost** with **5-fold CV out-of-fold predictions** on the 46-row sample:

- **CV accuracy (stated in file):** **71.74%** (33 correct, 13 incorrect; note tiny rounding difference vs 71.78% in `full_model_comparison.json` due to estimator / prediction path).  
- **Content:** class list, confusion matrix, and a **misclassified** list with ticker, `company_name`, fiscal year, actual vs predicted category, and max predicted probability.

**Companion figures** (see `results/pipeline_figures_meta.json`): rating distribution, confusion matrix, PCA scatter, ratio boxplots, confidence histogram, error scatter and pattern plots—both `*_multisource.png` variants and legacy filenames (`pca_scatter.png`, `confusion_matrix_gb.png`, etc.) used by `app.py`.

---

## 12. LLM verdict layer

`models/llm_verdict.py` consumes the merged multisource rows and the multiclass XGBoost prediction.  

**Latest template run** (`results/verdicts/verdict_summary.json`, `2026-03-26`):

- **Verdicts written:** 46  
- **Method:** `template`  
- **Average overall quality score:** 99.5%  
- **Average citation rate:** 97.3%  
- **Average number accuracy:** 92.3%  
- **ML category accuracy (matches stored predictions):** 71.7%  

**Artifacts:** `results/verdicts/all_verdicts.json`, `results/verdicts/verdict_summary.json`. The repository also contains `results/verdicts/finetuned_verdicts.json` from a separate fine-tuned run; this report does **not** merge those metrics into the template summary above.

**Fine-tuning utilities (present in repo):** `models/prepare_finetune_data.py`, `models/finetune_qwen.py`, `models/evaluate_finetune.py`, `models/improve_verdicts.py`, adapter weights under `models/lora_adapter/`, and `data/processed/finetune_dataset.jsonl`.

---

## 13. Figures present in `figures/`

The following PNG files exist in the repository (diagnostic + SHAP + legacy extras):

- `agency_disagreement.png`, `confidence_dist.png`, `confidence_distribution_multisource.png`  
- `confusion_matrix_gb.png`, `confusion_matrix_multisource.png`  
- `decision_tree.png`  
- `error_patterns.png`, `error_patterns_multisource.png`, `error_scatter.png`, `error_scatter_multisource.png`  
- `feature_distributions.png`, `feature_distributions_multisource.png`  
- `kam_ablation.png`, `kam_profiles.png`  
- `model_comparison.png`, `multisource_model_comparison.png`  
- `pca_scatter.png`, `pca_multisource.png`, `rating_distribution.png`  
- `shap_bar_importance.png`, `shap_beeswarm.png`, `shap_dependence_leverage_profitab.png`, `shap_dependence_liquid_cumprof.png`, `shap_dependence_news_sentiment.png`  
- `shap_waterfall_*.png` (eight files in the current tree)

Only the **multisource benchmark**, **SHAP**, and **pipeline** figures are guaranteed to match the JSON timestamps if you run `scripts/regenerate_artifacts.py`. Other PNGs (e.g. `decision_tree.png`, `kam_ablation.png`) may stem from earlier analyses unless regenerated.

---

## 14. Reproducing outputs

From the project root (with dependencies installed, e.g. `.venv`):

```bash
PYTHONPATH=. python scripts/regenerate_artifacts.py
```

Use `--skip-verdicts` if you do not want to refresh `results/verdicts/`. Individual stages can be run via the Python modules listed in Section 5.

---

## 15. Repository layout (essential paths)

```
mxa1438/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   ├── processed/          # kams, financials, news, merged_multisource_training, …
│   ├── raw/                # financials JSON, news JSON
│   └── templates/          # scraping / extraction helpers
├── docs/                   # this file, DOCKER.md, FINBERT_NEWS_SENTIMENT.md, …
├── figures/                # plots (SHAP, diagnostics, benchmarks)
├── models/                 # xgboost_*, multisource_data, shap, llm, finetune, …
├── results/                # JSON metrics; verdicts/ subdirectory
└── scripts/                # rebuild, align, FinBERT collector, regenerate_artifacts
```

---

## 16. References

The following numbered list matches `docs/PROJECT_REPORT.md` (Section 24). **Section 2** of this document explains how each stream of work (KAMs, ratios, benchmarks, SHAP, LLMs, data sources) connects to these sources. Empirical **numbers** for the current pipeline still come only from the JSON files named in the header and from Sections 6–12.

1. Muñoz-Izquierdo, N., Segovia-Vargas, M.J., Camacho-Miñano, M.M., & Pérez-Pérez, Y. (2022). Machine learning in corporate credit rating assessment using the expanded audit report. *Machine Learning*, 111, 4183–4215. https://doi.org/10.1007/s10994-022-06226-4

2. Altman, E.I. (1983). *Corporate Financial Distress: A Complete Guide to Predicting, Avoiding, and Dealing with Bankruptcy*. Wiley.

3. Lundberg, S.M., & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*, 30.

4. Tassnief — Saudi Credit Rating Agency. https://tassnief.com

5. Saudi Exchange (Tadawul). https://www.saudiexchange.sa

6. Argaam (2025). “Insight into TASI-listed companies with credit ratings.” https://www.argaam.com/en/article/articledetail/id/1850297

7. Charitou, A., Neophytou, E., & Charalambous, C. (2004). Predicting corporate failure: Empirical evidence for the UK. *European Accounting Review*, 13(3), 465–497.

8. Fitch Ratings. Saudi Arabia Entity Ratings. https://www.fitchratings.com

9. Bao, W., Lianju, N., & Yue, K. (2019). Integration of unsupervised and supervised machine learning algorithms for credit risk assessment. *Expert Systems with Applications*, 128, 301–315. https://doi.org/10.1016/j.eswa.2019.02.033

10. Lei, Y., Wang, J., Chen, Z., Li, X., Chen, J., & Chen, C. (2025). ZiGong 1.0: A Large Language Model for Financial Credit. *arXiv preprint* arXiv:2502.16159.

---

*End of current pipeline report.*
