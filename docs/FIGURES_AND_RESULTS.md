# Figures and results catalog

This document describes **every** file under [`figures/`](../figures) and [`results/`](../results) (plus [`results/verdicts/`](../results/verdicts)) in this repository. Paths are relative to the **`mxa1438/`** project root.

**Regenerate everything** (from `mxa1438/`):

```bash
PYTHONPATH=. python scripts/regenerate_artifacts.py
```

Individual writers are noted below. After changing `data/processed/*.csv` or model code, run the script again so figures and JSON match the current panel.

---

## Figures (`figures/`)

### Pipeline diagnostics (`models/generate_pipeline_figures.py`)

All of these use the **merged multisource training table** (14 features), **4-class** `rating_category`, and **stratified CV** with the same XGBoost settings as `app.py` / `xgboost_full.py` (unless noted).

| Reference | What it shows |
|-----------|----------------|
| [`figures/rating_distribution.png`](../figures/rating_distribution.png) | Bar chart: number of `(ticker, fiscal_year)` rows per rating category (A, AA, BB, BBB). Shows **class balance** on the modelling sample after dropping rows with missing core ratios. |
| [`figures/confusion_matrix_multisource.png`](../figures/confusion_matrix_multisource.png) | **Confusion matrix** for out-of-fold CV predictions of multisource XGBoost. Title includes CV accuracy and sample size. |
| [`figures/confusion_matrix_gb.png`](../figures/confusion_matrix_gb.png) | **Identical copy** of `confusion_matrix_multisource.png` (legacy filename for older docs/UI). |
| [`figures/pca_multisource.png`](../figures/pca_multisource.png) | **PCA** (two components) of **standardized** 14 features; points coloured by `rating_category`. Shows overlap / separation of categories in linear projection. |
| [`figures/pca_scatter.png`](../figures/pca_scatter.png) | **Identical copy** of `pca_multisource.png` (alias used by the Dataset Explorer tab). |
| [`figures/feature_distributions_multisource.png`](../figures/feature_distributions_multisource.png) | **Boxplots** of the four financial ratios (`liquid`, `cumprof`, `profitab`, `leverage`) by `rating_category`. |
| [`figures/feature_distributions.png`](../figures/feature_distributions.png) | **Identical copy** of `feature_distributions_multisource.png`. |
| [`figures/confidence_distribution_multisource.png`](../figures/confidence_distribution_multisource.png) | **Histogram** of **max predicted class probability** per row from CV `predict_proba` (model confidence when the predicted class is taken as argmax). |
| [`figures/confidence_dist.png`](../figures/confidence_dist.png) | **Identical copy** of `confidence_distribution_multisource.png`. |
| [`figures/error_scatter_multisource.png`](../figures/error_scatter_multisource.png) | Same **PCA coordinates** as above, but points coloured **correct vs misclassified** under CV (red = wrong, grey = correct). |
| [`figures/error_scatter.png`](../figures/error_scatter.png) | **Identical copy** of `error_scatter_multisource.png`. |
| [`figures/error_patterns_multisource.png`](../figures/error_patterns_multisource.png) | **Horizontal bar chart**: counts of **actual → predicted** category pairs for CV errors only (e.g. `BBB → A`). |
| [`figures/error_patterns.png`](../figures/error_patterns.png) | **Identical copy** of `error_patterns_multisource.png`. |

### Multisource model benchmark (`models/evaluate_multisource_models.py`)

| Reference | What it shows |
|-----------|----------------|
| [`figures/multisource_model_comparison.png`](../figures/multisource_model_comparison.png) | **Horizontal bar chart**: **mean CV accuracy** for each of **ten** models (XGBoost, Random Forest, Extra Trees, gradient boosting variants, tree, linear, kNN, MLP) on the **same** 14-feature matrix. Numeric details: [`results/multisource_model_comparison.json`](../results/multisource_model_comparison.json). |
| [`figures/model_comparison.png`](../figures/model_comparison.png) | **Copy** of `multisource_model_comparison.png` when the benchmark PNG exists (written by `generate_pipeline_figures.py` after the benchmark is generated). |

### All-features benchmark (`models/xgboost_all_features.py`)

| Reference | What it shows |
|-----------|----------------|
| [`figures/all_features_model_comparison.png`](../figures/all_features_model_comparison.png) | **Bar chart**: **mean CV accuracy** for **nine** learners on the **21-feature** superset matrix (4 financial ratios + the full 12-column KAM/firm block + 5 news aggregates), same 45 rows. Linear SVC leads at **82.2%**, Extra Trees second at **80.0%**; read alongside macro-F1, which is **0.76** for the leader. Numeric details: [`results/all_features_model_results.json`](../results/all_features_model_results.json). |

### SHAP explainability (`models/shap_explainability.py`)

Trains **full multisource XGBoost** on all in-sample rows and computes **TreeExplainer** SHAP values (multiclass).

| Reference | What it shows |
|-----------|----------------|
| [`figures/shap_beeswarm.png`](../figures/shap_beeswarm.png) | **SHAP summary (dot / beeswarm)** using the **majority class** output of the multiclass explainer: each point is a row; x-axis is SHAP value; features sorted by impact. |
| [`figures/shap_bar_importance.png`](../figures/shap_bar_importance.png) | **Bar chart**: **mean absolute SHAP** per feature (aggregated across classes as in the script). Global importance for the fitted full model. |
| [`figures/shap_dependence_leverage_profitab.png`](../figures/shap_dependence_leverage_profitab.png) | **Dependence plot**: SHAP for **leverage** vs feature value, coloured by **profitab** (interaction-style view). |
| [`figures/shap_dependence_liquid_cumprof.png`](../figures/shap_dependence_liquid_cumprof.png) | **Dependence plot**: SHAP for **liquid** vs value, coloured by **cumprof**. |
| [`figures/shap_dependence_news_sentiment.png`](../figures/shap_dependence_news_sentiment.png) | **Dependence plot**: SHAP for **news_count** vs value, coloured by **sentiment_mean**. |
| [`figures/shap_waterfall_<ticker>_SR_<year>.png`](../figures) | **Waterfall** plots for a **sample** of companies (filename encodes Tadawul ticker and fiscal year, e.g. `shap_waterfall_1202_SR_2024.png`). Explains the **predicted class** for that row. Old waterfalls are deleted when the script re-runs. |

---

## Results (`results/`)

### Benchmarks and ablations

| Reference | What it contains |
|-----------|------------------|
| [`results/multisource_model_comparison.json`](../results/multisource_model_comparison.json) | **Metadata**: sample size, feature list, class counts, CV settings. **`models`**: per-algorithm `accuracy_mean`, `accuracy_std`, F1 macro/weighted. **`ranking_by_accuracy`**: model names sorted best-first. Produced by `evaluate_multisource_models.py`. |
| [`results/full_model_comparison.json`](../results/full_model_comparison.json) | **XGBoost ablation** on the **multisource merge**: `financials_only` → `financials_kams` → `financials_sentiment` → `full_model` with CV **accuracy** and **std** per stage. **`feature_sets`**: which columns belong to financial / KAM / sentiment blocks. From `xgboost_full.py`. |
| [`results/combined_model_results.json`](../results/combined_model_results.json) | **Extended KAM / financial comparison** from `xgboost_with_kams.py`: e.g. `financials_only`, `combined` (financials + rich KAM set on merged rows), and **`kams_only_merged_sample`** (KAM-only on the same rows as the financial merge — contrast with full KAM panel in `kams_only_model_results.json`). Includes optional **paper_reference** fields for write-up. |
| [`results/kams_only_model_results.json`](../results/kams_only_model_results.json) | **`xgboost_kams_only.py`** on **`data/processed/kams_processed.csv`**: 12 KAM/firm features, CV accuracy, train accuracy, **XGBoost feature importance** map. Larger KAM-only sample than the merged multisource table. |
| [`results/all_features_model_results.json`](../results/all_features_model_results.json) | **`xgboost_all_features.py`** on the **45×21 superset** (financial ratios + full 12-column KAM/firm block + news aggregates). **`ablations`**: five XGBoost feature-subset runs with accuracy, std, and top-3 importances. **`benchmark_21_features`**: nine learners with accuracy and macro/weighted F1. **`benchmark_ranking`**: model names best-first. Highest accuracy in the repo (**Linear SVC, 82.2% ± 5.4%**, macro-F1 **0.76**) — see §15.5.1 of `PROJECT_REPORT.md` for the caveats, including why this ranking reorders across library versions. |

### Error analysis and pipeline metadata

| Reference | What it contains |
|-----------|------------------|
| [`results/error_analysis_multisource.json`](../results/error_analysis_multisource.json) | **Static snapshot** from `generate_pipeline_figures.py`: CV accuracy, confusion matrix, list of **misclassified** rows (ticker, year, actual/predicted category, confidence), summary counts. |
| [`results/error_analysis.json`](../results/error_analysis.json) | **Copy** of `error_analysis_multisource.json` (alias). |
| [`results/pipeline_figures_meta.json`](../results/pipeline_figures_meta.json) | **UTC timestamp**, **`n_rows`**, and lists **`figures_written`** / **`results_written`** for the last `generate_pipeline_figures.py` run. Quick audit of what was regenerated. |

### SHAP report

| Reference | What it contains |
|-----------|------------------|
| [`results/shap_report.json`](../results/shap_report.json) | Model description, `n_samples`, feature list, classes. **`global_importance`**: per-feature `rank`, `mean_abs_shap`, and **text `description`**. Aligns with `shap_explainability.py` and the static SHAP PNGs. |

---

## Verdicts (`results/verdicts/`)

Produced by **`models/llm_verdict.py`** (template mode in the default pipeline; no GPU required).

| Reference | What it contains |
|-----------|------------------|
| [`results/verdicts/all_verdicts.json`](../results/verdicts/all_verdicts.json) | Full list of **per-row verdicts** (structured fields: strengths, weaknesses, risks, etc.) aligned with ML inputs. |
| [`results/verdicts/verdict_summary.json`](../results/verdicts/verdict_summary.json) | **Aggregate QA-style metrics**: average scores, citation rate, agreement with numeric mentions, **min/max** score, plus **`ml_performance`** (how often template verdict category matches XGBoost prediction on the batch). |

---

## Streamlit app mapping

| App tab | Figures / results surfaced |
|---------|----------------------------|
| **Company Predictor** | Live XGBoost + SHAP waterfall + template verdict (not file-based). |
| **Model Performance** | `multisource_model_comparison.png` or chart from JSON; table from `multisource_model_comparison.json`; ablation from `full_model_comparison.json`. |
| **SHAP Explainability** | Live SHAP plots + static `shap_*.png` + `shap_report.json`. |
| **Error Analysis** | **Live** CV plots + JSON (same logic as pipeline; may differ slightly from frozen `error_analysis.json` if data changed). |
| **Dataset Explorer** | `pca_scatter.png` + data table. |
| **Pipeline outputs** | Gallery of **pipeline diagnostic** PNGs, benchmark PNG, pointers to SHAP files, and **expanders** for major JSON results. |

For a single reference document of paths and meanings, this file is the source of truth: **`docs/FIGURES_AND_RESULTS.md`**.
