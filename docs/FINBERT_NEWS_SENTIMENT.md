# FinBERT news sentiment (implemented)

News aggregates in `data/processed/ratings_financials_sentiment.csv` (and downstream `news_features_processed.csv`) are produced with **FinBERT** (`ProsusAI/finbert`), not VADER.

## Workflow

1. **Raw articles:** `data/raw/news/{NUM}_SR_{YYYY}_news.json` (MarketAux cache; same as before).
2. **Rescore:** from repo root `mxa1438/`:

   ```bash
   python scripts/collect_news_sentiment_finbert.py
   ```

   This loads FinBERT locally, scores each article’s `title` + `description`/`snippet`, then overwrites the five sentiment columns for rows whose `(ticker, fiscal_year)` matches a news file.

3. **Refresh derived tables:**

   ```bash
   python scripts/rebuild_processed_datasets.py
   ```

## Per-article score

For each article, after softmax over the three logits:

`score = P(positive) − P(negative)`

Positive and negative class indices are taken from `model.config.id2label` at runtime (do not hard-code label order).

## Aggregates (unchanged column names)

By `(ticker, fiscal_year)`:

| Column | Definition |
|--------|--------------|
| `sentiment_mean` | Mean of per-article scores |
| `sentiment_std` | Population std of scores (0 if one article) |
| `sentiment_pos_pct` | Share of articles with `score > 0` |
| `sentiment_neg_pct` | Share of articles with `score < 0` |
| `news_count` | Number of articles |

Rows in `ratings_financials_sentiment.csv` with no matching `*_news.json` keep their previous sentiment values.

## Dependencies

`torch`, `transformers`, and `safetensors` (see `requirements.txt`). If `from_pretrained(..., use_safetensors=True)` fails (older stacks), the script falls back to `pytorch_model.bin`.

## Legacy VADER collector

`scripts/collect_news_sentiment.py` (VADER + API) is legacy; use the FinBERT script above for reproducible financial sentiment on cached JSON.

## References

- FinBERT (arXiv): *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models* (Huang, Wang, Yang).
- Model card: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) on Hugging Face.
