#!/usr/bin/env python3
"""
Recompute news sentiment features with FinBERT (ProsusAI/finbert) from cached
MarketAux JSON files under data/raw/news/.

Writes the same aggregate columns as the legacy VADER pipeline:
  sentiment_mean, sentiment_std, sentiment_pos_pct, sentiment_neg_pct, news_count

Updates data/processed/ratings_financials_sentiment.csv in place (merges into
existing financial columns). Then run:
  python scripts/rebuild_processed_datasets.py

Requires: torch, transformers, safetensors (see requirements.txt). CPU is fine; GPU used if available.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEWS_DIR = PROJECT_ROOT / "data" / "raw" / "news"
RATINGS_SENTIMENT_CSV = PROJECT_ROOT / "data" / "processed" / "ratings_financials_sentiment.csv"
MODEL_NAME = "ProsusAI/finbert"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_news_filename(path: Path) -> tuple[str, int] | None:
    """
    4263_SR_2025_news.json -> ('4263.SR', 2025)
    """
    m = re.match(r"^(.+)_(\d{4})_news\.json$", path.name, re.I)
    if not m:
        return None
    body, year_s = m.group(1), m.group(2)
    if "_" not in body:
        return None
    num, suf = body.rsplit("_", 1)
    ticker = f"{num}.{suf}"
    return ticker, int(year_s)


def article_text(art: dict) -> str:
    title = (art.get("title") or "").strip()
    desc = (art.get("description") or art.get("snippet") or "").strip()
    return (title + " " + desc).strip() or title or desc or " "


def resolve_pos_neg_indices(model) -> tuple[int, int]:
    """Map FinBERT label ids to positive / negative probability columns."""
    id2label = model.config.id2label
    # keys may be str or int
    norm = {}
    for k, v in id2label.items():
        idx = int(k)
        norm[idx] = str(v).lower()

    pos_idx = neg_idx = None
    for idx, lab in norm.items():
        if lab == "positive":
            pos_idx = idx
        elif lab == "negative":
            neg_idx = idx
    if pos_idx is None or neg_idx is None:
        raise RuntimeError(f"Unexpected id2label for FinBERT: {id2label}")
    return pos_idx, neg_idx


def score_texts_batched(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    pos_idx: int,
    neg_idx: int,
    batch_size: int,
) -> list[float]:
    scores: list[float] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        pos_p = probs[:, pos_idx]
        neg_p = probs[:, neg_idx]
        diff = (pos_p - neg_p).cpu().tolist()
        scores.extend(float(x) for x in diff)
    return scores


def aggregate_scores(scores: list[float]) -> dict:
    if not scores:
        return {
            "sentiment_mean": 0.0,
            "sentiment_std": 0.0,
            "sentiment_pos_pct": 0.0,
            "sentiment_neg_pct": 0.0,
            "news_count": 0,
        }
    n = len(scores)
    pos_pct = sum(1 for s in scores if s > 0) / n
    neg_pct = sum(1 for s in scores if s < 0) / n
    mean = statistics.mean(scores)
    std = statistics.pstdev(scores) if n > 1 else 0.0
    return {
        "sentiment_mean": float(mean),
        "sentiment_std": float(std),
        "sentiment_pos_pct": float(pos_pct),
        "sentiment_neg_pct": float(neg_pct),
        "news_count": n,
    }


def collect_all_articles(news_dir: Path) -> tuple[list[tuple[str, int, list[dict]]], list[str]]:
    """Returns list of (ticker, fiscal_year, articles) and flat list of texts in parallel order."""
    groups: list[tuple[str, int, list[dict]]] = []
    flat_texts: list[str] = []

    paths = sorted(news_dir.glob("*_news.json"))
    for p in paths:
        parsed = parse_news_filename(p)
        if not parsed:
            logger.warning("Skip unreadable filename: %s", p.name)
            continue
        ticker, fy = parsed
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Skip %s: %s", p, e)
            continue
        articles = data.get("articles") or []
        if not articles:
            groups.append((ticker, fy, []))
            continue
        groups.append((ticker, fy, articles))
        for art in articles:
            flat_texts.append(article_text(art))

    return groups, flat_texts


def main() -> None:
    ap = argparse.ArgumentParser(description="FinBERT sentiment from raw news JSON")
    ap.add_argument(
        "--news-dir",
        type=Path,
        default=NEWS_DIR,
        help="Directory with *_news.json files",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=RATINGS_SENTIMENT_CSV,
        help="CSV to update (same schema as ratings_financials_sentiment)",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Score articles but do not write CSV",
    )
    args = ap.parse_args()

    if not args.news_dir.is_dir():
        raise SystemExit(f"News directory not found: {args.news_dir}")

    groups, flat_texts = collect_all_articles(args.news_dir)
    logger.info("Found %s article texts across %s ticker-year files", len(flat_texts), len(groups))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading %s on %s ...", MODEL_NAME, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Newer transformers + old torch may require safetensors; older transformers
    # load pytorch_model.bin. Extra kwargs are ignored if unsupported.
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            use_safetensors=True,
        )
    except OSError:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    pos_idx, neg_idx = resolve_pos_neg_indices(model)
    logger.info("FinBERT label indices: positive=%s negative=%s", pos_idx, neg_idx)

    all_scores: list[float] = []
    if flat_texts:
        all_scores = score_texts_batched(
            flat_texts,
            tokenizer,
            model,
            device,
            pos_idx,
            neg_idx,
            args.batch_size,
        )

    # Map (ticker, fy) -> aggregates
    idx = 0
    sentiment_by_key: dict[tuple[str, int], dict] = {}
    for ticker, fy, articles in groups:
        key = (ticker, fy)
        if not articles:
            sentiment_by_key[key] = aggregate_scores([])
            continue
        n = len(articles)
        chunk = all_scores[idx : idx + n]
        idx += n
        sentiment_by_key[key] = aggregate_scores(chunk)

    assert idx == len(all_scores), "Score alignment bug"

    if not args.output.exists():
        raise SystemExit(f"Missing base CSV {args.output}; create it before running this script.")

    with args.output.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames or "fiscal_year" not in fieldnames or "ticker" not in fieldnames:
            raise SystemExit("Base CSV missing required columns (ticker, fiscal_year)")
        rows = list(reader)

    updated = 0
    for row in rows:
        try:
            fy = int(str(row["fiscal_year"]).strip())
        except (TypeError, ValueError):
            continue
        key = (str(row["ticker"]).strip(), fy)
        if key not in sentiment_by_key:
            continue
        s = sentiment_by_key[key]
        row["sentiment_mean"] = str(s["sentiment_mean"])
        row["sentiment_std"] = str(s["sentiment_std"])
        row["sentiment_pos_pct"] = str(s["sentiment_pos_pct"])
        row["sentiment_neg_pct"] = str(s["sentiment_neg_pct"])
        row["news_count"] = str(s["news_count"])
        updated += 1

    logger.info("Updated sentiment for %s rows (matched ticker+fiscal_year to news files)", updated)

    if args.dry_run:
        logger.info("Dry run: not writing CSV")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s", args.output)
    print("Next: python scripts/rebuild_processed_datasets.py")


if __name__ == "__main__":
    main()
