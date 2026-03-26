#!/usr/bin/env python3
"""
Align processed modeling CSVs to the same (ticker, fiscal_year) panel as kams_processed.csv.

Use when you cannot run rebuild_processed_datasets.py (e.g. missing ratings_with_financials.csv).
Run from repo root mxa1438/:

  python scripts/align_processed_to_kams.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import rebuild_processed_datasets as rbd


def main() -> None:
    kams = pd.read_csv(rbd.OUT_KAMS)
    fin_tickers = rbd.financial_ticker_set(pd.DataFrame(), kams)

    news = pd.read_csv(rbd.OUT_NEWS)
    news = rbd.drop_financial_tickers(news, fin_tickers)
    news = rbd.finalize_news_features_panel(rbd.align_to_kams_panel(kams, news))
    news.to_csv(rbd.OUT_NEWS, index=False)
    print(f"Wrote {rbd.OUT_NEWS.name} ({len(news)} rows)")

    fin_aligned = None
    if rbd.OUT_FINANCIALS_MULTI.exists():
        fin_m = pd.read_csv(rbd.OUT_FINANCIALS_MULTI)
        fin_m = rbd.drop_financial_tickers(fin_m, fin_tickers)
        aligned_m = rbd.align_financials_processed_to_kams(kams, fin_m)
        aligned_m.to_csv(rbd.OUT_FINANCIALS_MULTI, index=False)
        print(f"Wrote {rbd.OUT_FINANCIALS_MULTI.name} ({len(aligned_m)} rows)")

    if rbd.OUT_FIN.exists():
        fin = pd.read_csv(rbd.OUT_FIN)
        fin = rbd.drop_financial_tickers(fin, fin_tickers)
        fin_aligned = rbd.align_to_kams_panel(kams, fin)
        fin_aligned.to_csv(rbd.OUT_FIN, index=False)
        print(f"Wrote {rbd.OUT_FIN.name} ({len(fin_aligned)} rows)")

    if fin_aligned is not None:
        merged = rbd.build_merged_multisource(fin_aligned, kams, news)
        rbd.OUT_MERGED.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(rbd.OUT_MERGED, index=False)
        print(f"Wrote {rbd.OUT_MERGED.name} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
