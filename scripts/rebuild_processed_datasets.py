#!/usr/bin/env python3
"""
Rebuild the three processed modeling tables + merged training file.

Outputs (data/processed/):
  - kams_processed.csv           — from templates/kams_priority.csv (paper-style KAMs)
  - financial_ratios_processed.csv — from yfinance (--yfinance) OR from ratings_with_financials.csv
  - news_features_processed.csv  — from ratings_financials_sentiment.csv (FinBERT columns after
    collect_news_sentiment_finbert.py); rows are restricted to the same (ticker, fiscal_year) panel as kams
  - financials_processed.csv     — if present before rebuild, rewritten aligned to the KAM panel
  - merged_multisource_training.csv — inner join of the three (for Streamlit / LLM / finetune prep)

All outputs drop financial-sector tickers (banks, insurance, finance names, yfinance Financial Services),
using heuristics + a small explicit ticker list — see financial_ticker_set().

Usage:
  python scripts/rebuild_processed_datasets.py              # financials slice from ratings CSV only
  python scripts/rebuild_processed_datasets.py --yfinance # refresh ratios via yfinance for each KAM row

Manual figures from Saudi Exchange (or annual reports): fill
  data/templates/saudiexchange_financial_supplement.csv
with ticker, fiscal_year, and accounting lines (same currency as statements, typically SAR).
The script recomputes liquid / cumprof / profitab / leverage and merges into
ratings_with_financials.csv on rebuild. Columns outlook, rating_action, and data_available
are dropped from the financial CSVs.

Requires: pandas; yfinance only if --yfinance
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
TEMPLATES = DATA / "templates"
PROCESSED = DATA / "processed"

KAMS_SOURCE = TEMPLATES / "kams_priority.csv"
FINANCIALS_SOURCE = PROCESSED / "ratings_with_financials.csv"
FINANCIAL_SUPPLEMENT = TEMPLATES / "saudiexchange_financial_supplement.csv"
SENTIMENT_SOURCE = PROCESSED / "ratings_financials_sentiment.csv"
RATINGS_FIN = PROCESSED / "ratings_with_financials.csv"

# Removed from persisted financial tables (not needed for modeling).
FIN_COLS_TO_DROP = ("outlook", "rating_action", "data_available")

OUT_KAMS = PROCESSED / "kams_processed.csv"
OUT_FIN = PROCESSED / "financial_ratios_processed.csv"
OUT_NEWS = PROCESSED / "news_features_processed.csv"
OUT_MERGED = PROCESSED / "merged_multisource_training.csv"
OUT_FINANCIALS_MULTI = PROCESSED / "financials_processed.csv"

PAPER_AUDIT_KAM = [
    "AUSIZE", "AUOP", "EMP", "GCUP",
    "GCKAM", "REVKAM", "ASSETKAM", "LIABKAM", "OTHERKAM",
]
PAPER_CONTROLS = ["FIRMAGE", "FIRMSIZE", "INDUSTRY"]
KAM_FIVE = ["GCKAM", "REVKAM", "ASSETKAM", "LIABKAM", "OTHERKAM"]
NEWS_COLS = [
    "sentiment_mean", "sentiment_std", "sentiment_pos_pct",
    "sentiment_neg_pct", "news_count",
]

SLEEP_S = 0.35
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Exclude banks, insurers, finance firms (non-financial sample for ratio-based models).
FINANCIAL_TICKERS_EXPLICIT: frozenset[str] = frozenset(
    {
        "9596.SR",  # Quara Finance
        "8050.SR",  # Saudi Arabian Cooperative Insurance
        "8060.SR",  # Malath Insurance
        "8200.SR",  # Saudi Re
    }
)
FINANCIAL_SECTORS_EXACT: frozenset[str] = frozenset({"financial services"})
FINANCIAL_INDUSTRY_SUBSTR: tuple[str, ...] = (
    "insurance",
    "reinsurance",
    "bank",
    "banks",
    "capital market",
    "asset management",
    "credit services",
    "mortgage",
    "consumer finance",
    "financial -",
    "shell companies",
)
FINANCIAL_NAME_SUBSTR: tuple[str, ...] = (
    "insurance",
    "reinsurance",
    "finance company",
    " islamic bank",
    "national bank",
    "commercial bank",
    "investment bank",
)


def _is_financial_entity(
    ticker: str,
    yfinance_sector: str,
    yfinance_industry: str,
    company_name: str,
) -> bool:
    t = str(ticker).strip()
    if t in FINANCIAL_TICKERS_EXPLICIT:
        return True
    sec = (str(yfinance_sector) if pd.notna(yfinance_sector) else "").strip().lower()
    if sec in FINANCIAL_SECTORS_EXACT:
        return True
    ind = (str(yfinance_industry) if pd.notna(yfinance_industry) else "").strip().lower()
    for sub in FINANCIAL_INDUSTRY_SUBSTR:
        if sub in ind:
            return True
    name = (str(company_name) if pd.notna(company_name) else "").lower()
    for sub in FINANCIAL_NAME_SUBSTR:
        if sub in name:
            return True
    if " bank" in name or name.endswith(" bank"):
        return True
    return False


def financial_ticker_set(
    ratings_df: pd.DataFrame, kams_df: pd.DataFrame
) -> frozenset[str]:
    """Tickers classified as financial (banks, insurance, finance) to drop from processed outputs."""
    bad: set[str] = set()
    if ratings_df is not None and len(ratings_df) > 0:
        r = ratings_df.copy()
        for ticker, grp in r.groupby("ticker", sort=False):
            if "fiscal_year" in grp.columns:
                grp = grp.sort_values("fiscal_year", ascending=False)
            name = ""
            sector, industry = "", ""
            for _, row in grp.iterrows():
                if not name and pd.notna(row.get("company_name")):
                    name = str(row["company_name"]).strip()
                ys = row.get("yfinance_sector")
                if not sector and pd.notna(ys) and str(ys).strip():
                    sector = str(ys).strip()
                    industry = str(row.get("yfinance_industry") or "").strip()
                    break
            if not name:
                name = str(grp.iloc[0].get("company_name") or "").strip()
            if _is_financial_entity(str(ticker), sector, industry, name):
                bad.add(str(ticker).strip())

    rated: set[str] = set()
    if ratings_df is not None and len(ratings_df) > 0 and "ticker" in ratings_df.columns:
        rated = set(ratings_df["ticker"].astype(str).str.strip())
    for t in kams_df["ticker"].unique():
        ts = str(t).strip()
        if ts in bad:
            continue
        if ts in rated:
            continue
        sub = kams_df[kams_df["ticker"] == t].iloc[0]
        nm = str(sub.get("company_name") or "")
        if _is_financial_entity(ts, "", "", nm):
            bad.add(ts)

    return frozenset(bad)


def drop_financial_tickers(df: pd.DataFrame, financial: frozenset[str]) -> pd.DataFrame:
    if not len(financial) or "ticker" not in df.columns:
        return df
    return df[~df["ticker"].astype(str).str.strip().isin(financial)].copy()


def align_to_kams_panel(
    kams: pd.DataFrame,
    df: pd.DataFrame,
    *,
    dedupe_prefer_tassnief: bool = False,
) -> pd.DataFrame:
    """One row per KAM row (same order as kams); left-join rows from df on (ticker, fiscal_year)."""
    keys = kams[["ticker", "fiscal_year"]].copy()
    keys["fiscal_year"] = keys["fiscal_year"].astype(int)
    if df is None or df.empty:
        return keys
    d = df.copy()
    d["fiscal_year"] = pd.to_numeric(d["fiscal_year"], errors="coerce").astype("Int64")
    d = d.dropna(subset=["fiscal_year"])
    d["fiscal_year"] = d["fiscal_year"].astype(int)
    if dedupe_prefer_tassnief and "rating_agency" in d.columns:
        agency = d["rating_agency"].astype(str).str.strip().str.lower()
        d = d.assign(_pri=(agency == "tassnief").astype(int))
        d = d.sort_values("_pri", ascending=False).drop(columns=["_pri"])
    d = d.drop_duplicates(subset=["ticker", "fiscal_year"], keep="first")
    return keys.merge(d, on=["ticker", "fiscal_year"], how="left")


def finalize_news_features_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure FinBERT-style aggregate columns exist.

    Missing sentiment aggregates are left as NaN rather than zero-filled: a
    sentiment_mean of 0.0 is a real, neutral reading, and using the same
    value to mean "no articles were retrieved" makes the two indistinguishable
    downstream (see docs/PROJECT_REPORT.md, "silent zero" failure mode).
    `news_count` is a true count, so an absence of articles is legitimately
    zero there. `has_news_coverage` makes the distinction explicit for any
    consumer that doesn't want to reason about NaN directly.
    """
    out = df.copy()
    sentiment_cols = [c for c in NEWS_COLS if c != "news_count"]
    if "news_count" not in out.columns:
        out["news_count"] = 0
    else:
        out["news_count"] = pd.to_numeric(out["news_count"], errors="coerce").fillna(0)
    for c in sentiment_cols:
        if c not in out.columns:
            out[c] = np.nan
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["has_news_coverage"] = (out["news_count"] > 0).astype(int)
    return out[["ticker", "fiscal_year"] + NEWS_COLS + ["has_news_coverage"]].copy()


def _rating_to_binary_target(rating) -> int:
    if pd.isna(rating):
        return 0
    s = str(rating).strip().upper()
    if s.startswith("AAA"):
        return 0
    if s.startswith("AA"):
        return 0
    if s.startswith("A"):
        return 0
    if s.startswith("BBB"):
        return 0
    return 1


def align_financials_processed_to_kams(
    kams: pd.DataFrame, fin_multi: pd.DataFrame
) -> pd.DataFrame:
    """Multi-agency financials table: one row per KAM (ticker, year); prefer Tassnief when deduping."""
    keys = kams[["ticker", "fiscal_year", "company_name"]].copy()
    keys["fiscal_year"] = keys["fiscal_year"].astype(int)
    d = fin_multi.copy()
    d["fiscal_year"] = pd.to_numeric(d["fiscal_year"], errors="coerce").astype("Int64")
    d = d.dropna(subset=["fiscal_year"])
    d["fiscal_year"] = d["fiscal_year"].astype(int)
    if "rating_agency" in d.columns:
        agency = d["rating_agency"].astype(str).str.strip().str.lower()
        d = d.assign(_pri=(agency == "tassnief").astype(int))
        d = d.sort_values("_pri", ascending=False).drop(columns=["_pri"])
    d = d.drop_duplicates(subset=["ticker", "fiscal_year"], keep="first")
    drop_co = [c for c in ("company_name",) if c in d.columns]
    merged = keys.merge(d.drop(columns=drop_co, errors="ignore"), on=["ticker", "fiscal_year"], how="left")

    rk = kams["rating"].reset_index(drop=True)
    if "rating" in merged.columns:
        merged["rating"] = merged["rating"].fillna(rk)
    else:
        merged["rating"] = rk

    if "rating_agency" not in merged.columns:
        merged["rating_agency"] = ""
    else:
        merged["rating_agency"] = merged["rating_agency"].fillna("").astype(str)

    computed_bt = merged["rating"].map(_rating_to_binary_target)
    if "binary_target" in merged.columns:
        merged["binary_target"] = (
            pd.to_numeric(merged["binary_target"], errors="coerce")
            .fillna(computed_bt)
            .astype(int)
        )
    else:
        merged["binary_target"] = computed_bt.astype(int)

    col_order = [
        "ticker",
        "company_name",
        "rating_agency",
        "rating",
        "fiscal_year",
        "liquid",
        "cumprof",
        "profitab",
        "leverage",
        "binary_target",
    ]
    for c in col_order:
        if c not in merged.columns:
            merged[c] = float("nan") if c != "binary_target" else 0
    return merged[col_order].copy()


def _fin_float(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def ratios_from_accounting_inputs(
    total_assets: float | None,
    current_assets: float | None,
    current_liabilities: float | None,
    retained_earnings: float | None,
    total_equity: float | None,
    total_liabilities: float | None,
    ebit: float | None,
) -> dict[str, float | None]:
    """Same definitions as calculate_financial_ratios (Altman-style project ratios)."""
    out: dict[str, float | None] = {
        "liquid": None,
        "cumprof": None,
        "profitab": None,
        "leverage": None,
    }
    ta = total_assets
    if ta is not None and ta != 0:
        if current_assets is not None and current_liabilities is not None:
            out["liquid"] = (current_assets - current_liabilities) / ta
        if retained_earnings is not None:
            out["cumprof"] = retained_earnings / ta
        if ebit is not None:
            out["profitab"] = ebit / ta
    if (
        total_equity is not None
        and total_liabilities is not None
        and total_liabilities != 0
    ):
        out["leverage"] = total_equity / total_liabilities
    return out


def _normalize_financial_ratings_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in FIN_COLS_TO_DROP:
        if c in out.columns:
            out = out.drop(columns=[c])
    return out


def _supplement_row_updates(row: pd.Series) -> dict:
    """Map one supplement CSV row -> columns to write onto ratings_with_financials."""
    ebit = _fin_float(row.get("operating_profit"))
    if ebit is None and "ebit" in row.index:
        ebit = _fin_float(row.get("ebit"))

    ta = _fin_float(row.get("total_assets"))
    ca = _fin_float(row.get("current_assets"))
    cl = _fin_float(row.get("current_liabilities"))
    re_ = _fin_float(row.get("retained_earnings"))
    te = _fin_float(row.get("total_equity"))
    tl = _fin_float(row.get("total_liabilities"))

    ratios = ratios_from_accounting_inputs(ta, ca, cl, re_, te, tl, ebit)
    upd = {k: v for k, v in ratios.items() if v is not None}

    for col in ("total_assets", "total_liabilities", "total_equity"):
        v = _fin_float(row.get(col))
        if v is not None:
            upd[col] = v

    spe = row.get("statement_period_end")
    if spe is not None and str(spe).strip() and not pd.isna(spe):
        upd["statement_period_end"] = str(spe).strip()

    for col in ("yfinance_sector", "yfinance_industry"):
        if col not in row.index:
            continue
        s = row.get(col)
        if s is not None and str(s).strip() and not pd.isna(s):
            upd[col] = str(s).strip()

    return upd


def apply_financial_supplement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge manually keyed figures from saudiexchange_financial_supplement.csv
    (ticker + fiscal_year). Supplement values override existing cells when present.
    """
    if not FINANCIAL_SUPPLEMENT.exists():
        return df
    sup = pd.read_csv(FINANCIAL_SUPPLEMENT)
    if sup.empty:
        return df
    sup = sup.dropna(subset=["ticker", "fiscal_year"], how="any")
    if sup.empty:
        return df
    sup["fiscal_year"] = pd.to_numeric(sup["fiscal_year"], errors="coerce").astype(int)
    sup["ticker"] = sup["ticker"].astype(str).str.strip()

    out = df.copy()
    for _, srow in sup.iterrows():
        t = str(srow["ticker"]).strip()
        fy = int(srow["fiscal_year"])
        mask = (out["ticker"].astype(str).str.strip() == t) & (out["fiscal_year"] == fy)
        if not mask.any():
            logger.warning(
                "Supplement row skipped (no matching ratings row): %s FY%s",
                t,
                fy,
            )
            continue
        upd = _supplement_row_updates(srow)
        for i in out.index[mask]:
            for col, val in upd.items():
                if col not in out.columns:
                    out[col] = pd.NA
                out.at[i, col] = val
    logger.info(
        "Applied %s rows from %s",
        len(sup),
        FINANCIAL_SUPPLEMENT.name,
    )
    return out


def load_merge_and_persist_ratings_financials() -> pd.DataFrame:
    """
    Read ratings_with_financials.csv, drop unneeded columns, apply Saudi/manual
    supplement, save back, return enriched frame.
    """
    if not FINANCIALS_SOURCE.exists():
        raise FileNotFoundError(FINANCIALS_SOURCE)
    df = pd.read_csv(FINANCIALS_SOURCE)
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["fiscal_year"])
    df["fiscal_year"] = df["fiscal_year"].astype(int)
    df = _normalize_financial_ratings_df(df)
    df = apply_financial_supplement(df)
    df.to_csv(FINANCIALS_SOURCE, index=False)
    return df


def _coerce_int_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = 0
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df


def build_kams_processed() -> pd.DataFrame:
    df = pd.read_csv(KAMS_SOURCE)
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["fiscal_year"])
    df["fiscal_year"] = df["fiscal_year"].astype(int)
    all_paper = PAPER_AUDIT_KAM + PAPER_CONTROLS
    for c in all_paper:
        if c not in df.columns:
            df[c] = 0
    df = _coerce_int_cols(df, all_paper)
    id_cols = ["ticker", "company_name", "fiscal_year", "rating"]
    id_cols = [c for c in id_cols if c in df.columns]
    out_cols = id_cols + PAPER_AUDIT_KAM + PAPER_CONTROLS
    out_cols = [c for c in out_cols if c in df.columns]
    return df[out_cols].copy()


def financial_processed_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Columns written to financial_ratios_processed.csv."""
    out = df.copy()
    for c in ["liquid", "cumprof", "profitab", "leverage"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    keep = [
        "ticker",
        "company_name",
        "rating_date",
        "rating",
        "fiscal_year",
        "liquid",
        "cumprof",
        "profitab",
        "leverage",
        "total_assets",
        "total_liabilities",
        "total_equity",
        "yfinance_sector",
        "yfinance_industry",
        "statement_period_end",
    ]
    keep = [c for c in keep if c in out.columns]
    return out[keep].copy()


def build_news_features_processed() -> pd.DataFrame:
    df = pd.read_csv(SENTIMENT_SOURCE)
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["fiscal_year"])
    df["fiscal_year"] = df["fiscal_year"].astype(int)
    for c in NEWS_COLS:
        if c not in df.columns:
            df[c] = 0 if c == "news_count" else np.nan
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["ticker", "fiscal_year"] + NEWS_COLS].copy()


# --- yfinance (annual only), inlined from former collect_financials.py ---

def pull_financials_for_ticker(ticker: str):
    import yfinance as yf

    logger.info("Fetching financials for %s...", ticker)
    try:
        stock = yf.Ticker(ticker)
        return {
            "ticker": ticker,
            "info": stock.info,
            "balance_sheet": stock.balance_sheet,
            "income_statement": stock.financials,
            "cashflow": stock.cashflow,
        }
    except Exception as e:
        logger.error("Error fetching %s: %s", ticker, e)
        return None


def calculate_financial_ratios(balance_sheet: pd.Series, income_statement: pd.Series):
    ratios = {"liquid": None, "cumprof": None, "profitab": None, "leverage": None}
    if balance_sheet is None:
        return ratios

    def safe_get(series, keys, default=None):
        if series is None:
            return default
        for key in keys:
            if key in series.index:
                val = series[key]
                if pd.notna(val):
                    return val
        return default

    total_assets = safe_get(balance_sheet, ["Total Assets", "TotalAssets"])
    current_assets = safe_get(
        balance_sheet,
        ["Current Assets", "CurrentAssets", "Total Current Assets"],
    )
    current_liabilities = safe_get(
        balance_sheet,
        ["Current Liabilities", "CurrentLiabilities", "Total Current Liabilities"],
    )
    retained_earnings = safe_get(
        balance_sheet, ["Retained Earnings", "RetainedEarnings"]
    )
    total_equity = safe_get(
        balance_sheet,
        [
            "Total Stockholder Equity",
            "Total Equity Gross Minority Interest",
            "Stockholders Equity",
            "Total Equity",
            "TotalEquityGrossMinorityInterest",
        ],
    )
    total_liabilities = safe_get(
        balance_sheet,
        [
            "Total Liabilities Net Minority Interest",
            "Total Liab",
            "Total Liabilities",
            "TotalLiabilitiesNetMinorityInterest",
        ],
    )
    ebit = None
    if income_statement is not None:
        ebit = safe_get(
            income_statement, ["EBIT", "Operating Income", "OperatingIncome"]
        )

    if total_assets and total_assets != 0:
        if current_assets is not None and current_liabilities is not None:
            ratios["liquid"] = (current_assets - current_liabilities) / total_assets
        if retained_earnings is not None:
            ratios["cumprof"] = retained_earnings / total_assets
        if ebit is not None:
            ratios["profitab"] = ebit / total_assets
    if (
        total_equity is not None
        and total_liabilities is not None
        and total_liabilities != 0
    ):
        ratios["leverage"] = total_equity / total_liabilities
    return ratios


def extract_annual_fiscal_year_column(
    df: pd.DataFrame | None, target_year: int
) -> tuple[pd.Series | None, str | None]:
    if df is None or df.empty:
        return None, None
    preferred: list[tuple[pd.Timestamp, object]] = []
    fallback: list[tuple[pd.Timestamp, object]] = []
    for col in df.columns:
        try:
            ts = pd.Timestamp(col)
        except (TypeError, ValueError):
            continue
        if ts.year != target_year:
            continue
        if ts.month == 12 and ts.day == 31:
            preferred.append((ts, col))
        else:
            fallback.append((ts, col))

    def pick_best(pairs: list[tuple[pd.Timestamp, object]]):
        if not pairs:
            return None, None
        pairs.sort(key=lambda x: x[0], reverse=True)
        ts, col = pairs[0]
        return df[col], ts.strftime("%Y-%m-%d")

    s, end = pick_best(preferred)
    if s is not None:
        return s, end
    return pick_best(fallback)


def ratios_for_ticker_year(data: dict, fiscal_year: int) -> tuple[dict, bool, str | None]:
    bs_series, bs_end = extract_annual_fiscal_year_column(
        data.get("balance_sheet"), fiscal_year
    )
    is_series, is_end = extract_annual_fiscal_year_column(
        data.get("income_statement"), fiscal_year
    )
    ratios = calculate_financial_ratios(bs_series, is_series)
    period_end = bs_end or is_end
    if bs_series is None:
        return ratios, False, period_end

    def sg(series, keys):
        if series is None:
            return None
        for k in keys:
            if k in series.index and pd.notna(series[k]):
                return series[k]
        return None

    ta = sg(bs_series, ["Total Assets", "TotalAssets"])
    tl = sg(
        bs_series,
        [
            "Total Liabilities Net Minority Interest",
            "Total Liab",
            "Total Liabilities",
            "TotalLiabilitiesNetMinorityInterest",
        ],
    )
    te = sg(
        bs_series,
        [
            "Total Stockholder Equity",
            "Total Equity Gross Minority Interest",
            "Stockholders Equity",
            "Total Equity",
            "TotalEquityGrossMinorityInterest",
        ],
    )
    out = {**ratios, "total_assets": ta, "total_liabilities": tl, "total_equity": te}
    return out, True, period_end


def pull_financials_for_kams_rows(kams: pd.DataFrame) -> None:
    """Fetch yfinance annuals for KAM rows and merge into ratings_with_financials.csv."""
    ratings_extra = None
    if RATINGS_FIN.exists():
        ratings_extra = pd.read_csv(RATINGS_FIN)
        ratings_extra = _normalize_financial_ratings_df(ratings_extra)
        ratings_extra["fiscal_year"] = ratings_extra["fiscal_year"].astype(int)
        rcols = ["ticker", "fiscal_year", "rating_date"]
        rcols = [c for c in rcols if c in ratings_extra.columns]
        ratings_extra = ratings_extra[rcols].drop_duplicates(
            subset=["ticker", "fiscal_year"]
        )

    cache: dict[str, dict | None] = {}
    for i, ticker in enumerate(kams["ticker"].unique(), 1):
        print(f"[{i}/{kams['ticker'].nunique()}] yfinance {ticker} (annual)...")
        cache[ticker] = pull_financials_for_ticker(ticker)
        time.sleep(SLEEP_S)

    rows = []
    for _, row in kams.iterrows():
        ticker = row["ticker"]
        fy = int(row["fiscal_year"])
        company_name = row.get("company_name", "")
        rating_kam = row.get("rating", "")
        data = cache.get(ticker)
        sector, industry = "", ""
        if data and data.get("info"):
            info = data["info"] or {}
            sector = str(info.get("sector") or "")
            industry = str(info.get("industry") or "")
        rd = ""
        if ratings_extra is not None:
            m = ratings_extra[
                (ratings_extra["ticker"] == ticker)
                & (ratings_extra["fiscal_year"] == fy)
            ]
            if not m.empty:
                m0 = m.iloc[0]
                rd = str(m0.get("rating_date", "") or "")

        if data is None:
            rows.append(
                {
                    "ticker": ticker,
                    "company_name": company_name,
                    "rating_date": rd,
                    "rating": rating_kam,
                    "fiscal_year": fy,
                    "liquid": None,
                    "cumprof": None,
                    "profitab": None,
                    "leverage": None,
                    "total_assets": None,
                    "total_liabilities": None,
                    "total_equity": None,
                    "yfinance_sector": sector,
                    "yfinance_industry": industry,
                    "statement_period_end": None,
                }
            )
            continue

        rdict, _ok, period_end = ratios_for_ticker_year(data, fy)
        rows.append(
            {
                "ticker": ticker,
                "company_name": company_name,
                "rating_date": rd,
                "rating": rating_kam,
                "fiscal_year": fy,
                "liquid": rdict.get("liquid"),
                "cumprof": rdict.get("cumprof"),
                "profitab": rdict.get("profitab"),
                "leverage": rdict.get("leverage"),
                "total_assets": rdict.get("total_assets"),
                "total_liabilities": rdict.get("total_liabilities"),
                "total_equity": rdict.get("total_equity"),
                "yfinance_sector": sector,
                "yfinance_industry": industry,
                "statement_period_end": period_end,
            }
        )

    out_df = pd.DataFrame(rows)
    if RATINGS_FIN.exists():
        base = pd.read_csv(RATINGS_FIN)
        base = _normalize_financial_ratings_df(base)
        base["fiscal_year"] = base["fiscal_year"].astype(int)
        fin_cols = [
            "liquid",
            "cumprof",
            "profitab",
            "leverage",
            "total_assets",
            "total_liabilities",
            "total_equity",
            "yfinance_sector",
            "yfinance_industry",
            "statement_period_end",
        ]
        right = out_df[["ticker", "fiscal_year"] + fin_cols].copy()
        for c in fin_cols:
            if c in base.columns:
                base = base.drop(columns=[c])
        base = base.merge(right, on=["ticker", "fiscal_year"], how="left")
        have = set(zip(base["ticker"], base["fiscal_year"]))
        for _, r in out_df.iterrows():
            k = (r["ticker"], int(r["fiscal_year"]))
            if k not in have:
                nr = {
                    c: (r[c] if c in r.index and pd.notna(r[c]) else None)
                    for c in base.columns
                }
                base = pd.concat([base, pd.DataFrame([nr])], ignore_index=True)
                have.add(k)
        base.to_csv(RATINGS_FIN, index=False)
        logger.info("Updated %s (%s rows)", RATINGS_FIN.name, len(base))


def build_merged_multisource(
    fin: pd.DataFrame, kams: pd.DataFrame, news: pd.DataFrame
) -> pd.DataFrame:
    m = fin.merge(kams, on=["ticker", "fiscal_year"], how="inner", suffixes=("", "_kam"))
    drop_suffix = [c for c in m.columns if c.endswith("_kam")]
    m = m.drop(columns=drop_suffix, errors="ignore")
    m = m.merge(news, on=["ticker", "fiscal_year"], how="inner")
    if "yfinance_sector" in m.columns:
        m["sector"] = m["yfinance_sector"].fillna("")
    else:
        m["sector"] = ""
    m["rating_agency"] = "Tassnief"
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--yfinance",
        action="store_true",
        help="Pull annual financials from Yahoo for each KAM (ticker, year)",
    )
    args = ap.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)

    ratings_enriched = pd.DataFrame()
    if FINANCIALS_SOURCE.exists():
        ratings_enriched = load_merge_and_persist_ratings_financials()

    kams_raw = build_kams_processed()
    fin_tickers = financial_ticker_set(ratings_enriched, kams_raw)
    if fin_tickers:
        logger.info(
            "Excluding financial tickers from processed outputs: %s",
            sorted(fin_tickers),
        )

    kams = drop_financial_tickers(kams_raw, fin_tickers)
    kams.to_csv(OUT_KAMS, index=False)
    print(f"Wrote {OUT_KAMS} ({len(kams)} rows)")

    if args.yfinance:
        pull_financials_for_kams_rows(kams)
        if not FINANCIALS_SOURCE.exists():
            raise SystemExit(
                f"Expected {FINANCIALS_SOURCE} after yfinance merge; file missing."
            )
        ratings_enriched = load_merge_and_persist_ratings_financials()
    elif ratings_enriched.empty:
        raise SystemExit(f"Missing {FINANCIALS_SOURCE}; use --yfinance or restore CSV.")

    fin = drop_financial_tickers(
        financial_processed_subset(ratings_enriched),
        fin_tickers,
    )
    fin = align_to_kams_panel(kams, fin)
    fin.to_csv(OUT_FIN, index=False)
    print(f"Wrote {OUT_FIN} ({len(fin)} rows, aligned to KAM panel)")

    news = build_news_features_processed()
    news = drop_financial_tickers(news, fin_tickers)
    news = finalize_news_features_panel(align_to_kams_panel(kams, news))
    news.to_csv(OUT_NEWS, index=False)
    print(f"Wrote {OUT_NEWS} ({len(news)} rows, aligned to KAM panel)")

    if OUT_FINANCIALS_MULTI.exists():
        raw_multi = pd.read_csv(OUT_FINANCIALS_MULTI)
        raw_multi = drop_financial_tickers(raw_multi, fin_tickers)
        aligned_multi = align_financials_processed_to_kams(kams, raw_multi)
        aligned_multi.to_csv(OUT_FINANCIALS_MULTI, index=False)
        print(
            f"Wrote {OUT_FINANCIALS_MULTI} ({len(aligned_multi)} rows, aligned to KAM panel)"
        )

    merged = build_merged_multisource(fin, kams, news)
    merged.to_csv(OUT_MERGED, index=False)
    print(f"Wrote {OUT_MERGED} ({len(merged)} rows, inner join)")


if __name__ == "__main__":
    main()
