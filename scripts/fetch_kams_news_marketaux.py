#!/usr/bin/env python3
"""
Fetch English news from MarketAux for (ticker, fiscal_year) rows in
kams_processed.csv that do not yet have a matching file under data/raw/news/.

Output files match the existing cache layout:
  data/raw/news/{NUM}_SR_{YYYY}_news.json

Requires MARKETAUX_API_KEY (or MARKET_AUX_API_TOKEN) in the environment, or a
mxa1438/.env file with either variable (python-dotenv optional).

After fetching, re-run FinBERT + rebuild:
  python scripts/collect_news_sentiment_finbert.py
  python scripts/rebuild_processed_datasets.py

Minimal API usage (only rows with no file or empty articles), from repo root:
  python scripts/fetch_kams_news_marketaux.py --dry-run --refetch-empty   # preview
  python scripts/fetch_kams_news_marketaux.py --refetch-empty             # fetch those only

Surgical rerun (subset), e.g. one company-year:
  python scripts/fetch_kams_news_marketaux.py --refetch-empty --only-pairs 7204.SR:2024
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KAMS_CSV = PROJECT_ROOT / "data" / "processed" / "kams_processed.csv"
NEWS_DIR = PROJECT_ROOT / "data" / "raw" / "news"
API_URL = "https://api.marketaux.com/v1/news/all"

# Kams template ticker -> older news cache ticker (same issuer, renamed code).
NEWS_TICKER_ALIASES: dict[str, str] = {
    "7204.SR": "2070.SR",  # Perfect Presentation: historical JSON used 2070.SR
}

# Extra MarketAux `search` terms when symbol filters return nothing.
EXTRA_SEARCH_BY_TICKER: dict[str, list[str]] = {
    "2222.SR": ["Aramco", "Saudi Aramco", "Saudi Arabian Oil"],
    "3008.SR": ["Al Kathiri Holding", "Kathiri Holding", "AKHC"],
    "4321.SR": ["Cenomi Centers", "Cenomi", "Arabian Centres"],
    "5110.SR": [
        "Saudi Electricity Company",
        "Saudi Electricity",
        "Saudi Energy",
    ],
}


def ticker_to_file_stem(ticker: str) -> str:
    """7010.SR -> 7010_SR"""
    return ticker.replace(".", "_")


def news_path(ticker: str, year: int) -> Path:
    return NEWS_DIR / f"{ticker_to_file_stem(ticker)}_{year}_news.json"


def _path_to_ticker_year(path: Path) -> tuple[str, int] | None:
    m = re.match(r"^(.+)_(\d{4})_news\.json$", path.name, re.I)
    if not m:
        return None
    body, ys = m.group(1), int(m.group(2))
    if "_" not in body:
        return None
    num, suf = body.rsplit("_", 1)
    return f"{num}.{suf}", ys


def parse_existing_news_keys() -> set[tuple[str, int]]:
    have: set[tuple[str, int]] = set()
    for p in NEWS_DIR.glob("*_news.json"):
        parsed = _path_to_ticker_year(p)
        if parsed:
            have.add(parsed)
    return have


def pairs_with_empty_news_files() -> set[tuple[str, int]]:
    """Files that exist but have no articles (retry targets)."""
    empty: set[tuple[str, int]] = set()
    for p in NEWS_DIR.glob("*_news.json"):
        parsed = _path_to_ticker_year(p)
        if not parsed:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        arts = data.get("articles") or []
        if len(arts) == 0:
            empty.add(parsed)
    return empty


def http_get_with_retry(
    session: requests.Session, url: str, params: dict, *, timeout: int = 90
) -> requests.Response:
    """GET with backoff on HTTP 429 (does not log api_token)."""
    last: requests.Response | None = None
    for attempt in range(12):
        last = session.get(url, params=params, timeout=timeout)
        if last.status_code != 429:
            last.raise_for_status()
            return last
        wait = min(120, 25 + attempt * 12)
        print(f"    (429 rate limit, sleeping {wait}s …)")
        time.sleep(wait)
    assert last is not None
    last.raise_for_status()
    return last


def load_kams_pairs() -> list[tuple[str, int, str]]:
    rows = list(csv.DictReader(KAMS_CSV.open(encoding="utf-8")))
    out: list[tuple[str, int, str]] = []
    for r in rows:
        t = r["ticker"].strip()
        y = int(r["fiscal_year"])
        name = (r.get("company_name") or "").strip()
        out.append((t, y, name))
    return out


def search_phrase(company_name: str) -> str:
    """Short phrase for MarketAux `search` fallback."""
    s = company_name.strip()
    s = re.sub(r"\([^)]*\)", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s[:120] if s else company_name[:120]


def article_from_ma(item: dict, ticker: str, fallback_search: str) -> dict:
    search_term = fallback_search
    for ent in item.get("entities") or []:
        sym = (ent.get("symbol") or "").strip()
        if sym:
            search_term = sym
            break
    return {
        "ticker": ticker,
        "title": (item.get("title") or "").strip(),
        "description": ((item.get("description") or "").strip())[:4000],
        "snippet": ((item.get("snippet") or "").strip())[:4000],
        "published_at": item.get("published_at"),
        "source_name": (item.get("source") or "").strip(),
        "url": (item.get("url") or "").strip(),
        "search_term": search_term,
        "sentiment_compound": 0.0,
        "sentiment_pos": 0.0,
        "sentiment_neg": 0.0,
        "sentiment_neu": 1.0,
    }


def fetch_year(
    token: str,
    ticker: str,
    year: int,
    company_name: str,
    *,
    limit: int,
    max_pages: int,
    session: requests.Session,
) -> list[dict]:
    """Try symbol filter first, then full-text search."""
    published_after = f"{year}-01-01"
    published_before = f"{year}-12-31T23:59:59"
    phrase = search_phrase(company_name)

    def run_params(extra: dict) -> list[dict]:
        articles: list[dict] = []
        seen_urls: set[str] = set()
        for page in range(1, max_pages + 1):
            params = {
                "api_token": token,
                "language": "en",
                "published_after": published_after,
                "published_before": published_before,
                "limit": limit,
                "page": page,
                "group_similar": "false",
                **extra,
            }
            r = http_get_with_retry(session, API_URL, params, timeout=90)
            data = r.json()
            if data.get("error"):
                raise RuntimeError(data["error"])
            batch = data.get("data") or []
            for item in batch:
                url = (item.get("url") or "").strip()
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                articles.append(
                    article_from_ma(item, ticker, phrase or ticker)
                )
            meta = data.get("meta") or {}
            if len(batch) < limit:
                break
            if meta.get("returned", 0) == 0:
                break
        return articles

    def search_articles(term: str) -> list[dict]:
        t = term.strip()
        if not t:
            return []
        q = f'"{t}"' if " " in t else t
        return run_params({"search": q})

    # 0) Symbol only (broader than filter_entities)
    sym_articles = run_params({"symbols": ticker})
    if sym_articles:
        return sym_articles

    # 1) Symbol + entity filter
    sym_articles = run_params({"symbols": ticker, "filter_entities": "true"})
    if sym_articles:
        return sym_articles

    # 2) Same but restrict to Saudi exchange country (when API tags entities as SA)
    sym_articles = run_params(
        {"symbols": ticker, "filter_entities": "true", "countries": "sa"}
    )
    if sym_articles:
        return sym_articles

    # 3) Full-text search on company name (quoted phrase if multi-word)
    if phrase:
        arts = search_articles(phrase)
        if arts:
            return arts

    for extra in EXTRA_SEARCH_BY_TICKER.get(ticker, []):
        arts = search_articles(extra)
        if arts:
            return arts

    return []


def copy_alias_news(
    need: list[tuple[str, int, str]], *, skip: bool
) -> list[tuple[str, int, str]]:
    """Duplicate news JSON from an older ticker when the kams row uses a new code."""
    if skip:
        return need
    remaining: list[tuple[str, int, str]] = []
    for ticker, year, company_name in need:
        src_ticker = NEWS_TICKER_ALIASES.get(ticker)
        if not src_ticker:
            remaining.append((ticker, year, company_name))
            continue
        src = news_path(src_ticker, year)
        dst = news_path(ticker, year)
        if not src.exists():
            remaining.append((ticker, year, company_name))
            continue
        data = json.loads(src.read_text(encoding="utf-8"))
        data["ticker"] = ticker
        for art in data.get("articles") or []:
            art["ticker"] = ticker
        NEWS_DIR.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Alias copy: {src.name} -> {dst.name} ({company_name})")
    return remaining


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch missing kams news from MarketAux")
    ap.add_argument(
        "--skip-alias",
        action="store_true",
        help="Do not copy from NEWS_TICKER_ALIASES (e.g. 2070.SR -> 7204.SR)",
    )
    ap.add_argument("--dry-run", action="store_true", help="List missing pairs only")
    ap.add_argument(
        "--refetch-empty",
        action="store_true",
        help="Re-fetch files that exist but have zero articles (e.g. after rate limits)",
    )
    ap.add_argument(
        "--only-pairs",
        nargs="*",
        metavar="TICKER:YEAR",
        help=(
            "After computing missing/empty pairs, keep only these (e.g. 7204.SR:2024 "
            "5110.SR:2021). Useful to stay under API quota."
        ),
    )
    ap.add_argument("--limit", type=int, default=12, help="Articles per API page")
    ap.add_argument("--max-pages", type=int, default=3, help="Max pages per query strategy")
    ap.add_argument("--sleep", type=float, default=2.5, help="Seconds between API calls")
    args = ap.parse_args()

    have = parse_existing_news_keys()
    if args.refetch_empty:
        empty = pairs_with_empty_news_files()
        if empty:
            print(f"Refetch-empty: treating {len(empty)} (ticker,year) with 0 articles as missing.")
        have -= empty

    pairs = load_kams_pairs()
    need = sorted({(t, y, n) for t, y, n in pairs if (t, y) not in have})

    if args.only_pairs:
        allowed: set[tuple[str, int]] = set()
        for s in args.only_pairs:
            if ":" not in s:
                raise SystemExit(
                    f"Bad --only-pairs entry {s!r}: use TICKER:YEAR (e.g. 7204.SR:2024)"
                )
            t_part, y_part = s.rsplit(":", 1)
            allowed.add((t_part.strip(), int(y_part.strip())))
        before = len(need)
        need = [row for row in need if (row[0], row[1]) in allowed]
        if before and not need:
            raise SystemExit(
                "--only-pairs did not match any of the computed missing/empty rows. "
                "Check ticker spelling and fiscal year."
            )
        fetched_keys = {(t, y) for t, y, _ in need}
        not_in_gap_set = allowed - fetched_keys
        if not_in_gap_set:
            print(
                "Note: these --only-pairs are not in the current missing/empty set "
                "(skip fetching; already have articles or not in kams_processed): "
                f"{sorted(not_in_gap_set)}"
            )

    if not need:
        print("No missing (ticker, fiscal_year) relative to kams_processed.csv.")
        if args.refetch_empty:
            print("(No empty files to refetch, or none match kams rows.)")
        return

    print(f"Missing {len(need)} company-year pairs (vs kams_processed.csv).")
    for t, y, n in need:
        print(f"  {t} {y} — {n}")

    if args.dry_run:
        still_api = []
        would_alias = 0
        for t, y, n in need:
            if not args.skip_alias:
                src_t = NEWS_TICKER_ALIASES.get(t)
                if src_t and news_path(src_t, y).exists():
                    would_alias += 1
                    continue
            still_api.append((t, y, n))
        if would_alias:
            print(
                f"\n(--dry-run) Would alias-copy {would_alias} file(s); "
                f"{len(still_api)} would still need MarketAux."
            )
        else:
            print(f"\n(--dry-run) All {len(need)} would need MarketAux (no alias sources).")
        if still_api:
            print("Still need API:")
            for t, y, n in still_api:
                print(f"  {t} {y} — {n}")
        return

    need = copy_alias_news(need, skip=args.skip_alias)
    if not need:
        print("All gaps filled via ticker aliases. Nothing to fetch.")
        return

    token = (
        os.environ.get("MARKETAUX_API_KEY")
        or os.environ.get("MARKET_AUX_API_TOKEN")
        or os.environ.get("MARKET_AUX_KEY")
    )
    if not token:
        raise SystemExit(
            f"{len(need)} pairs still need MarketAux. Set MARKETAUX_API_KEY in the "
            "environment or mxa1438/.env, then re-run this script (see docstring)."
        )

    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "mxa1438-fetch-kams-news/1.0"})

    for i, (ticker, year, company_name) in enumerate(need):
        path = news_path(ticker, year)
        print(f"[{i+1}/{len(need)}] Fetching {ticker} {year} -> {path.name} ...")
        try:
            articles = fetch_year(
                token,
                ticker,
                year,
                company_name,
                limit=args.limit,
                max_pages=args.max_pages,
                session=session,
            )
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            print(f"  ERROR: HTTP {code} from MarketAux (request URL omitted).")
            articles = []
        except Exception as e:
            print(f"  ERROR: {e}")
            articles = []

        payload = {
            "ticker": ticker,
            "year": year,
            "fetch_date": datetime.now(timezone.utc).isoformat(),
            "article_count": len(articles),
            "articles": articles,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  wrote {len(articles)} articles")
        if i < len(need) - 1:
            time.sleep(args.sleep)

    print("Done. Next: python scripts/collect_news_sentiment_finbert.py && python scripts/rebuild_processed_datasets.py")


if __name__ == "__main__":
    main()
