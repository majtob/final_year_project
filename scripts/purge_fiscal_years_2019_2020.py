#!/usr/bin/env python3
"""Remove rows with fiscal_year or fiscal_year_applicable in {2019, 2020} from all CSVs under data/."""

from __future__ import annotations

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
DROP = frozenset({2019, 2020})
YEAR_COLS = ("fiscal_year", "fiscal_year_applicable")


def _year_int(val: str) -> int | None:
    v = str(val).strip()
    if not v:
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def keep_row(row: dict[str, str], fieldnames: list[str]) -> bool:
    for col in YEAR_COLS:
        if col not in fieldnames:
            continue
        y = _year_int(row.get(col, ""))
        if y is not None and y in DROP:
            return False
    return True


def filter_csv(path: Path) -> tuple[bool, int, int]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            return False, 0, 0
        if not any(c in fieldnames for c in YEAR_COLS):
            return False, 0, 0
        rows_in = list(reader)
    kept = [r for r in rows_in if keep_row(r, fieldnames)]
    if len(kept) == len(rows_in):
        return False, len(rows_in), len(kept)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)
    return True, len(rows_in), len(kept)


def delete_news_years() -> int:
    n = 0
    news = DATA / "raw" / "news"
    if not news.is_dir():
        return 0
    for y in DROP:
        for p in news.glob(f"*_{y}_news.json"):
            p.unlink()
            n += 1
    return n


def main() -> None:
    changed = 0
    for path in sorted(DATA.rglob("*.csv")):
        did, n_in, n_out = filter_csv(path)
        if did:
            rel = path.relative_to(PROJECT_ROOT)
            print(f"{rel}: {n_in} -> {n_out} rows")
            changed += 1
    nd = delete_news_years()
    print(f"Deleted {nd} raw news JSON file(s) for 2019/2020.")
    print(f"Updated {changed} CSV file(s).")


if __name__ == "__main__":
    main()
