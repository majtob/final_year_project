## Project Skeleton

```
final_project/
│
├── README.md                      # Quickstart, architecture, run commands
├── LICENSE
├── .env.example                   # API keys/paths placeholders
├── requirements.txt / environment.yml
├── docker-compose.yml (optional)
│
├── data/
│   ├── raw/                       # Mirrored filings, PDFs, HTML, XLS
│   ├── interim/                   # OCR/text chunks, parsed tables
│   ├── processed/                 # Tidy company-period tables, ratio inputs
│   └── market/                    # yfinance extracts (CSV/Parquet)
│
├── dvc.yaml / dvc.lock (if using DVC)
│
├── notebooks/
│   ├── 01_exploration.ipynb       # Inspect filings, schema notes
│   ├── 02_preprocessing_tests.ipynb
│   ├── 03_rag_prototype.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── config/                    # Hydra/OmegaConf configs (model, schema, paths)
│   ├── pipelines/
│   │   ├── ingest_filings.py
│   │   ├── preprocess_documents.py
│   │   ├── build_vector_store.py
│   │   └── extract_financials.py
│   ├── retrieval/
│   │   ├── index_builder.py       # FAISS/Chroma wrapper
│   │   └── retriever.py           # Similarity search + filters
│   ├── models/
│   │   ├── prompts.py             # Prompt templates, JSON schema enforcement
│   │   └── rag_runner.py          # Orchestration for open-source LLM calls
│   ├── evaluation/
│   │   ├── ratio_baseline.py      # Deterministic ratio calculations
│   │   ├── risk_flagger.py        # Heuristic score/binary flag
│   │   └── metrics.py             # NHR, citation accuracy, schema compliance
│   ├── utils/
│   │   ├── io.py                  # Storage helpers (DuckDB/Postgres)
│   │   ├── logging.py
│   │   └── validation.py          # Pandera/Great Expectations wrappers
│   └── cli.py                     # Prefect/Airflow entry points or Typer CLI
│
├── configs/                       # Pipeline schedules, model settings
│   ├── preprocessing.yaml
│   ├── retrieval.yaml
│   ├── evaluation.yaml
│   └── environment.yaml
│
├── scripts/
│   ├── sync_market_data.py        # yfinance fetch, coverage checks
│   ├── run_pipeline.py            # End-to-end orchestration
│   └── monitor_quality.py         # Generate QA reports
│
├── tests/
│   ├── test_parsers.py
│   ├── test_schema.py
│   ├── test_rag_outputs.py
│   └── fixtures/
│
├── docs/
│   ├── architecture.md
│   ├── data_dictionary.md
│   ├── evaluation_report.md
│   └── progress_log.md
│
└── logs/
    ├── pipeline/                  # Prefect/Airflow runs
    └── evaluation/
```

### Checkpoints

- `data/raw` populated and hashed before running `pipelines/ingest_filings.py`.
- `pipelines/preprocess_documents.py` outputs schema-valid tables validated via `tests/test_parsers.py`.
- `retrieval/index_builder.py` builds the vector store; smoke test with `retriever.py`.
- `models/rag_runner.py` produces citation-backed JSON validated by `tests/test_rag_outputs.py`.
- `evaluation/ratio_baseline.py` plus `risk_flagger.py` generate heuristics-only risk flags documented in `docs/evaluation_report.md`.
- `scripts/sync_market_data.py` verifies yfinance coverage and records gaps in `docs/data_dictionary.md`.

