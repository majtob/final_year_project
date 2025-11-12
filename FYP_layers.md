Open-Source LLMs for Credit Risk on Saudi Exchange Financials (2021–2024): Retrieval-Augmented Extraction and Calibration.


Definition Layer

Aim: evaluate whether open-source LLMs can perform credit analysis for Saudi listed companies using only public filings.
Scope: limited to main-market firms, English and Arabic financial materials, fiscal years 2021–2024. All non-public data sources (banks, terminals, rating agencies) are excluded. The study focuses on LLM-based extraction accuracy, hallucination control, and signal calibration, not on developing a proprietary credit rating model.

Research Questions / Hypotheses

1.Can retrieval-augmented open LLMs extract key financial ratios and risk indicators from Tadawul filings with acceptable accuracy?

2.Do hallucination-aware extraction and calibration methods reduce factual and numeric errors relative to unconstrained LLM outputs?

3.Are the resulting credit risk signals consistent with basic financial ratio benchmarks?

Deliverables
– Data pipeline: a reproducible ingestion and normalisation process converting Tadawul disclosures into structured datasets.
– Extraction and evaluation framework: RAG + schema-constrained LLM system capable of producing JSON outputs with citation links, plus evaluation metrics.
– Credit risk outputs: per-company, per-period ratio set and risk score.
– Documentation: reproducible codebase, technical report, evaluation summary.

Constraints
– Public data only; no paid financial terminals or private credit data.
– Use of open-source LLMs (e.g., Mistral, Llama, Falcon) only.
– Fixed temporal coverage (2021–2024) to ensure comparability and manageable scope.
– All outputs must be traceable to exact source text or table to verify factual grounding.
– Ethical and compliance restrictions: no personal data, no scraping beyond allowed endpoints.

Data and Infrastructure Layer

Data Inventory
Full catalogue of all inputs. Includes Saudi Exchange company filings (annual reports, financial statements, board reports, disclosures), fiscal years 2021–2024, URLs, file formats (PDF, HTML, XLS), and metadata (company name, ticker, language, publication date).
Add external market data via yfinance for each ticker (e.g., 2222.SR). Use it to obtain open-source price, volume, and market-cap data for market ratio and calibration tasks. This creates two parallel data channels: filings data and market data.

Collection
Automated retrieval of Tadawul filings and related documents using scripted scraping (with requests, aiohttp, BeautifulSoup4, lxml). Download logs include URL, timestamp, and hash for reproducibility. Integrate a yfinance routine to pull daily or quarterly market data. Implement rate-limit controls, retries, and missing-file detection. Output: complete mirrored dataset stored by company and fiscal year.

Pre-Processing
Convert all collected documents into machine-readable form.
– OCR for scanned PDFs (pytesseract, docTR).
– Table and text parsing (pdfplumber, pymupdf, BeautifulSoup4).
– HTML cleanup and tag normalization.
– Language detection (langdetect) and Unicode repair (ftfy, unidecode).
– Numeric standardization (comma/decimal, percentage normalization).
Output: structured text-table pairs with consistent formatting.

Schema Design
Define a unified “tidy” schema: one record per company-period with standardised financial line items (balance sheet, income statement, cash flow, ratios). Include source metadata (file ID, URL, section, page reference). Market data columns (price, market cap, volume) are merged by ticker and date. Schema enforced with Pandera or Pydantic models.

Infrastructure
– Storage: DuckDB or PostgreSQL for structured tables; filesystem hierarchy for raw filings.
– Processing: Python-based ETL pipeline (Pandas, DVC for data versioning).
– Pipeline Orchestration: Prefect or Airflow for scheduled data refresh and dependency tracking.
– Environment: reproducible setup with Docker or Conda environments.
– Logging: loguru or native logging for persistent run records.

Data Versioning and Lineage
DVC or git-lfs to track data and model artefacts. Each transformation stage records lineage: which raw file produced which structured row. Enables rollback and reproducibility checks.

Quality Checks
Automated validation using Great Expectations or custom Pandera rules. Tests for schema integrity, missing values, OCR errors, numeric validity, and temporal consistency between filings and market data. Store quality reports as structured logs.

Modeling and Evaluation Layer

Baseline Calculation Module
Implements deterministic financial ratio computation. Inputs: structured financial data from Layer 2. Outputs: liquidity, leverage, coverage, and profitability ratios for each company-period. Ratios follow standard definitions (current ratio, debt-to-equity, interest coverage, return on assets, net margin). This establishes a non-LLM baseline for all subsequent comparisons.

Extraction Pipeline (RAG + Schema Enforcement)
Retrieval-augmented generation system built on open-source LLMs (Mistral, Llama 3, Falcon).
– Document retrieval uses vector search (FAISS, Chroma, or ElasticSearch) over preprocessed filings.
– Prompt templates enforce JSON schema outputs corresponding to the defined data schema.
– Each extracted value must include its citation (document ID, section, page, or line reference).
– Post-processing validates JSON against schema constraints.

Risk Scoring Module
Combines deterministic ratios and extracted features into binary and calibrated risk scores.
– Binary flag: 0 = low credit risk, 1 = elevated credit risk.
– Continuous score: probability estimate between 0 and 1, derived through logistic regression or isotonic calibration on validation data.
– Optional integration with market signals (e.g., volatility, market-cap trend) from yfinance.

Data Assessment Framework and Expert Validation
Defines evaluation rubric for model outputs.
– Criteria: factual correctness, numeric accuracy, citation validity, completeness, interpretive consistency.
– Expert independently rates a representative sample of extractions.
– Ratings form the ground-truth dataset.
– Ground truth used to compute quantitative metrics (precision, recall, numeric deviation, hallucination rate).

Evaluation Metrics
– Numeric Hallucination Rate (NHR): proportion of extracted numbers without verifiable source match.
– Citation Accuracy (CA): share of cited spans that correctly support extracted values.
– Schema Compliance (SC): proportion of valid JSON outputs passing all field checks.
– Calibration Error (CE): difference between predicted risk score and empirical frequency of risk events.
– Overall Accuracy (OA): average of factual correctness across categories.

Model Comparison and Iteration
Evaluate multiple open-source models under identical prompts and retrieval setups. Compare performance on accuracy, hallucination control, and computation cost. Use the expert-validated ground truth for benchmarking. Document all experiments with configuration files and fixed random seeds.

Output Validation and Logging
All model outputs stored with full metadata: model name, prompt ID, data version, timestamp, and citation list. Evaluation reports automatically generated and versioned for reproducibility.

Project Management Layer

Timeline
Structured six-month progression ensuring sequential dependency and measurable checkpoints.
– Month 1: environment setup, data inventory completion, collection scripts tested.
– Month 2: pre-processing and schema validation operational; OCR and table extraction verified.
– Month 3: retrieval pipeline and vector index built; first extraction prototypes tested on sample filings.
– Month 4: ratio calculator and baseline risk scoring implemented; integration of yfinance data.
– Month 5: data assessment framework finalized; expert evaluation executed; hallucination metrics computed.
– Month 6: calibration, full validation, report writing, and repository finalization.

Milestones

Dataset fully collected and normalized.

Baseline ratio module operational.

RAG extraction producing schema-valid JSON outputs.

Expert-validated ground truth established.

Model evaluation report completed with hallucination and calibration metrics.

Final reproducible repository and documentation delivered.

Version Control and Reproducibility
– Git for source code and scripts.
– DVC or git-lfs for data and model artefact versioning.
– Tagged releases for each major milestone.
– All experiments logged with configuration snapshots.

Progress Monitoring
– Weekly progress log summarizing work completed, blockers, and next tasks.
– Error tracker for failed document parses or schema validation issues.
– Automated run reports generated after each batch extraction and evaluation.

Documentation and Reporting
– README and pipeline guide explaining environment setup, directory structure, and run commands.
– Technical report describing system architecture, methods, evaluation, and findings.
– Appendix tables summarizing metrics and expert assessment outcomes.

Risk Management
– Backup of all collected filings and intermediate data.
– Redundant logs for scraping and OCR to prevent reprocessing loss.
– Contingency schedule for expert delays or failed extractions.

Completion Criteria
All deliverables reproducible from raw data to evaluation results using open-source components, with complete documentation and validated credit risk outputs.