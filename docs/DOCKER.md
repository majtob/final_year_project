# Docker (examiner / reproducible demo)

This project supports a **small CPU-only Docker image** that runs the **Streamlit** app (`app.py`). It matches the **hybrid** workflow: examiners get a consistent environment; you keep using a **local GPU + venv** for QLoRA training and fine-tuned inference.

## What the Docker image includes

- Python 3.11 + `requirements-docker.txt` (Streamlit, scikit-learn, SHAP, matplotlib, pandas, …)
- **No** PyTorch / `transformers` / bitsandbytes — keeps the image fast to build and suitable for laptops without a GPU
- The demo uses **template-based** verdict text in the UI (same as current `app.py`); it does **not** load Qwen inside the container

## What the Docker image does *not* include

- **Fine-tuned Qwen 2.5 3B** inference — run `models/llm_verdict.py` or extend the image with CUDA + torch if you need that in a container
- **Ollama** — optional; run Ollama on the host and point the app to it only if you change the app to call it from the container network

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) or Docker Engine (Linux)

## Build and run

From the repository root:

```bash
docker build -t mxa1438-demo .
docker run --rm -p 8501:8501 mxa1438-demo
```

Or with Compose:

```bash
docker compose up --build
```

Then open **http://localhost:8501** in a browser.

## Development vs examiner

| Goal | Approach |
|------|----------|
| **Examiner / marker** | Use `Dockerfile` + `docker compose up` |
| **You (GPU, training, evaluate)** | Local venv: `python models/finetune_qwen.py`, `python models/evaluate_finetune.py` |
| **Larger “full stack” image** | Possible but not checked in — would add `torch`, `transformers`, `peft`, and optional NVIDIA runtime |

## Files

| File | Role |
|------|------|
| `Dockerfile` | Image definition |
| `docker-compose.yml` | One-command `docker compose up` |
| `requirements-docker.txt` | Slim deps for the demo only |
| `.dockerignore` | Skips `venv`, checkpoints, cache dirs |

## Troubleshooting

- **Port in use:** change the host port, e.g. `docker run -p 8502:8501 mxa1438-demo`
- **Missing data:** ensure `data/processed/merged_multisource_training.csv` (built by `python scripts/rebuild_processed_datasets.py`), plus `figures/` and `results/` if the app references them, are present in the build context (they are not excluded by `.dockerignore`)
