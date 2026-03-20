# CPU-only image for the Streamlit credit-rating demo (examiner-friendly).
# For GPU / fine-tuned Qwen development, use a local venv — see docs/DOCKER.md

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# shap / sklearn wheels are prebuilt for manylinux; slim image stays small
COPY requirements-docker.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements-docker.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
