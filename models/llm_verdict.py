"""
LLM Verdict Generator for Credit Rating Predictions.

Supports three generation modes:
  1. fine-tuned  -- Qwen 2.5 3B with QLoRA adapter (recommended)
  2. ollama      -- Zero-shot via local Ollama (Mistral/Llama)
  3. template    -- Deterministic rule-based fallback

Falls back through the chain automatically:
  fine-tuned -> ollama -> template
"""

import json
import sys
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.multisource_data import FULL_FEATURE_COLS

VERDICTS_DIR = PROJECT_ROOT / "results" / "verdicts"
# No pre-merged file for the 21-feature frame: it is built from the processed
# CSVs by multisource_data. None means "let that module build it".
DATA_FILE = None
ADAPTER_DIR = PROJECT_ROOT / "models" / "lora_adapter"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODELS = ["mistral", "llama3", "llama3.1"]

QWEN_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
QWEN_MAX_SEQ_LENGTH = 1024

FINETUNED_SYSTEM_PROMPT = (
    "You are a credit analyst specializing in Saudi Exchange (Tadawul) listed companies. "
    "Given financial ratios, auditor Key Audit Matter (KAM) dummies, FinBERT news aggregates, "
    "and a multicategory ML prediction (AA/A/BBB/BB), produce a structured JSON credit verdict. "
    "Cite specific numbers from the prompt only; do not invent facts."
)

_finetuned_model = None
_finetuned_tokenizer = None

RATING_TO_NUMERIC = {
    'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18,
    'A+': 17, 'A': 16, 'A-': 15,
    'BBB+': 14, 'BBB': 13, 'BBB-': 12,
    'BB+': 11, 'BB': 10, 'BB-': 9,
    'B+': 8, 'B': 7, 'B-': 6,
    'CCC+': 5, 'CCC': 4, 'CCC-': 3,
    'CC': 2, 'C': 1, 'D': 0,
}

NUMERIC_TO_RATING = {v: k for k, v in RATING_TO_NUMERIC.items()}

RATIO_DESCRIPTIONS = {
    'liquid': ('Liquidity (Working Capital / Total Assets)',
               'Measures short-term financial health. Higher values indicate '
               'greater ability to meet short-term obligations.'),
    'cumprof': ('Cumulative Profitability (Retained Earnings / Total Assets)',
                'Reflects accumulated profits relative to size. Higher values '
                'suggest a longer track record of profitability.'),
    'profitab': ('Profitability (EBIT / Total Assets)',
                 'Measures operational efficiency. Higher values indicate '
                 'stronger earnings generation from assets.'),
    'leverage': ('Leverage (Book Equity / Total Liabilities)',
                 'Indicates financial structure. Higher values suggest less '
                 'reliance on debt financing.'),
}

KAM_DESCRIPTIONS = {
    'GCKAM': 'Auditor Key Audit Matter: going concern (1=present)',
    'REVKAM': 'KAM: revenue recognition (1=present)',
    'ASSETKAM': 'KAM: impairment / assets (1=present)',
    'LIABKAM': 'KAM: liabilities (1=present)',
    'OTHERKAM': 'KAM: other audit topics (1=present)',
}

SENTIMENT_DESCRIPTIONS = {
    'sentiment_mean': 'FinBERT mean article score (≈ P(pos)−P(neg))',
    'sentiment_std': 'Dispersion of FinBERT scores across articles',
    'sentiment_pos_pct': 'Share of articles scored positive by FinBERT',
    'sentiment_neg_pct': 'Share of articles scored negative by FinBERT',
    'news_count': 'Number of news articles in the fiscal year',
}


def _bucket_for_category(category: str) -> str:
    if category in ('AA', 'A', 'BBB'):
        return 'Investment-grade (AA / A / BBB buckets)'
    return 'Speculative (BB bucket)'


def check_ollama():
    """Check if Ollama is running and return available model name."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m['name'].split(':')[0] for m in resp.json().get('models', [])]
            for preferred in OLLAMA_MODELS:
                if preferred in models:
                    return preferred
            if models:
                return models[0]
    except (requests.ConnectionError, requests.Timeout):
        pass
    return None


def check_finetuned():
    """Check if the LoRA adapter exists and can be loaded."""
    adapter_config = ADAPTER_DIR / "adapter_config.json"
    return adapter_config.exists()


def load_finetuned_model():
    """Load the fine-tuned Qwen model with LoRA adapter (lazy singleton)."""
    global _finetuned_model, _finetuned_tokenizer
    if _finetuned_model is not None:
        return _finetuned_model, _finetuned_tokenizer

    try:
        import os

        _hf = str(PROJECT_ROOT / ".hf_cache")
        os.environ.setdefault("HF_HOME", _hf)
        os.environ.setdefault("TRANSFORMERS_CACHE", _hf)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
    except ImportError:
        print("transformers/peft not installed; fine-tuned mode unavailable")
        return None, None

    if not check_finetuned():
        print(f"LoRA adapter not found at {ADAPTER_DIR}")
        return None, None

    adapter_cfg_path = ADAPTER_DIR / "adapter_config.json"
    try:
        with open(adapter_cfg_path, encoding="utf-8") as f:
            adapter_base = json.load(f).get("base_model_name_or_path")
        if adapter_base and adapter_base != QWEN_MODEL_NAME:
            print(
                f"LoRA adapter was trained on {adapter_base} but this code expects "
                f"{QWEN_MODEL_NAME}. Re-run: python models/finetune_qwen.py"
            )
            return None, None
    except (OSError, json.JSONDecodeError):
        pass

    print(f"Loading fine-tuned model from {ADAPTER_DIR}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    _finetuned_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    _finetuned_model.eval()

    _finetuned_tokenizer = AutoTokenizer.from_pretrained(
        QWEN_MODEL_NAME,
        trust_remote_code=True,
    )
    if _finetuned_tokenizer.pad_token is None:
        _finetuned_tokenizer.pad_token = _finetuned_tokenizer.eos_token

    print("Fine-tuned model ready")
    return _finetuned_model, _finetuned_tokenizer


def generate_finetuned_verdict(company_row, predicted_rating, confidence, actual_rating=None):
    """Generate a verdict using the fine-tuned Qwen model."""
    import torch

    model, tokenizer = load_finetuned_model()
    if model is None:
        return None

    name = company_row.get('company_name', 'Unknown')
    ticker = company_row.get('ticker', 'N/A')
    sector = company_row.get('sector', 'N/A')
    agency = company_row.get('rating_agency', 'N/A')
    year = int(company_row.get('fiscal_year', 0))
    actual = actual_rating or company_row.get('rating', 'N/A')
    risk_class = _bucket_for_category(predicted_rating)

    kam_bits = ", ".join(
        f"{k}={int(company_row.get(k, 0) or 0)}"
        for k in ("GCKAM", "REVKAM", "ASSETKAM", "LIABKAM", "OTHERKAM")
    )
    sent_bits = (
        f"SENT_MEAN={float(company_row.get('sentiment_mean', 0) or 0):.4f}, "
        f"SENT_STD={float(company_row.get('sentiment_std', 0) or 0):.4f}, "
        f"POS_PCT={float(company_row.get('sentiment_pos_pct', 0) or 0):.2f}, "
        f"NEG_PCT={float(company_row.get('sentiment_neg_pct', 0) or 0):.2f}, "
        f"NEWS_N={int(company_row.get('news_count', 0) or 0)}"
    )

    user_input = (
        f"Company: {name} | Ticker: {ticker} | Sector: {sector}\n"
        f"Fiscal Year: {year} | Agency: {agency} | Actual Rating: {actual}\n"
        f"ML Predicted category: {predicted_rating} ({risk_class}) | Confidence: {confidence:.0%}\n"
        f"Ratios: LIQUID={company_row.get('liquid', 0):.4f}, "
        f"CUMPROF={company_row.get('cumprof', 0):.4f}, "
        f"PROFITAB={company_row.get('profitab', 0):.4f}, "
        f"LEVERAGE={company_row.get('leverage', 0):.4f}\n"
        f"KAM dummies (0/1): {kam_bits}\n"
        f"News / FinBERT aggregates: {sent_bits}"
    )

    messages = [
        {"role": "system", "content": FINETUNED_SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True,
    )
    return parse_llm_response(response)


def build_prompt(company_row, predicted_rating, confidence, actual_rating=None):
    """Build the LLM prompt from a company's data row."""
    ticker = company_row.get('ticker', 'N/A')
    name = company_row.get('company_name', 'N/A')
    agency = company_row.get('rating_agency', 'N/A')
    year = company_row.get('fiscal_year', 'N/A')

    ratio_lines = []
    for ratio_key, (label, _) in RATIO_DESCRIPTIONS.items():
        val = company_row.get(ratio_key)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            ratio_lines.append(f"  - {label}: {val:.4f}")

    ratios_text = "\n".join(ratio_lines) if ratio_lines else "  No financial ratios available."

    kam_lines = [
        f"  - {KAM_DESCRIPTIONS[k]}: {int(company_row.get(k, 0) or 0)}"
        for k in KAM_DESCRIPTIONS
    ]
    kam_text = "\n".join(kam_lines)

    sent_lines = []
    for k, lab in SENTIMENT_DESCRIPTIONS.items():
        v = company_row.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if k == "news_count":
            sent_lines.append(f"  - {lab}: {int(v)}")
        else:
            sent_lines.append(f"  - {lab}: {float(v):.4f}")
    sent_text = "\n".join(sent_lines) if sent_lines else "  (No sentiment aggregates in row)"

    risk_class = _bucket_for_category(predicted_rating)
    actual_line = f"Actual Rating ({agency}): {actual_rating}" if actual_rating else "Actual Rating: Not available"

    prompt = f"""You are a credit analyst writing a structured credit verdict for a Saudi-listed company.

COMPANY INFORMATION:
  Company: {name}
  Ticker: {ticker}
  Rating Agency: {agency}
  Fiscal Year: {year}

FINANCIAL RATIOS (Altman-style components):
{ratios_text}

KEY AUDIT MATTER (KAM) DUMMIES (1 = topic disclosed as KAM, 0 = absent):
{kam_text}

NEWS SENTIMENT (FinBERT aggregates for this fiscal year):
{sent_text}

ML MODEL PREDICTION (multicategory XGBoost: AA / A / BBB / BB):
  Predicted category: {predicted_rating}
  Risk bucket: {risk_class}
  Model confidence: {confidence:.1%}
  {actual_line}

TASK:
Write a credit verdict in the following JSON format. Cite specific numbers from ratios, KAM flags,
and sentiment fields where relevant. Do NOT invent information.

{{
  "company": "{name}",
  "ticker": "{ticker}",
  "fiscal_year": {year},
  "predicted_rating": "{predicted_rating}",
  "risk_classification": "{risk_class}",
  "overall_assessment": "1-2 sentence summary using ratios + KAMs + news context",
  "strengths": ["2-3 items citing provided numbers"],
  "weaknesses": ["1-3 items citing provided numbers"],
  "key_risks": ["1-2 forward-looking risks"],
  "prediction_analysis": "Does the evidence support the predicted category? Mention KAMs/news if material."
}}

Return ONLY the JSON object, no other text."""

    return prompt


def query_ollama(prompt, model_name):
    """Send prompt to Ollama and return the response text."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 800,
        }
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json().get("response", "")


def generate_template_verdict(company_row, predicted_rating, confidence, actual_rating=None):
    """Template-based fallback when Ollama is unavailable."""
    name = company_row.get('company_name', 'Unknown')
    ticker = company_row.get('ticker', 'N/A')
    agency = company_row.get('rating_agency', 'N/A')
    year = int(company_row.get('fiscal_year', 0))
    row_cat = company_row.get("rating_category")

    liquid = company_row.get('liquid', 0)
    cumprof = company_row.get('cumprof', 0)
    profitab = company_row.get('profitab', 0)
    leverage = company_row.get('leverage', 0)

    risk_class = _bucket_for_category(predicted_rating)

    strengths = []
    weaknesses = []
    key_risks = []

    if liquid > 0.15:
        strengths.append(f"Strong liquidity position (LIQUID={liquid:.4f}), indicating adequate "
                         "working capital relative to total assets")
    elif liquid > 0.05:
        strengths.append(f"Adequate liquidity (LIQUID={liquid:.4f}), suggesting the company can "
                         "meet short-term obligations")
    elif liquid > -0.05:
        weaknesses.append(f"Low liquidity (LIQUID={liquid:.4f}), indicating limited working capital "
                          "buffer relative to total assets")
    else:
        weaknesses.append(f"Negative liquidity (LIQUID={liquid:.4f}), signaling potential "
                          "difficulty meeting short-term obligations")

    if cumprof > 0.20:
        strengths.append(f"Substantial accumulated profitability (CUMPROF={cumprof:.4f}), "
                         "reflecting a strong track record of retained earnings")
    elif cumprof > 0.05:
        strengths.append(f"Positive cumulative profitability (CUMPROF={cumprof:.4f}), "
                         "showing consistent earnings retention")
    elif cumprof > 0:
        weaknesses.append(f"Modest cumulative profitability (CUMPROF={cumprof:.4f}), "
                          "suggesting limited earnings retention over time")
    else:
        weaknesses.append(f"Negative cumulative profitability (CUMPROF={cumprof:.4f}), "
                          "indicating accumulated losses exceed retained earnings")

    if profitab > 0.08:
        strengths.append(f"Strong operational profitability (PROFITAB={profitab:.4f}), "
                         "demonstrating efficient asset utilization for earnings")
    elif profitab > 0.03:
        pass  # neutral
    elif profitab > 0:
        weaknesses.append(f"Low operational profitability (PROFITAB={profitab:.4f}), "
                          "suggesting limited earnings generation from assets")
    else:
        weaknesses.append(f"Negative profitability (PROFITAB={profitab:.4f}), "
                          "indicating the company is not generating operating income from its assets")

    if leverage > 1.5:
        strengths.append(f"Conservative financial structure (LEVERAGE={leverage:.4f}), "
                         "with equity substantially exceeding liabilities")
    elif leverage > 0.8:
        pass  # neutral
    elif leverage > 0.3:
        weaknesses.append(f"Elevated debt reliance (LEVERAGE={leverage:.4f}), "
                          "with liabilities forming a significant portion of financing")
    else:
        weaknesses.append(f"High leverage risk (LEVERAGE={leverage:.4f}), "
                          "indicating heavy dependence on debt financing")

    if not strengths:
        strengths.append("No strong positive financial indicators identified from the ratios")
    if not weaknesses:
        weaknesses.append("No major financial weaknesses identified from the ratios")

    if liquid < 0:
        key_risks.append("Negative working capital may constrain operational flexibility "
                         "and increase refinancing risk")
    if leverage < 0.5:
        key_risks.append("High leverage creates vulnerability to interest rate increases "
                         "and credit tightening in the Saudi market")
    if profitab < 0.02:
        key_risks.append("Low profitability may limit the company's ability to service debt "
                         "and fund growth without additional borrowing")
    if not key_risks:
        key_risks.append("Monitor macroeconomic conditions in Saudi Arabia and sector-specific "
                         "regulatory changes that could affect creditworthiness")

    nc = int(company_row.get("news_count", 0) or 0)
    sm = float(company_row.get("sentiment_mean", 0) or 0)
    if nc == 0:
        key_risks.append("No news articles in cache for this year — sentiment features are neutral "
                         "and may understate headline risk")
    elif sm < -0.2:
        weaknesses.append(f"Negative FinBERT news tone (sentiment_mean={sm:.3f} across {nc} articles)")

    if int(company_row.get("REVKAM", 0) or 0) == 1:
        key_risks.append("Revenue-recognition KAM flagged — review accounting judgments and disclosures")
    if int(company_row.get("ASSETKAM", 0) or 0) == 1:
        strengths.append("Asset / impairment KAM present — auditor focused on balance-sheet carrying values")

    avg_score = (liquid + cumprof + profitab) / 3
    ig_bucket = predicted_rating in ("AA", "A", "BBB")
    if ig_bucket and avg_score > 0.05:
        prediction_analysis = (
            f"Ratios align with an investment-grade-style bucket ({predicted_rating}): "
            f"LIQUID={liquid:.4f}, CUMPROF={cumprof:.4f}, PROFITAB={profitab:.4f}, "
            f"LEVERAGE={leverage:.4f}. KAM/news context should be read alongside these figures."
        )
    elif ig_bucket and avg_score <= 0.05:
        prediction_analysis = (
            f"The model predicts {predicted_rating} but average ratio strength is modest "
            f"(avg={avg_score:.4f}), so outcomes are sensitive to KAM and news signals."
        )
    elif not ig_bucket and avg_score < 0.05:
        prediction_analysis = (
            f"Weak liquidity (LIQUID={liquid:.4f}), profitability (PROFITAB={profitab:.4f}), "
            f"and cumulative earnings (CUMPROF={cumprof:.4f}) support a {predicted_rating} / "
            f"speculative-bucket view."
        )
    else:
        prediction_analysis = (
            f"Mixed signals: model predicts {predicted_rating} with avg ratio score {avg_score:.4f}; "
            f"review KAM flags and FinBERT aggregates for confirmation."
        )

    if actual_rating and row_cat and str(predicted_rating) != str(row_cat):
        prediction_analysis += (
            f" Note: actual agency rating is {actual_rating} (category {row_cat}) vs predicted "
            f"category {predicted_rating}."
        )

    overall = (
        f"{name} ({ticker}) shows a {'solid' if avg_score > 0.08 else 'moderate' if avg_score > 0.03 else 'weak'} "
        f"financial profile for FY{year}, with the ML model predicting a {predicted_rating} "
        f"({risk_class}) rating at {confidence:.0%} confidence."
    )

    return {
        "company": name,
        "ticker": ticker,
        "fiscal_year": year,
        "predicted_rating": predicted_rating,
        "risk_classification": risk_class,
        "overall_assessment": overall,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "key_risks": key_risks,
        "prediction_analysis": prediction_analysis,
    }


def parse_llm_response(response_text):
    """Extract JSON from LLM response, handling markdown fences."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


def evaluate_verdict(verdict, company_row):
    """
    Evaluate factual accuracy of a verdict against actual data.
    Returns a score dict with checks for each criterion.
    """
    checks = {}

    checks['has_company_name'] = verdict.get('company', '') != ''
    checks['has_overall_assessment'] = len(verdict.get('overall_assessment', '')) > 20
    checks['has_strengths'] = len(verdict.get('strengths', [])) > 0
    checks['has_weaknesses'] = len(verdict.get('weaknesses', [])) > 0
    checks['has_prediction_analysis'] = len(verdict.get('prediction_analysis', '')) > 20

    all_text = json.dumps(verdict).lower()
    ratio_keys = ['liquid', 'cumprof', 'profitab', 'leverage']
    cited = sum(1 for k in ratio_keys if k.lower() in all_text)
    checks['ratios_cited'] = cited
    checks['citation_rate'] = cited / len(ratio_keys)

    import re
    numbers_in_verdict = re.findall(r'[\-]?\d+\.\d{2,}', all_text)
    actual_values = [f"{company_row.get(k, 0):.4f}" for k in ratio_keys]
    correct_numbers = sum(1 for n in numbers_in_verdict if any(n in av for av in actual_values))
    checks['numbers_accuracy'] = correct_numbers / max(len(numbers_in_verdict), 1)

    checks['mentions_risk'] = any(w in all_text for w in ['risk', 'concern', 'vulnerability', 'weakness'])
    checks['mentions_strength'] = any(w in all_text for w in ['strong', 'solid', 'adequate', 'positive', 'healthy'])

    score_components = [
        checks['has_overall_assessment'],
        checks['has_strengths'],
        checks['has_weaknesses'],
        checks['has_prediction_analysis'],
        checks['citation_rate'] >= 0.5,
        checks['numbers_accuracy'] >= 0.5,
        checks['mentions_risk'],
        checks['mentions_strength'],
    ]
    checks['overall_score'] = sum(score_components) / len(score_components)

    return checks


def generate_verdicts(data_path=None, mode="auto", limit=None):
    """
    Generate verdicts for all companies in the dataset.

    mode: "auto" (try fine-tuned -> ollama -> template),
          "finetuned", "ollama", or "template"
    Returns list of (verdict, evaluation) tuples.
    """
    if data_path is None:
        from models.multisource_data import load_or_build_merged_training

        df = load_or_build_merged_training(save=True)
    else:
        df = pd.read_csv(data_path)
    df = df.dropna(subset=["liquid", "cumprof", "profitab", "leverage"], how="any").reset_index(
        drop=True
    )
    if "rating_category" not in df.columns:
        from models.multisource_data import prepare_target

        df = prepare_target(df)

    if mode == "auto":
        if check_finetuned():
            active_mode = "finetuned"
        else:
            ollama_model = check_ollama()
            active_mode = "ollama" if ollama_model else "template"
    else:
        active_mode = mode

    ollama_model = None
    if active_mode == "ollama":
        ollama_model = check_ollama()
        if not ollama_model:
            print("Ollama not available, falling back to template")
            active_mode = "template"

    method_labels = {
        "finetuned": "Fine-tuned Qwen 2.5 3B (QLoRA)",
        "ollama": f"Ollama ({ollama_model})" if ollama_model else "Ollama",
        "template": "Template-based (deterministic)",
    }
    print(f"Verdict generation method: {method_labels.get(active_mode, active_mode)}")

    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier

    X = df[FULL_FEATURE_COLS].values.astype(np.float64)
    le_ml = LabelEncoder()
    y = le_ml.fit_transform(df["rating_category"].values)
    counts = np.bincount(y)
    n_splits = int(min(5, counts.min()))
    if n_splits < 2:
        n_splits = 2
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    y_pred = cross_val_predict(xgb, X, y, cv=cv)
    y_proba = cross_val_predict(xgb, X, y, cv=cv, method="predict_proba")

    pred_ratings = []
    for i in range(len(df)):
        pred_ratings.append(
            (
                le_ml.inverse_transform([int(y_pred[i])])[0],
                float(np.max(y_proba[i])),
            )
        )

    results = []
    n_rows = min(limit, len(df)) if limit else len(df)

    for i in range(n_rows):
        row = df.iloc[i]
        pred_rating, confidence = pred_ratings[i]
        actual_rating = row["rating"]
        row_dict = row.to_dict()
        verdict = None

        if active_mode == "finetuned":
            try:
                verdict = generate_finetuned_verdict(row_dict, pred_rating, confidence, actual_rating)
                if verdict:
                    verdict["generation_method"] = "finetuned_qwen"
            except Exception as e:
                print(f"  Fine-tuned failed for {row['ticker']}: {e}")

        if verdict is None and active_mode == "ollama" and ollama_model:
            prompt = build_prompt(row_dict, pred_rating, confidence, actual_rating)
            try:
                response = query_ollama(prompt, ollama_model)
                verdict = parse_llm_response(response)
                if verdict:
                    verdict["generation_method"] = f"ollama_{ollama_model}"
            except Exception as e:
                print(f"  Ollama failed for {row['ticker']}: {e}")

        if verdict is None:
            verdict = generate_template_verdict(row_dict, pred_rating, confidence, actual_rating)
            fallback_label = "template_fallback" if active_mode != "template" else "template"
            verdict["generation_method"] = fallback_label

        evaluation = evaluate_verdict(verdict, row_dict)
        results.append(
            {
                "verdict": verdict,
                "evaluation": evaluation,
                "actual_rating": actual_rating,
                "actual_category": row["rating_category"],
                "predicted_category": pred_rating,
                "category_match": pred_rating == row["rating_category"],
            }
        )

        co = str(row.get("company_name", "") or "")
        company_label = f"{row['ticker']} ({co[:25]})"
        print(
            f"  [{i + 1}/{n_rows}] {company_label:<40} "
            f"pred={pred_rating} actual_cat={row['rating_category']} "
            f"quality={evaluation['overall_score']:.0%}"
        )

    return results


def save_verdicts(results, output_dir=None):
    """Save all verdicts and evaluations to files."""
    if output_dir is None:
        output_dir = VERDICTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_verdicts = []
    for r in results:
        entry = {**r['verdict'], '_evaluation': r['evaluation'],
                 '_actual_rating': r['actual_rating']}
        all_verdicts.append(entry)

    verdicts_file = output_dir / "all_verdicts.json"
    with open(verdicts_file, 'w') as f:
        json.dump(all_verdicts, f, indent=2, default=str)

    scores = [r['evaluation']['overall_score'] for r in results]
    citation_rates = [r['evaluation']['citation_rate'] for r in results]
    number_acc = [r['evaluation']['numbers_accuracy'] for r in results]

    summary = {
        'generation_timestamp': datetime.now().isoformat(),
        'total_verdicts': len(results),
        'generation_method': results[0]['verdict'].get('generation_method', 'unknown') if results else 'none',
        'quality_metrics': {
            'avg_overall_score': float(np.mean(scores)),
            'avg_citation_rate': float(np.mean(citation_rates)),
            'avg_number_accuracy': float(np.mean(number_acc)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
        },
        'ml_performance': {
            'category_correct': sum(1 for r in results if r.get('category_match')),
            'total': len(results),
            'category_accuracy': sum(1 for r in results if r.get('category_match'))
            / max(len(results), 1),
        }
    }

    summary_file = output_dir / "verdict_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nVerdict Quality Summary:")
    print(f"  Total verdicts:      {summary['total_verdicts']}")
    print(f"  Avg quality score:   {summary['quality_metrics']['avg_overall_score']:.1%}")
    print(f"  Avg citation rate:   {summary['quality_metrics']['avg_citation_rate']:.1%}")
    print(f"  Avg number accuracy: {summary['quality_metrics']['avg_number_accuracy']:.1%}")
    print(
        f"  ML category accuracy: {summary['ml_performance']['category_accuracy']:.1%}"
    )
    print(f"\nSaved to {output_dir}")

    return summary


def main():
    print("=" * 70)
    print("LLM CREDIT VERDICT GENERATOR")
    print("=" * 70)

    mode = "auto"
    if len(sys.argv) > 1 and sys.argv[1] in ("finetuned", "ollama", "template", "auto"):
        mode = sys.argv[1]
    print(f"Requested mode: {mode}\n")

    results = generate_verdicts(mode=mode)
    summary = save_verdicts(results)

    print("\n" + "=" * 70)
    print("SAMPLE VERDICT")
    print("=" * 70)
    if results:
        sample = results[0]['verdict']
        print(json.dumps(sample, indent=2, default=str))


if __name__ == "__main__":
    main()
