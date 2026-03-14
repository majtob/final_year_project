"""
LLM Verdict Generator for Credit Rating Predictions.

Supports three generation modes:
  1. fine-tuned  -- Qwen 2.5 7B with QLoRA adapter (recommended)
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
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "model_training_data_v2.csv"
VERDICTS_DIR = PROJECT_ROOT / "results" / "verdicts"
ADAPTER_DIR = PROJECT_ROOT / "models" / "lora_adapter"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODELS = ["mistral", "llama3", "llama3.1"]

QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
QWEN_MAX_SEQ_LENGTH = 1024

FINETUNED_SYSTEM_PROMPT = (
    "You are a credit analyst specializing in Saudi Exchange (Tadawul) listed companies. "
    "Given a company's financial ratios and ML model prediction, generate a structured "
    "credit verdict as a JSON object. Every claim must cite specific ratio values. "
    "Do not invent information beyond what is provided."
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
        os.environ["HF_HOME"] = "D:\\hf_cache"
        os.environ["TRANSFORMERS_CACHE"] = "D:\\hf_cache"
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
    except ImportError:
        print("transformers/peft not installed; fine-tuned mode unavailable")
        return None, None

    if not check_finetuned():
        print(f"LoRA adapter not found at {ADAPTER_DIR}")
        return None, None

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
    num_rating = RATING_TO_NUMERIC.get(predicted_rating, 0)
    risk_class = "Investment Grade" if num_rating >= 12 else "Speculative Grade"

    user_input = (
        f"Company: {name} | Ticker: {ticker} | Sector: {sector}\n"
        f"Fiscal Year: {year} | Agency: {agency} | Actual Rating: {actual}\n"
        f"ML Predicted: {predicted_rating} ({risk_class}) | Confidence: {confidence:.0%}\n"
        f"Ratios: LIQUID={company_row.get('liquid', 0):.4f}, "
        f"CUMPROF={company_row.get('cumprof', 0):.4f}, "
        f"PROFITAB={company_row.get('profitab', 0):.4f}, "
        f"LEVERAGE={company_row.get('leverage', 0):.4f}"
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

    risk_class = "Investment Grade" if RATING_TO_NUMERIC.get(predicted_rating, 0) >= 12 else "Speculative Grade"
    actual_line = f"Actual Rating ({agency}): {actual_rating}" if actual_rating else "Actual Rating: Not available"

    prompt = f"""You are a credit analyst writing a structured credit verdict for a Saudi-listed company.

COMPANY INFORMATION:
  Company: {name}
  Ticker: {ticker}
  Rating Agency: {agency}
  Fiscal Year: {year}

FINANCIAL RATIOS (Altman Z''-Score Components):
{ratios_text}

ML MODEL PREDICTION:
  Predicted Rating: {predicted_rating}
  Risk Classification: {risk_class}
  Model Confidence: {confidence:.1%}
  {actual_line}

TASK:
Write a credit verdict in the following JSON format. Be specific -- cite the actual ratio values.
Every claim must be supported by the data provided above. Do NOT invent information.

{{
  "company": "{name}",
  "ticker": "{ticker}",
  "fiscal_year": {year},
  "predicted_rating": "{predicted_rating}",
  "risk_classification": "{risk_class}",
  "overall_assessment": "1-2 sentence summary of creditworthiness based on the ratios",
  "strengths": ["list 2-3 positive factors citing specific ratio values"],
  "weaknesses": ["list 1-3 risk factors citing specific ratio values"],
  "key_risks": ["list 1-2 forward-looking risk items to monitor"],
  "prediction_analysis": "Does the financial data support the predicted rating? Explain why."
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

    liquid = company_row.get('liquid', 0)
    cumprof = company_row.get('cumprof', 0)
    profitab = company_row.get('profitab', 0)
    leverage = company_row.get('leverage', 0)

    num_rating = RATING_TO_NUMERIC.get(predicted_rating, 0)
    risk_class = "Investment Grade" if num_rating >= 12 else "Speculative Grade"

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

    avg_score = (liquid + cumprof + profitab) / 3
    if risk_class == "Investment Grade" and avg_score > 0.05:
        prediction_analysis = (
            f"The financial data supports the {predicted_rating} prediction. "
            f"The company shows adequate liquidity (LIQUID={liquid:.4f}), "
            f"positive cumulative profitability (CUMPROF={cumprof:.4f}), and "
            f"operational returns (PROFITAB={profitab:.4f}), consistent with "
            f"an investment-grade classification."
        )
    elif risk_class == "Investment Grade" and avg_score <= 0.05:
        prediction_analysis = (
            f"The {predicted_rating} prediction is borderline. While classified as "
            f"investment grade, the weak financial ratios (avg score={avg_score:.4f}) "
            f"suggest the company is near the boundary between investment and speculative grade."
        )
    elif risk_class == "Speculative Grade" and avg_score < 0.05:
        prediction_analysis = (
            f"The financial data supports the {predicted_rating} speculative-grade prediction. "
            f"Weak liquidity (LIQUID={liquid:.4f}), limited profitability (PROFITAB={profitab:.4f}), "
            f"and modest earnings retention (CUMPROF={cumprof:.4f}) are consistent with "
            f"below-investment-grade creditworthiness."
        )
    else:
        prediction_analysis = (
            f"The {predicted_rating} prediction may be conservative. Some financial ratios "
            f"show moderate strength (avg score={avg_score:.4f}), though the model assigns "
            f"speculative grade based on the overall pattern of ratios."
        )

    if actual_rating and actual_rating != predicted_rating:
        actual_num = RATING_TO_NUMERIC.get(actual_rating, 0)
        diff = actual_num - num_rating
        direction = "higher" if diff > 0 else "lower"
        prediction_analysis += (
            f" Note: the actual {agency} rating is {actual_rating}, which is {abs(diff)} notch(es) "
            f"{direction} than predicted, suggesting the model "
            f"{'underestimates' if diff > 0 else 'overestimates'} this company's creditworthiness."
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
        data_path = DATA_FILE
    df = pd.read_csv(data_path)

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
        "finetuned": "Fine-tuned Qwen 2.5 7B (QLoRA)",
        "ollama": f"Ollama ({ollama_model})" if ollama_model else "Ollama",
        "template": "Template-based (deterministic)",
    }
    print(f"Verdict generation method: {method_labels.get(active_mode, active_mode)}")

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict

    features = ['liquid', 'cumprof', 'profitab', 'leverage']
    df['target'] = df['rating'].apply(lambda r: 0 if RATING_TO_NUMERIC.get(r, 0) >= 15 else 1)

    X = df[features].values
    y = df['target'].values
    groups = df['ticker'].values

    ml_model = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                          learning_rate=0.1, random_state=42)
    n_splits = min(5, len(set(groups)))
    cv = GroupKFold(n_splits=n_splits)

    y_pred = cross_val_predict(ml_model, X, y, cv=cv, groups=groups)
    y_proba = cross_val_predict(ml_model, X, y, cv=cv, groups=groups, method='predict_proba')

    pred_ratings = []
    for p, prob in zip(y_pred, y_proba):
        confidence = float(max(prob))
        pred_rating = "A" if p == 0 else "BBB"
        pred_ratings.append((pred_rating, confidence))

    results = []
    rows_to_process = df.head(limit) if limit else df

    for idx, row in rows_to_process.iterrows():
        pred_rating, confidence = pred_ratings[idx]
        actual_rating = row['rating']
        row_dict = row.to_dict()
        verdict = None

        if active_mode == "finetuned":
            try:
                verdict = generate_finetuned_verdict(row_dict, pred_rating, confidence, actual_rating)
                if verdict:
                    verdict['generation_method'] = 'finetuned_qwen'
            except Exception as e:
                print(f"  Fine-tuned failed for {row['ticker']}: {e}")

        if verdict is None and active_mode == "ollama" and ollama_model:
            prompt = build_prompt(row_dict, pred_rating, confidence, actual_rating)
            try:
                response = query_ollama(prompt, ollama_model)
                verdict = parse_llm_response(response)
                if verdict:
                    verdict['generation_method'] = f'ollama_{ollama_model}'
            except Exception as e:
                print(f"  Ollama failed for {row['ticker']}: {e}")

        if verdict is None:
            verdict = generate_template_verdict(row_dict, pred_rating, confidence, actual_rating)
            fallback_label = 'template_fallback' if active_mode != 'template' else 'template'
            verdict['generation_method'] = fallback_label

        evaluation = evaluate_verdict(verdict, row_dict)
        results.append({
            'verdict': verdict,
            'evaluation': evaluation,
            'actual_rating': actual_rating,
            'predicted_binary': int(y_pred[idx]),
            'actual_binary': int(y[idx]),
        })

        company_label = f"{row['ticker']} ({row['company_name'][:25]})"
        print(f"  [{idx+1}/{len(rows_to_process)}] {company_label:<40} "
              f"pred={pred_rating} actual={actual_rating} "
              f"quality={evaluation['overall_score']:.0%}")

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
            'correct_predictions': sum(1 for r in results if r['predicted_binary'] == r['actual_binary']),
            'total': len(results),
            'accuracy': sum(1 for r in results if r['predicted_binary'] == r['actual_binary']) / max(len(results), 1),
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
    print(f"  ML accuracy:         {summary['ml_performance']['accuracy']:.1%}")
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
