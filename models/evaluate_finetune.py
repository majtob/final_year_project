"""
Evaluate fine-tuned Qwen 2.5 3B LoRA adapter against template verdicts.

Loads the base model + adapter, generates verdicts for all 75 companies,
and compares quality metrics against the template baseline.
"""

import json
import re
import os
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime

os.environ["HF_HOME"] = "D:\\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:\\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_FILE = PROJECT_ROOT / "data" / "processed" / "finetune_dataset.jsonl"
ADAPTER_DIR = PROJECT_ROOT / "models" / "lora_adapter"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LENGTH = 1024

SYSTEM_PROMPT = (
    "You are a credit analyst specializing in Saudi Exchange (Tadawul) listed companies. "
    "Given a company's financial ratios and ML model prediction, generate a structured "
    "credit verdict as a JSON object. Every claim must cite specific ratio values. "
    "Do not invent information beyond what is provided."
)


def load_model_and_tokenizer():
    """Load base Qwen model with LoRA adapter."""
    cfg_path = ADAPTER_DIR / "adapter_config.json"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            adapter_base = json.load(f).get("base_model_name_or_path")
        if adapter_base and adapter_base != MODEL_NAME:
            raise SystemExit(
                f"Adapter was trained on {adapter_base} but evaluate_finetune.py expects "
                f"{MODEL_NAME}. Re-run training (models/finetune_qwen.py) or temporarily "
                f"set MODEL_NAME to match adapter_config.json."
            )

    print(f"Loading {MODEL_NAME} + LoRA adapter...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded for inference")
    return model, tokenizer


def generate_verdict(model, tokenizer, user_input):
    """Generate a verdict from the fine-tuned model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
    return response


def parse_json_response(text):
    """Extract JSON from model response."""
    text = text.strip()
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


def evaluate_single_verdict(verdict, user_input=""):
    """Evaluate a single verdict for quality metrics."""
    metrics = {
        "json_valid": verdict is not None,
        "has_company": False,
        "has_overall": False,
        "has_strengths": False,
        "has_weaknesses": False,
        "has_analysis": False,
        "ratios_cited": 0,
        "citation_rate": 0.0,
        "numbers_found": 0,
        "overall_score": 0.0,
    }

    if verdict is None:
        return metrics

    metrics["has_company"] = bool(verdict.get("company", ""))
    metrics["has_overall"] = len(verdict.get("overall_assessment", "")) > 20
    metrics["has_strengths"] = len(verdict.get("strengths", [])) > 0
    metrics["has_weaknesses"] = len(verdict.get("weaknesses", [])) > 0
    metrics["has_analysis"] = len(verdict.get("prediction_analysis", "")) > 20

    all_text = json.dumps(verdict).lower()
    ratio_keys = ['liquid', 'cumprof', 'profitab', 'leverage']
    cited = sum(1 for k in ratio_keys if k.lower() in all_text)
    metrics["ratios_cited"] = cited
    metrics["citation_rate"] = cited / len(ratio_keys)

    numbers = re.findall(r'[\-]?\d+\.\d{2,}', all_text)
    metrics["numbers_found"] = len(numbers)

    score_parts = [
        metrics["json_valid"],
        metrics["has_overall"],
        metrics["has_strengths"],
        metrics["has_weaknesses"],
        metrics["has_analysis"],
        metrics["citation_rate"] >= 0.5,
        metrics["numbers_found"] >= 2,
    ]
    metrics["overall_score"] = sum(score_parts) / len(score_parts)

    return metrics


def measure_diversity(verdicts):
    """Measure linguistic diversity across verdicts via trigram variety."""
    all_phrases = []
    for v in verdicts:
        if v is None:
            continue
        assessment = v.get("overall_assessment", "")
        words = assessment.lower().split()
        for i in range(len(words) - 2):
            trigram = " ".join(words[i:i+3])
            all_phrases.append(trigram)

    if not all_phrases:
        return {"unique_trigrams": 0, "total_trigrams": 0, "diversity_ratio": 0.0}

    unique = len(set(all_phrases))
    total = len(all_phrases)
    return {
        "unique_trigrams": unique,
        "total_trigrams": total,
        "diversity_ratio": unique / max(total, 1),
    }


def load_template_verdicts():
    """Load the template verdicts from the dataset for comparison."""
    verdicts = []
    with open(DATASET_FILE) as f:
        for line in f:
            ex = json.loads(line)
            v = json.loads(ex["output"])
            verdicts.append(v)
    return verdicts


def main():
    print("=" * 60)
    print("FINE-TUNED MODEL EVALUATION")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer()

    examples = []
    with open(DATASET_FILE) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} test inputs")

    ft_verdicts = []
    ft_metrics = []

    for i, ex in enumerate(examples):
        print(f"  [{i+1}/{len(examples)}] Generating...", end=" ", flush=True)
        raw_response = generate_verdict(model, tokenizer, ex["input"])
        verdict = parse_json_response(raw_response)
        m = evaluate_single_verdict(verdict, ex["input"])

        ft_verdicts.append(verdict)
        ft_metrics.append(m)

        status = "OK" if m["json_valid"] else "FAIL"
        company = "?"
        if verdict and verdict.get("company"):
            c = verdict["company"]
            company = c if isinstance(c, str) else str(c)[:30]
        else:
            try:
                c = json.loads(ex["output"]).get("company", "?")
                company = c if isinstance(c, str) else str(c)[:30]
            except Exception:
                pass
        score_val = float(m['overall_score']) if m['overall_score'] is not None else 0.0
        print(f"{company:<30} {status} score={score_val:.0%}")

    template_verdicts = load_template_verdicts()
    template_metrics = [evaluate_single_verdict(v) for v in template_verdicts]

    ft_diversity = measure_diversity(ft_verdicts)
    template_diversity = measure_diversity(template_verdicts)

    def avg_metric(metrics_list, key):
        vals = [m[key] for m in metrics_list]
        return sum(vals) / max(len(vals), 1)

    results = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "adapter_path": str(ADAPTER_DIR),
        "num_examples": len(examples),
        "fine_tuned": {
            "json_success_rate": avg_metric(ft_metrics, "json_valid"),
            "avg_citation_rate": avg_metric(ft_metrics, "citation_rate"),
            "avg_overall_score": avg_metric(ft_metrics, "overall_score"),
            "avg_ratios_cited": avg_metric(ft_metrics, "ratios_cited"),
            "diversity": ft_diversity,
        },
        "template_baseline": {
            "json_success_rate": avg_metric(template_metrics, "json_valid"),
            "avg_citation_rate": avg_metric(template_metrics, "citation_rate"),
            "avg_overall_score": avg_metric(template_metrics, "overall_score"),
            "avg_ratios_cited": avg_metric(template_metrics, "ratios_cited"),
            "diversity": template_diversity,
        },
        "comparison": {
            "json_improvement": (
                avg_metric(ft_metrics, "json_valid") -
                avg_metric(template_metrics, "json_valid")
            ),
            "citation_improvement": (
                avg_metric(ft_metrics, "citation_rate") -
                avg_metric(template_metrics, "citation_rate")
            ),
            "score_improvement": (
                avg_metric(ft_metrics, "overall_score") -
                avg_metric(template_metrics, "overall_score")
            ),
            "diversity_improvement": (
                ft_diversity["diversity_ratio"] -
                template_diversity["diversity_ratio"]
            ),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / "finetune_evaluation.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    ft_verdicts_file = RESULTS_DIR / "verdicts" / "finetuned_verdicts.json"
    ft_verdicts_file.parent.mkdir(parents=True, exist_ok=True)
    with open(ft_verdicts_file, 'w') as f:
        json.dump([v for v in ft_verdicts if v is not None], f, indent=2)

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Metric':<30} {'Template':>12} {'Fine-tuned':>12} {'Delta':>10}")
    print("-" * 64)

    comparisons = [
        ("JSON Success Rate", "json_success_rate"),
        ("Avg Citation Rate", "avg_citation_rate"),
        ("Avg Overall Score", "avg_overall_score"),
        ("Avg Ratios Cited", "avg_ratios_cited"),
    ]

    for label, key in comparisons:
        t_val = results["template_baseline"][key]
        f_val = results["fine_tuned"][key]
        delta = f_val - t_val
        sign = "+" if delta > 0 else ""
        if isinstance(t_val, float) and t_val <= 1.0:
            print(f"{label:<30} {t_val:>11.1%} {f_val:>11.1%} {sign}{delta:>9.1%}")
        else:
            print(f"{label:<30} {t_val:>12.2f} {f_val:>12.2f} {sign}{delta:>10.2f}")

    td = template_diversity["diversity_ratio"]
    fd = ft_diversity["diversity_ratio"]
    print(f"{'Diversity Ratio':<30} {td:>11.1%} {fd:>11.1%} {'+' if fd>td else ''}{fd-td:>9.1%}")

    print(f"\nResults saved to {output_file}")
    print(f"Fine-tuned verdicts saved to {ft_verdicts_file}")


if __name__ == "__main__":
    main()
