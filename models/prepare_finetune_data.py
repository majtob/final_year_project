"""
Prepare instruction-format dataset for QLoRA fine-tuning of Qwen 2.5 3B.

Reads the 75 training records and their template verdicts,
then formats them into instruction/input/output JSONL for SFTTrainer.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "merged_multisource_training.csv"
VERDICTS_FILE = PROJECT_ROOT / "results" / "verdicts" / "all_verdicts.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "finetune_dataset.jsonl"

RATING_TO_NUMERIC = {
    'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18,
    'A+': 17, 'A': 16, 'A-': 15,
    'BBB+': 14, 'BBB': 13, 'BBB-': 12,
    'BB+': 11, 'BB': 10, 'BB-': 9,
    'B+': 8, 'B': 7, 'B-': 6,
    'CCC+': 5, 'CCC': 4, 'CCC-': 3,
    'CC': 2, 'C': 1, 'D': 0,
}

SYSTEM_PROMPT = (
    "You are a credit analyst specializing in Saudi Exchange (Tadawul) listed companies. "
    "Given a company's financial ratios and ML model prediction, generate a structured "
    "credit verdict as a JSON object. Every claim must cite specific ratio values. "
    "Do not invent information beyond what is provided."
)


def get_ml_predictions(df):
    """Run the same Gradient Boosting model to get predictions and confidence."""
    features = ['liquid', 'cumprof', 'profitab', 'leverage']
    df['target'] = df['rating'].apply(
        lambda r: 0 if RATING_TO_NUMERIC.get(r, 0) >= 15 else 1
    )
    X = df[features].values
    y = df['target'].values
    groups = df['ticker'].values

    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    n_splits = min(5, len(set(groups)))
    cv = GroupKFold(n_splits=n_splits)

    y_pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
    y_proba = cross_val_predict(model, X, y, cv=cv, groups=groups, method='predict_proba')

    predictions = []
    for p, prob in zip(y_pred, y_proba):
        confidence = float(max(prob))
        pred_rating = "A" if p == 0 else "BBB"
        predictions.append((pred_rating, confidence))
    return predictions


def format_input(row, pred_rating, confidence):
    """Format a single record into the user input string."""
    name = row.get('company_name', 'Unknown')
    ticker = row.get('ticker', 'N/A')
    sector = row.get('sector') or row.get('yfinance_sector', 'N/A')
    agency = row.get('rating_agency', 'N/A')
    year = int(row.get('fiscal_year', 0))
    actual = row.get('rating', 'N/A')
    risk_class = "Investment Grade" if RATING_TO_NUMERIC.get(pred_rating, 0) >= 12 else "Speculative Grade"

    return (
        f"Company: {name} | Ticker: {ticker} | Sector: {sector}\n"
        f"Fiscal Year: {year} | Agency: {agency} | Actual Rating: {actual}\n"
        f"ML Predicted: {pred_rating} ({risk_class}) | Confidence: {confidence:.0%}\n"
        f"Ratios: LIQUID={float(row['liquid']):.4f}, CUMPROF={float(row['cumprof']):.4f}, "
        f"PROFITAB={float(row['profitab']):.4f}, LEVERAGE={float(row['leverage']):.4f}"
    )


def format_output(verdict):
    """Format the verdict dict into the expected JSON output string."""
    output = {
        "company": verdict["company"],
        "ticker": verdict["ticker"],
        "fiscal_year": verdict["fiscal_year"],
        "predicted_rating": verdict["predicted_rating"],
        "risk_classification": verdict["risk_classification"],
        "overall_assessment": verdict["overall_assessment"],
        "strengths": verdict["strengths"],
        "weaknesses": verdict["weaknesses"],
        "key_risks": verdict["key_risks"],
        "prediction_analysis": verdict["prediction_analysis"],
    }
    return json.dumps(output, indent=2)


def main():
    print("=" * 60)
    print("PREPARING FINE-TUNING DATASET")
    print("=" * 60)

    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["liquid", "cumprof", "profitab", "leverage"], how="any")
    print(f"Loaded {len(df)} records from {DATA_FILE.name}")

    with open(VERDICTS_FILE) as f:
        verdicts = json.load(f)
    print(f"Loaded {len(verdicts)} template verdicts")

    predictions = get_ml_predictions(df)
    print(f"Generated {len(predictions)} ML predictions")

    n = min(len(df), len(verdicts))
    examples = []

    for i in range(n):
        row = df.iloc[i]
        verdict = verdicts[i]
        pred_rating, confidence = predictions[i]

        user_input = format_input(row.to_dict(), pred_rating, confidence)
        output = format_output(verdict)

        example = {
            "instruction": SYSTEM_PROMPT,
            "input": user_input,
            "output": output,
        }
        examples.append(example)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nSaved {len(examples)} examples to {OUTPUT_FILE}")
    print(f"Sample input:\n{examples[0]['input']}\n")
    print(f"Output length: {len(examples[0]['output'])} chars")


if __name__ == "__main__":
    main()
