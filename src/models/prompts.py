#!/usr/bin/env python3
"""
Prompt Templates for LLM Verdict Generation
Contains prompts for generating credit risk verdicts using open-source LLMs.
"""

from typing import Dict, List, Optional
import json


# Main verdict generation prompt
VERDICT_SYSTEM_PROMPT = """You are an expert credit analyst specializing in Saudi Arabian companies. 
Your task is to provide a structured credit risk verdict based on the provided financial data, 
key audit matters, and news information.

Be objective and factual. Base your analysis only on the data provided.
Cite specific numbers and facts from the input data to support your conclusions."""


VERDICT_USER_PROMPT_TEMPLATE = """
Analyze the following company data and provide a credit risk verdict.

=== COMPANY INFORMATION ===
Company: {company_name}
Ticker: {ticker}
Fiscal Year: {fiscal_year}

=== FINANCIAL RATIOS ===
Liquidity:
  - Current Ratio: {current_ratio:.2f}
  - Quick Ratio: {quick_ratio:.2f}
  - Cash Ratio: {cash_ratio:.2f}

Leverage:
  - Debt to Equity: {debt_to_equity:.2f}
  - Debt to Assets: {debt_to_assets:.2f}

Coverage:
  - Interest Coverage: {interest_coverage:.2f}

Profitability:
  - Return on Assets: {return_on_assets:.2%}
  - Return on Equity: {return_on_equity:.2%}
  - Net Margin: {net_margin:.2%}
  - Operating Margin: {operating_margin:.2%}

Growth:
  - Revenue Growth: {revenue_growth:.2%}
  - Net Income Growth: {net_income_growth:.2%}

=== KEY AUDIT MATTERS ===
Number of KAMs: {kam_count}
{kam_details}

=== NEWS SUMMARY ===
Articles in Period: {news_count}
Average Sentiment: {avg_sentiment:.2f} (scale: -1 to 1)
Negative News Ratio: {negative_news_ratio:.1%}
{news_highlights}

=== ML MODEL PREDICTION ===
Predicted Rating: {predicted_rating}
Prediction Confidence: {prediction_confidence:.1%}
{actual_rating_line}

=== TASK ===
Provide your analysis as a JSON object with the following structure:
{{
  "overall_assessment": "1-2 sentence summary of credit profile",
  "risk_level": "LOW | MODERATE | ELEVATED | HIGH",
  "strengths": ["strength 1 with supporting data", "strength 2 with supporting data"],
  "weaknesses": ["weakness 1 with supporting data", "weakness 2 with supporting data"],
  "risk_factors": ["key risk to monitor 1", "key risk to monitor 2"],
  "prediction_analysis": "Does the data support the ML prediction? Why or why not?",
  "outlook": "POSITIVE | STABLE | NEGATIVE"
}}

Respond ONLY with the JSON object, no additional text.
"""


VERDICT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_assessment": {"type": "string"},
        "risk_level": {"type": "string", "enum": ["LOW", "MODERATE", "ELEVATED", "HIGH"]},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "prediction_analysis": {"type": "string"},
        "outlook": {"type": "string", "enum": ["POSITIVE", "STABLE", "NEGATIVE"]}
    },
    "required": ["overall_assessment", "risk_level", "strengths", "weaknesses", 
                 "risk_factors", "prediction_analysis", "outlook"]
}


def format_kam_details(kam_features: Dict) -> str:
    """Format KAM details for the prompt."""
    if kam_features.get("kam_count", 0) == 0:
        return "No Key Audit Matters reported."
    
    lines = []
    kam_types = [
        ("kam_going_concern", "Going Concern"),
        ("kam_impairment", "Asset Impairment"),
        ("kam_revenue_recognition", "Revenue Recognition"),
        ("kam_related_party", "Related Party Transactions"),
        ("kam_litigation", "Litigation/Contingencies"),
        ("kam_valuation", "Fair Value/Valuation")
    ]
    
    for key, label in kam_types:
        if kam_features.get(key, 0) > 0:
            lines.append(f"  - {label}: Yes")
    
    severity = kam_features.get("kam_severity_score", 0)
    lines.append(f"  - Severity Score: {severity:.2f} (scale: 0-1)")
    
    return "\n".join(lines) if lines else "No significant KAMs identified."


def format_news_highlights(news_features: Dict) -> str:
    """Format news highlights for the prompt."""
    if news_features.get("news_count", 0) == 0:
        return "No news coverage in the period."
    
    lines = []
    events = [
        ("event_lawsuit", "Legal/Lawsuit mentions"),
        ("event_expansion", "Expansion/Growth mentions"),
        ("event_regulatory", "Regulatory mentions"),
        ("event_earnings", "Earnings mentions")
    ]
    
    for key, label in events:
        count = news_features.get(key, 0)
        if count > 0:
            lines.append(f"  - {label}: {count}")
    
    return "\n".join(lines) if lines else "No significant news events."


def build_verdict_prompt(
    company_data: Dict,
    financial_ratios: Dict,
    kam_features: Dict,
    news_features: Dict,
    predicted_rating: str,
    prediction_confidence: float,
    actual_rating: Optional[str] = None
) -> str:
    """
    Build the complete verdict prompt.
    
    Args:
        company_data: Company identifiers (name, ticker, year)
        financial_ratios: Dictionary of financial ratios
        kam_features: Dictionary of KAM features
        news_features: Dictionary of news features
        predicted_rating: ML model's predicted rating
        prediction_confidence: Confidence score (0-1)
        actual_rating: Actual Tassnief rating (if known)
        
    Returns:
        Formatted prompt string
    """
    # Handle missing values with defaults
    def safe_get(d: Dict, key: str, default: float = 0.0) -> float:
        val = d.get(key, default)
        if val is None or (isinstance(val, float) and val != val):  # NaN check
            return default
        return val
    
    kam_details = format_kam_details(kam_features)
    news_highlights = format_news_highlights(news_features)
    
    actual_rating_line = ""
    if actual_rating:
        actual_rating_line = f"Actual Tassnief Rating: {actual_rating}"
    
    prompt = VERDICT_USER_PROMPT_TEMPLATE.format(
        company_name=company_data.get("company_name", "Unknown"),
        ticker=company_data.get("ticker", "Unknown"),
        fiscal_year=company_data.get("fiscal_year", "Unknown"),
        
        # Financial ratios
        current_ratio=safe_get(financial_ratios, "current_ratio", 1.0),
        quick_ratio=safe_get(financial_ratios, "quick_ratio", 1.0),
        cash_ratio=safe_get(financial_ratios, "cash_ratio", 0.5),
        debt_to_equity=safe_get(financial_ratios, "debt_to_equity", 1.0),
        debt_to_assets=safe_get(financial_ratios, "debt_to_assets", 0.5),
        interest_coverage=safe_get(financial_ratios, "interest_coverage", 3.0),
        return_on_assets=safe_get(financial_ratios, "return_on_assets", 0.05),
        return_on_equity=safe_get(financial_ratios, "return_on_equity", 0.10),
        net_margin=safe_get(financial_ratios, "net_margin", 0.05),
        operating_margin=safe_get(financial_ratios, "operating_margin", 0.10),
        revenue_growth=safe_get(financial_ratios, "revenue_growth", 0.0),
        net_income_growth=safe_get(financial_ratios, "net_income_growth", 0.0),
        
        # KAM
        kam_count=int(safe_get(kam_features, "kam_count", 0)),
        kam_details=kam_details,
        
        # News
        news_count=int(safe_get(news_features, "news_count", 0)),
        avg_sentiment=safe_get(news_features, "avg_sentiment", 0.0),
        negative_news_ratio=safe_get(news_features, "negative_news_ratio", 0.0),
        news_highlights=news_highlights,
        
        # Prediction
        predicted_rating=predicted_rating,
        prediction_confidence=prediction_confidence,
        actual_rating_line=actual_rating_line
    )
    
    return prompt


def parse_verdict_response(response: str) -> Dict:
    """
    Parse the LLM's JSON response.
    
    Args:
        response: Raw LLM response string
        
    Returns:
        Parsed verdict dictionary
    """
    # Try to extract JSON from response
    response = response.strip()
    
    # Handle markdown code blocks
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    
    response = response.strip()
    
    try:
        verdict = json.loads(response)
        return verdict
    except json.JSONDecodeError as e:
        # Try to find JSON in the response
        import re
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                verdict = json.loads(json_match.group())
                return verdict
            except json.JSONDecodeError:
                pass
        
        return {
            "overall_assessment": "Error parsing LLM response",
            "risk_level": "UNKNOWN",
            "strengths": [],
            "weaknesses": [],
            "risk_factors": [],
            "prediction_analysis": f"Parse error: {str(e)}",
            "outlook": "UNKNOWN",
            "raw_response": response
        }


def validate_verdict(verdict: Dict) -> bool:
    """
    Validate verdict against schema.
    
    Args:
        verdict: Parsed verdict dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["overall_assessment", "risk_level", "strengths", 
                       "weaknesses", "risk_factors", "prediction_analysis", "outlook"]
    
    for field in required_fields:
        if field not in verdict:
            return False
    
    valid_risk_levels = ["LOW", "MODERATE", "ELEVATED", "HIGH", "UNKNOWN"]
    if verdict.get("risk_level") not in valid_risk_levels:
        return False
    
    valid_outlooks = ["POSITIVE", "STABLE", "NEGATIVE", "UNKNOWN"]
    if verdict.get("outlook") not in valid_outlooks:
        return False
    
    return True


# Alternative simpler prompt for smaller models
SIMPLE_VERDICT_PROMPT = """
Company: {company_name} ({ticker}) - FY{fiscal_year}

Key Metrics:
- Current Ratio: {current_ratio:.2f}
- Debt/Equity: {debt_to_equity:.2f}
- ROA: {return_on_assets:.2%}
- Net Margin: {net_margin:.2%}
- KAMs: {kam_count}
- News Sentiment: {avg_sentiment:.2f}

ML Prediction: {predicted_rating}

Provide a brief credit assessment in 3-4 sentences covering:
1. Overall credit quality
2. Main strengths
3. Key risks

Assessment:
"""
