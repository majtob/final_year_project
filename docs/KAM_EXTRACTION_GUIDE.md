# KAM Extraction Guide

## Overview

Key Audit Matters (KAMs) are paragraphs in audit reports where auditors highlight:
- Significant risks of material misstatement
- Areas requiring significant judgment by management
- Critical accounting estimates

Based on research by Muñoz-Izquierdo et al. (2022), KAMs can predict credit ratings with **74% accuracy** alone, and **84% accuracy** when combined with financial ratios.

---

## KAM Categories (5 Types)

| Category | Variable | What to Look For |
|----------|----------|------------------|
| **Going Concern** | `kam_going_concern` | Viability concerns, refinancing risks, negative working capital, covenant breaches |
| **Revenue** | `kam_revenue` | Revenue recognition timing, multiple element arrangements, contract accounting, percentage of completion |
| **Assets** | `kam_assets` | Impairment of goodwill, intangibles, property, inventory valuation, receivables allowances |
| **Liabilities** | `kam_liabilities` | Contingent liabilities, provisions, pension obligations, debt covenants, tax uncertainties |
| **Other** | `kam_other` | IT systems, business combinations, related party transactions, regulatory compliance |

---

## Option 1: Manual Extraction (Recommended for Accuracy)

### Step-by-Step Process

1. **Download Annual Reports**
   - Source: [Tadawul Company Disclosures](https://www.saudiexchange.sa/)
   - Navigate to: Issuer News → Financial Reports → Annual Reports
   - Download the English version (if available)

2. **Locate the Audit Report**
   - Usually in the first 10-20 pages
   - Look for: "Independent Auditor's Report"
   - KAMs section titled: "Key Audit Matters" or "Matters Most Significant to the Audit"

3. **Extract KAM Information**
   For each KAM, record:
   - Title/Topic
   - Category (one of the 5 above)
   - Brief description
   
4. **Create Binary Features**
   ```
   kam_going_concern: 1 if any going concern KAM, else 0
   kam_revenue: 1 if any revenue recognition KAM, else 0
   kam_assets: 1 if any asset valuation KAM, else 0
   kam_liabilities: 1 if any liability/debt KAM, else 0
   kam_other: 1 if any other type KAM, else 0
   kam_count: total number of KAMs
   ```

### Time Estimate
- ~15-30 minutes per company per year
- 27 listed companies × 4 years = 108 company-years
- Total: ~27-54 hours of manual work

---

## Option 2: Semi-Automated with LLM (Faster)

Use an LLM to extract and classify KAMs from PDF text.

### Approach

1. **Extract text from audit report PDFs**
   - Use `pdfplumber` or `PyMuPDF` to extract text
   - Identify the KAM section using regex/keywords

2. **Use LLM to classify KAMs**
   ```python
   prompt = '''
   Extract Key Audit Matters from the following audit report section.
   For each KAM, classify it into ONE of these categories:
   - going_concern: viability, refinancing, covenant issues
   - revenue: revenue recognition, contract accounting
   - assets: impairment, valuation of assets
   - liabilities: provisions, contingencies, debt issues
   - other: IT systems, acquisitions, regulatory
   
   Return JSON format:
   {
     "kams": [
       {"title": "...", "category": "...", "description": "..."}
     ]
   }
   
   Audit Report Section:
   {text}
   '''
   ```

3. **Convert to binary features**

### Pros/Cons
- **Pros:** 5-10x faster than manual, consistent categorization
- **Cons:** May miss nuances, requires verification

---

## Option 3: Keyword-Based Extraction (Fastest, Less Accurate)

Use regular expressions to detect KAM categories.

```python
KAM_KEYWORDS = {
    'going_concern': ['going concern', 'viability', 'refinancing', 'covenant', 'liquidity risk'],
    'revenue': ['revenue recognition', 'contract revenue', 'percentage of completion', 'multiple element'],
    'assets': ['impairment', 'goodwill', 'intangible', 'valuation of', 'fair value of assets'],
    'liabilities': ['provision', 'contingent liab', 'pension', 'debt', 'borrowings'],
    'other': ['acquisition', 'business combination', 'IT system', 'internal control', 'related party']
}

def classify_kam(kam_text):
    kam_lower = kam_text.lower()
    for category, keywords in KAM_KEYWORDS.items():
        if any(kw in kam_lower for kw in keywords):
            return category
    return 'other'
```

---

## Data Template

Use the template at `data/templates/kams_template.csv`:

| Column | Type | Description |
|--------|------|-------------|
| ticker | string | Tadawul ticker (e.g., 7010.SR) |
| company_name | string | Full company name |
| fiscal_year | int | Year of annual report (2021-2024) |
| kam_going_concern | binary | 1 if going concern KAM present |
| kam_revenue | binary | 1 if revenue KAM present |
| kam_assets | binary | 1 if asset KAM present |
| kam_liabilities | binary | 1 if liability KAM present |
| kam_other | binary | 1 if other KAM present |
| kam_count | int | Total number of KAMs |
| kam_descriptions | string | Semicolon-separated KAM titles |
| source_document | string | Source file reference |
| notes | string | Additional notes |

---

## Priority Companies

Start with companies that have Tassnief ratings (27 listed):

1. Saudi Telecom Company (7010.SR)
2. Etihad Etisalat - Mobily (7020.SR)
3. Zain Saudi Arabia (7030.SR)
4. Dr. Sulaiman Al-Habib (4013.SR)
5. Middle East Healthcare (4004.SR)
6. Al Hammadi Holding (4007.SR)
7. CATRION (6004.SR)
8. Arabian Centers (4321.SR)
9. ...continue with remaining rated companies

---

## Expected Features for ML Model

After KAM extraction, combine with:

1. **Financial Ratios** (from Tadawul/yfinance):
   - Liquidity: Working Capital / Total Assets
   - Cumulative Profitability: Retained Earnings / Total Assets
   - Profitability: EBIT / Total Assets
   - Leverage: Book Value of Equity / Total Liabilities

2. **KAM Features**:
   - 5 binary KAM category indicators
   - KAM count

3. **Control Variables**:
   - Firm age
   - Firm size
   - Industry sector

---

## References

Muñoz-Izquierdo, N., Segovia-Vargas, M.J., Camacho-Miñano, M.M., & Pérez-Pérez, Y. (2022). 
Machine learning in corporate credit rating assessment using the expanded audit report. 
*Machine Learning*, 111, 4183–4215. https://doi.org/10.1007/s10994-022-06226-4
