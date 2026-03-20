"""
Improve 18 key verdicts in the fine-tuning dataset with analyst-style
language, Saudi market context, and more nuanced financial reasoning.

Replaces the formulaic template outputs for selected companies with
richer, expert-quality credit verdicts.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_FILE = PROJECT_ROOT / "data" / "processed" / "finetune_dataset.jsonl"

IMPROVED_VERDICTS = {
    0: {  # MEPCO (1202.SR) - Materials, A-
        "company": "MEPCO", "ticker": "1202.SR", "fiscal_year": 2024,
        "predicted_rating": "A", "risk_classification": "Investment Grade",
        "overall_assessment": "MEPCO demonstrates a resilient credit profile underpinned by strong liquidity and a conservative capital structure, though near-zero operating profitability in FY2024 warrants close monitoring.",
        "strengths": [
            "Robust liquidity buffer (LIQUID=0.2396) provides ample headroom to absorb short-term cash flow volatility in the Saudi packaging sector",
            "Equity-dominated balance sheet (LEVERAGE=1.6692) significantly reduces refinancing risk and signals conservative financial management",
            "Positive retained earnings base (CUMPROF=0.0720) indicates the company has historically generated surplus cash flow"
        ],
        "weaknesses": [
            "Operating profitability has turned negative (PROFITAB=-0.0095), suggesting margins are under pressure from input costs or pricing competition in the paper and packaging industry"
        ],
        "key_risks": [
            "Sustained negative operating returns could erode the retained earnings buffer and weaken the company's self-funding capacity",
            "As a materials company, MEPCO is exposed to commodity price swings and shifts in Saudi industrial demand linked to Vision 2030 infrastructure spending"
        ],
        "prediction_analysis": "The A prediction is supported by the company's strong balance sheet metrics (LIQUID=0.2396, LEVERAGE=1.6692), which offset the temporary profitability dip. The actual Financial Analytics rating of A- is one notch lower, likely reflecting the agency's greater weight on current-period earnings. The model's reliance on balance sheet strength over income statement metrics explains the slight overestimation."
    },

    7: {  # SABIC (2010.SR) FY2024 - Materials, A+
        "company": "SABIC", "ticker": "2010.SR", "fiscal_year": 2024,
        "predicted_rating": "A", "risk_classification": "Investment Grade",
        "overall_assessment": "SABIC maintains a strong investment-grade profile backed by its position as one of the world's largest petrochemical producers and its strategic importance to Saudi Arabia's industrial diversification agenda under Vision 2030.",
        "strengths": [
            "Well-capitalized balance sheet with equity nearly doubling total liabilities (LEVERAGE=1.9545), reflecting conservative financial policy consistent with its majority Saudi Aramco ownership",
            "Healthy liquidity position (LIQUID=0.1655) ensures adequate working capital for capital-intensive petrochemical operations",
            "Positive cumulative profitability (CUMPROF=0.0721) demonstrates consistent long-term earnings retention despite cyclical commodity markets"
        ],
        "weaknesses": [
            "Operating profitability has compressed to PROFITAB=0.0250, reflecting the global petrochemical downcycle and softening demand in key export markets"
        ],
        "key_risks": [
            "Petrochemical margins remain under pressure from overcapacity in Asia and slower Chinese demand growth",
            "Ongoing integration with Saudi Aramco's downstream strategy could reshape SABIC's standalone credit profile"
        ],
        "prediction_analysis": "The financial ratios strongly support an investment-grade rating. The model predicts A while Fitch assigns A+, a one-notch difference likely attributable to SABIC's strategic importance and implicit sovereign support that financial ratios alone cannot capture. The LEVERAGE of 1.9545 and LIQUID of 0.1655 are consistent with a mid-to-upper A rating."
    },

    17: {  # ACWA Power (2082.SR) FY2024 - Utilities, BBB-
        "company": "ACWA Power", "ticker": "2082.SR", "fiscal_year": 2024,
        "predicted_rating": "BBB", "risk_classification": "Investment Grade",
        "overall_assessment": "ACWA Power occupies a critical role in Saudi Arabia's energy transition as a leading developer of renewable and desalination projects, though its leveraged project-finance model places it at the lower end of investment grade.",
        "strengths": [
            "Improving cumulative profitability (CUMPROF=0.0857) signals that the company's project portfolio is maturing and generating consistent cash flows",
            "Adequate liquidity (LIQUID=0.0962) supported by long-term power purchase agreements with sovereign-backed offtakers",
            "Solid operating efficiency (PROFITAB=0.0583) demonstrates the earnings power of its contracted asset base"
        ],
        "weaknesses": [
            "Leverage remains elevated (LEVERAGE=0.7461), typical for infrastructure developers but leaving limited headroom for further project debt",
            "The project-finance business model inherently concentrates risk in large, capital-intensive assets with long payback periods"
        ],
        "key_risks": [
            "Rapid expansion into new geographies and technologies (green hydrogen, wind) introduces execution and currency risk",
            "Rising global interest rates increase the cost of refinancing project-level debt across ACWA Power's portfolio"
        ],
        "prediction_analysis": "The model's BBB prediction aligns with the lower end of investment grade, consistent with ACWA Power's moderate leverage (LEVERAGE=0.7461) and adequate but not exceptional liquidity (LIQUID=0.0962). Fitch's BBB- rating is one notch lower, reflecting the agency's view of project-finance concentration risk that pure financial ratios do not fully capture."
    },

    23: {  # Saudi Aramco (2222.SR) FY2024 - Energy, A+
        "company": "Saudi Aramco", "ticker": "2222.SR", "fiscal_year": 2024,
        "predicted_rating": "A", "risk_classification": "Investment Grade",
        "overall_assessment": "Saudi Aramco remains the world's most profitable company with an exceptionally strong balance sheet, though its credit rating is constrained by the sovereign ceiling of the Kingdom of Saudi Arabia rather than its standalone fundamentals.",
        "strengths": [
            "Industry-leading profitability (PROFITAB=0.3270) far exceeds global energy sector peers, driven by the lowest upstream production costs worldwide",
            "Exceptional retained earnings base (CUMPROF=0.5539) reflects decades of profit accumulation and positions the company uniquely in global capital markets",
            "Conservative leverage (LEVERAGE=2.1383) with equity more than doubling total liabilities, providing substantial financial flexibility"
        ],
        "weaknesses": [
            "Liquidity ratio (LIQUID=0.0974) is moderate relative to the company's enormous asset base, though absolute cash reserves are substantial",
            "High dividend commitments to the Saudi government limit retained cash flow available for deleveraging"
        ],
        "key_risks": [
            "Credit rating is effectively capped by Saudi Arabia's sovereign rating, meaning Aramco's standalone creditworthiness exceeds its assigned rating",
            "Energy transition and OPEC+ production discipline could constrain volume growth in the medium term"
        ],
        "prediction_analysis": "Aramco's financial ratios (PROFITAB=0.3270, CUMPROF=0.5539, LEVERAGE=2.1383) would support a rating well above A+ on standalone fundamentals. The model predicts A, one notch below Fitch's A+, because the model cannot account for sovereign support and strategic importance. Moody's assigns AA- to the same financials, illustrating the role of qualitative sovereign linkage in credit assessment."
    },

    26: {  # Almarai (2280.SR) FY2024 - Food & Beverages, BBB-
        "company": "Almarai", "ticker": "2280.SR", "fiscal_year": 2024,
        "predicted_rating": "BBB", "risk_classification": "Investment Grade",
        "overall_assessment": "Almarai is the GCC's largest integrated dairy and food company with strong brand equity, though its capital-intensive vertically integrated model results in moderate leverage metrics.",
        "strengths": [
            "Strong cumulative profitability (CUMPROF=0.2158) reflects Almarai's dominant market position and pricing power in Saudi Arabia's food sector",
            "Solid operating returns (PROFITAB=0.0855) supported by vertical integration from farming to retail distribution",
            "Balanced capital structure (LEVERAGE=1.1201) with equity slightly exceeding liabilities"
        ],
        "weaknesses": [
            "Moderate liquidity (LIQUID=0.0828) reflects the working capital demands of perishable goods distribution and large-scale agricultural operations"
        ],
        "key_risks": [
            "Ongoing capex requirements for agricultural expansion and cold-chain logistics may pressure free cash flow generation",
            "Saudi food sector growth is tied to population dynamics and subsidy reform, which could affect consumer spending patterns"
        ],
        "prediction_analysis": "The financial profile supports a lower investment-grade rating. CUMPROF of 0.2158 and PROFITAB of 0.0855 demonstrate solid earnings, while LEVERAGE of 1.1201 is consistent with BBB territory. The actual Moody's/S&P BBB- rating is one notch below the model's BBB prediction, likely reflecting the agencies' assessment of capex-driven cash flow pressure."
    },

    28: {  # ADES (2382.SR) FY2024 - Energy, B+
        "company": "ADES", "ticker": "2382.SR", "fiscal_year": 2024,
        "predicted_rating": "BBB", "risk_classification": "Investment Grade",
        "overall_assessment": "ADES is a growing oilfield services provider with improving fundamentals following its recent acquisition-driven expansion, though its speculative-grade rating reflects the inherent operational risks and leverage of the energy services sector.",
        "strengths": [
            "Improving cumulative profitability (CUMPROF=0.1131) suggests the company's expanded drilling fleet is generating returns above its cost of capital",
            "Solid operating margins (PROFITAB=0.0825) indicate effective utilization of its drilling rigs and production services assets"
        ],
        "weaknesses": [
            "Near-zero liquidity (LIQUID=0.0097) leaves minimal buffer against short-term cash flow disruptions",
            "Elevated leverage (LEVERAGE=0.4332) reflects the debt-funded acquisition strategy that expanded ADES's Middle East and North Africa footprint"
        ],
        "key_risks": [
            "Oil price volatility directly impacts drilling activity levels and day-rates across ADES's contract portfolio",
            "Tight liquidity combined with high leverage creates refinancing risk if energy sector conditions deteriorate"
        ],
        "prediction_analysis": "The model predicts BBB, significantly above the actual Fitch/S&P B+ rating. This gap highlights a limitation of ratio-based models: ADES's moderate profitability ratios mask the cyclical and operational risks inherent in oilfield services. The agencies' B+ rating reflects sector risk, acquisition integration concerns, and the company's shorter track record at its current scale."
    },

    38: {  # Cenomi Centers (4321.SR) FY2024 - Real Estate, BB (Fitch)
        "company": "Cenomi Centers", "ticker": "4321.SR", "fiscal_year": 2024,
        "predicted_rating": "BBB", "risk_classification": "Investment Grade",
        "overall_assessment": "Cenomi Centers is a leading Saudi retail REIT operator benefiting from strong consumer spending, though its below-investment-grade rating reflects the structural risks of leveraged real estate in a transforming Saudi retail landscape.",
        "strengths": [
            "Strong cumulative profitability (CUMPROF=0.2779) driven by high occupancy rates across premium mall assets in key Saudi cities",
            "Adequate operating returns (PROFITAB=0.0446) supported by long-term lease contracts with major international and regional retailers"
        ],
        "weaknesses": [
            "Tight liquidity (LIQUID=0.0405) is characteristic of REITs but limits financial flexibility during market disruptions",
            "Leverage remains below parity (LEVERAGE=0.8919), meaning liabilities slightly exceed equity, typical for real estate but limiting rating upside"
        ],
        "key_risks": [
            "Saudi retail sector is undergoing structural change with e-commerce growth and entertainment sector expansion under Vision 2030",
            "Rating agencies disagree materially on Cenomi: Fitch assigns BB, Financial Analytics assigns A-, and S&P assigns BB-, illustrating the uncertainty around Saudi REIT valuations"
        ],
        "prediction_analysis": "The model predicts BBB based on the apparently solid profitability metrics (CUMPROF=0.2779), but Fitch's BB rating reflects real estate-specific risks that financial ratios alone cannot capture: tenant concentration, lease maturity profile, and asset valuation sensitivity. The three-agency disagreement (BB-, BB, A-) confirms that Cenomi sits at a genuinely ambiguous credit boundary."
    },

    43: {  # Saudi Electricity (5110.SR) FY2024 - Utilities, A+
        "company": "Saudi Electricity", "ticker": "5110.SR", "fiscal_year": 2024,
        "predicted_rating": "A", "risk_classification": "Investment Grade",
        "overall_assessment": "Saudi Electricity Company (SEC) is the Kingdom's dominant power transmission and distribution utility, with its credit profile anchored by essential-service status and strong government linkage rather than standalone financial metrics.",
        "strengths": [
            "Balanced capital structure (LEVERAGE=0.8503) with equity broadly matching liabilities, appropriate for a regulated utility",
            "Consistent cumulative profitability (CUMPROF=0.0748) reflects the predictable revenue stream from regulated tariffs"
        ],
        "weaknesses": [
            "Negative liquidity (LIQUID=-0.0508) indicates current liabilities exceed current assets, typical for capital-intensive utilities with large ongoing infrastructure programs",
            "Modest operating profitability (PROFITAB=0.0221) reflects regulated returns that cap upside potential"
        ],
        "key_risks": [
            "SEC's massive capital expenditure program for grid expansion and renewable integration under Vision 2030 will require continued access to debt markets",
            "Tariff reform and the ongoing unbundling of Saudi Arabia's power sector could alter SEC's revenue model and cost structure"
        ],
        "prediction_analysis": "The model predicts A despite SEC's negative liquidity (LIQUID=-0.0508), correctly identifying that utility business models can sustain negative working capital due to predictable cash flows. Fitch's A+ and Moody's AA- exceed the model's prediction because sovereign linkage and essential-service status provide credit uplift that financial ratios cannot capture."
    },

    46: {  # STC (7010.SR) FY2024 - Telecom, A+ (Fitch)
        "company": "Saudi Telecom (STC)", "ticker": "7010.SR", "fiscal_year": 2024,
        "predicted_rating": "A", "risk_classification": "Investment Grade",
        "overall_assessment": "STC is the Saudi Kingdom's largest telecommunications operator and a cornerstone of the national digital infrastructure strategy, with strong financial fundamentals reinforced by government ownership and strategic importance.",
        "strengths": [
            "Strong liquidity position (LIQUID=0.2105) provides substantial headroom for working capital and investment needs",
            "Robust cumulative profitability (CUMPROF=0.2394) demonstrates STC's ability to consistently retain earnings from its dominant market position",
            "Conservative capital structure (LEVERAGE=1.3570) with equity comfortably exceeding liabilities"
        ],
        "weaknesses": [
            "Operating profitability (PROFITAB=0.0832) is moderate for a telecom incumbent, reflecting competitive pressure and heavy network investment"
        ],
        "key_risks": [
            "Ongoing 5G infrastructure investment and fiber-to-the-home rollout require sustained capex that may pressure free cash flow",
            "STC's international expansion strategy (including stc Group restructuring) introduces new market and regulatory risks"
        ],
        "prediction_analysis": "STC's ratios (LIQUID=0.2105, CUMPROF=0.2394, LEVERAGE=1.3570) strongly support an upper investment-grade rating. The model predicts A, while agencies assign between A+ (Fitch, S&P) and AAA (Tassnief). The gap between international and local agency ratings illustrates the different weight placed on sovereign support and strategic importance in Saudi credit assessment."
    },

    50: {  # Mobily (7020.SR) FY2024 - Telecom, AA
        "company": "Mobily", "ticker": "7020.SR", "fiscal_year": 2024,
        "predicted_rating": "A", "risk_classification": "Investment Grade",
        "overall_assessment": "Mobily has completed a successful financial turnaround and now demonstrates solid profitability as Saudi Arabia's second-largest mobile operator, though its balance sheet carries the legacy of past restructuring.",
        "strengths": [
            "Strong cumulative profitability (CUMPROF=0.2907) reflects the successful recovery from the 2014-2017 financial difficulties and sustained dividend resumption",
            "Solid operating returns (PROFITAB=0.0984) driven by data revenue growth and network sharing efficiencies"
        ],
        "weaknesses": [
            "Slightly negative liquidity (LIQUID=-0.0238) indicates that current liabilities marginally exceed current assets, reflecting ongoing network investment cycles",
            "Leverage below parity (LEVERAGE=0.9611) means liabilities slightly exceed equity, limiting headroom for additional debt-funded investments"
        ],
        "key_risks": [
            "Intensifying competition from STC and Zain in 5G and fixed broadband could pressure market share and pricing",
            "The Financial Analytics AA rating appears optimistic given the balance sheet metrics; divergence from this assessment is likely"
        ],
        "prediction_analysis": "The model predicts A, which is a reasonable assessment given the mixed balance sheet signals (strong CUMPROF=0.2907 but negative LIQUID=-0.0238 and below-parity LEVERAGE=0.9611). The Financial Analytics AA rating is notably higher, likely reflecting qualitative factors such as Etihad Etisalat's strategic position in Saudi telecoms and its strong EBITDA generation."
    },

    52: {  # Zain KSA (7030.SR) FY2024 - Telecom, A
        "company": "Zain KSA", "ticker": "7030.SR", "fiscal_year": 2024,
        "predicted_rating": "BBB", "risk_classification": "Investment Grade",
        "overall_assessment": "Zain KSA is Saudi Arabia's third mobile operator, showing improved financial stability but still carrying the legacy of its highly leveraged launch phase.",
        "strengths": [
            "Positive operating profitability (PROFITAB=0.0448) confirms that Zain KSA has achieved sustained profitability after years of operating losses",
            "Modest but positive cumulative profitability (CUMPROF=0.0596) indicates the company is gradually rebuilding its retained earnings base"
        ],
        "weaknesses": [
            "Significantly negative liquidity (LIQUID=-0.1944) signals that current liabilities substantially exceed current assets, creating refinancing pressure",
            "Below-parity leverage (LEVERAGE=0.6143) reflects the remaining debt burden from network build-out and spectrum acquisition costs"
        ],
        "key_risks": [
            "The deeply negative liquidity ratio requires careful management of debt maturities and bank facility renewals",
            "As the smallest of three Saudi mobile operators, Zain KSA has limited pricing power and must compete on network quality and customer experience"
        ],
        "prediction_analysis": "The model predicts BBB, reflecting the negative liquidity (LIQUID=-0.1944) and below-parity leverage (LEVERAGE=0.6143). The actual Financial Analytics/Tassnief A rating is higher, likely factoring in the strategic value of a Saudi telecom license, the improving profitability trajectory, and potential implicit support from the Zain Group parent."
    },

    54: {  # Ladun Investment (9535.SR) FY2024 - BBB+
        "company": "Ladun Investment Company", "ticker": "9535.SR", "fiscal_year": 2024,
        "predicted_rating": "BBB", "risk_classification": "Investment Grade",
        "overall_assessment": "Ladun Investment is a diversified Saudi holding company with modest but positive financial metrics, positioned at the lower end of investment grade.",
        "strengths": [
            "Positive cumulative profitability (CUMPROF=0.0512) indicates the company has maintained a track record of earnings retention across its portfolio",
            "Positive operating returns (PROFITAB=0.0341) demonstrate that the investment portfolio generates adequate income"
        ],
        "weaknesses": [
            "Near-zero liquidity (LIQUID=0.0058) leaves virtually no working capital cushion for unexpected cash needs",
            "High leverage (LEVERAGE=0.2670) means liabilities are approximately 3.7x equity, creating significant refinancing and interest rate risk"
        ],
        "key_risks": [
            "The combination of minimal liquidity and high leverage creates vulnerability to any disruption in cash flow or credit markets",
            "As a diversified holding company, Ladun's credit profile depends on the performance of its underlying portfolio companies"
        ],
        "prediction_analysis": "The model's BBB prediction is consistent with the mixed financial picture: positive but thin profitability (PROFITAB=0.0341, CUMPROF=0.0512) offset by concerning leverage (LEVERAGE=0.2670) and liquidity (LIQUID=0.0058). The actual Tassnief BBB+ is one notch higher, possibly reflecting the agency's assessment of portfolio diversification benefits."
    },

    55: {  # Mayar Holding (9568.SR) FY2024 - BBB (Financial Analytics)
        "company": "Mayar Holding", "ticker": "9568.SR", "fiscal_year": 2024,
        "predicted_rating": "BBB", "risk_classification": "Investment Grade",
        "overall_assessment": "Mayar Holding presents a challenging credit profile with deeply negative liquidity and minimal equity cushion, though its operating profitability provides a partial offset.",
        "strengths": [
            "Reasonable operating profitability (PROFITAB=0.0783) suggests the company's core businesses generate adequate returns on assets"
        ],
        "weaknesses": [
            "Deeply negative liquidity (LIQUID=-0.3496) signals severe working capital strain, with current liabilities far exceeding current assets",
            "Extremely high leverage (LEVERAGE=0.0681) means liabilities are approximately 14.7x equity, placing the company in a highly geared financial position",
            "Near-zero cumulative profitability (CUMPROF=0.0022) indicates the company has retained virtually no earnings over its operating history"
        ],
        "key_risks": [
            "The combination of extreme leverage and deeply negative liquidity creates material going-concern risk if operating cash flows weaken",
            "Financial Analytics assigns BBB while Tassnief assigns BB (unsolicited), a two-notch-plus gap that reflects genuine uncertainty about Mayar's creditworthiness"
        ],
        "prediction_analysis": "The model predicts BBB, which may be overly optimistic given the extreme leverage (LEVERAGE=0.0681) and negative liquidity (LIQUID=-0.3496). The Tassnief BB (unsolicited) rating better reflects the balance sheet risk. This case illustrates a limitation of the model: the binary classification boundary may not adequately penalize extreme ratio outliers."
    },

    58: {  # Multi Business Group (9619.SR) FY2024 - BB+
        "company": "Multi Business Group", "ticker": "9619.SR", "fiscal_year": 2024,
        "predicted_rating": "A", "risk_classification": "Investment Grade",
        "overall_assessment": "Multi Business Group displays unusually strong financial ratios for a speculative-grade company, suggesting that its below-investment-grade rating is driven by qualitative factors beyond the balance sheet.",
        "strengths": [
            "Exceptional liquidity (LIQUID=0.7789) indicates that working capital represents over three-quarters of total assets, providing an extraordinary cash buffer",
            "Strong cumulative profitability (CUMPROF=0.2920) demonstrates consistent historical earnings retention",
            "High operating profitability (PROFITAB=0.2090) signals efficient asset utilization and strong margins",
            "Conservative capital structure (LEVERAGE=3.2734) with equity exceeding liabilities by more than 3x"
        ],
        "weaknesses": [
            "Despite strong ratios, the company carries a BB+ rating from both Financial Analytics and Tassnief, suggesting qualitative concerns not captured by financial metrics alone"
        ],
        "key_risks": [
            "The disconnect between strong ratios and speculative-grade rating may indicate governance, transparency, or business model concerns that rating agencies weigh heavily",
            "Small-to-mid-cap Saudi companies can face liquidity constraints in debt capital markets regardless of their financial fundamentals"
        ],
        "prediction_analysis": "This is the most significant model misclassification: the model predicts A based on universally strong ratios (LIQUID=0.7789, CUMPROF=0.2920, PROFITAB=0.2090, LEVERAGE=3.2734), yet both agencies assign BB+. This case demonstrates that credit ratings incorporate qualitative dimensions -- business model sustainability, management quality, governance, market position -- that financial ratios cannot capture. It validates the need for an LLM-based qualitative overlay alongside the ML model."
    },

    13: {  # Alkhorayef (2081.SR) FY2024 - Utilities, A (Tassnief)
        "company": "Alkhorayef for Water and Power Technologies Company", "ticker": "2081.SR", "fiscal_year": 2024,
        "predicted_rating": "A", "risk_classification": "Investment Grade",
        "overall_assessment": "Alkhorayef Water & Power benefits from Saudi Arabia's massive water infrastructure investment under Vision 2030, with strong profitability metrics supporting its investment-grade rating.",
        "strengths": [
            "Strong operating profitability (PROFITAB=0.1238) indicates efficient execution of water and power technology projects",
            "Healthy liquidity (LIQUID=0.1627) provides adequate buffer for project-based working capital cycles",
            "Robust cumulative profitability (CUMPROF=0.1537) reflects consistent earnings retention across business cycles"
        ],
        "weaknesses": [
            "Below-parity leverage (LEVERAGE=0.4839) indicates that liabilities exceed equity by approximately 2:1, reflecting the capital-intensive nature of infrastructure projects"
        ],
        "key_risks": [
            "Revenue concentration in Saudi government water and power contracts creates dependence on public sector capital budgets",
            "The water and power technology sector is increasingly competitive, with international players entering the Saudi market"
        ],
        "prediction_analysis": "The model's A prediction aligns closely with the Tassnief A rating. The strong profitability metrics (PROFITAB=0.1238, CUMPROF=0.1537) and adequate liquidity (LIQUID=0.1627) support an investment-grade assessment, though the below-parity leverage (LEVERAGE=0.4839) introduces moderate balance sheet risk."
    },

    19: {  # Napco National (2210.SR) FY2024 - BBB+ (Tassnief)
        "company": "Napco National", "ticker": "2210.SR", "fiscal_year": 2024,
        "predicted_rating": "BBB", "risk_classification": "Investment Grade",
        "overall_assessment": "Napco National faces significant liquidity challenges despite maintaining positive operating profitability, placing it at the lower boundary of investment grade.",
        "strengths": [
            "Positive operating profitability (PROFITAB=0.0442) indicates the company's packaging operations remain viable and cash-generative"
        ],
        "weaknesses": [
            "Severely negative liquidity (LIQUID=-0.4946) is the weakest in the dataset, indicating that current liabilities vastly exceed current assets and creating acute refinancing risk",
            "High leverage (LEVERAGE=0.2855) means liabilities are approximately 3.5x equity, reflecting heavy debt financing",
            "Near-zero cumulative profitability (CUMPROF=0.0052) suggests the company has retained virtually none of its historical earnings"
        ],
        "key_risks": [
            "The extreme negative liquidity ratio requires immediate attention to debt maturity management and could trigger covenant breaches",
            "As a packaging manufacturer, Napco is exposed to raw material price fluctuations and competitive pressure from regional producers"
        ],
        "prediction_analysis": "The model predicts BBB, broadly consistent with Tassnief's BBB+, though the extreme negative liquidity (LIQUID=-0.4946) is a significant red flag. The model may be underweighting liquidity risk relative to the operating profitability signal. A more granular model might assign a lower rating given the severity of the working capital deficit."
    },

    36: {  # Cenomi Centers (4321.SR) FY2023 - Real Estate, BB (Fitch)
        "company": "Cenomi Centers", "ticker": "4321.SR", "fiscal_year": 2023,
        "predicted_rating": "BBB", "risk_classification": "Investment Grade",
        "overall_assessment": "Cenomi Centers in FY2023 showed deteriorating liquidity while maintaining strong cumulative profitability, a pattern typical of real estate companies undergoing portfolio repositioning.",
        "strengths": [
            "Strong cumulative profitability (CUMPROF=0.2966) driven by stable rental income from prime mall locations in Saudi Arabia's major cities",
            "Improving operating returns (PROFITAB=0.0595) suggest positive leasing momentum"
        ],
        "weaknesses": [
            "Negative liquidity (LIQUID=-0.0913) indicates current obligations exceed current assets, creating short-term funding pressure",
            "Leverage near parity (LEVERAGE=1.0649) provides limited equity cushion against asset value declines"
        ],
        "key_risks": [
            "Saudi retail real estate valuations are sensitive to occupancy rates, which could be disrupted by rapid e-commerce adoption and new entertainment-focused developments",
            "Rising interest rates in 2023 increase the cost of servicing Cenomi's floating-rate debt facilities"
        ],
        "prediction_analysis": "The model predicts BBB based on the strong CUMPROF of 0.2966, but Fitch assigns BB, reflecting the structural leverage and asset-concentration risks inherent in Saudi REITs. The one-year deterioration in LIQUID from 0.0078 (FY2022) to -0.0913 (FY2023) signals a negative trajectory that the model's static ratio analysis may not fully penalize."
    },

    2: {  # Maaden (1211.SR) FY2024 - Materials, BBB+ (Fitch)
        "company": "Maaden", "ticker": "1211.SR", "fiscal_year": 2024,
        "predicted_rating": "BBB", "risk_classification": "Investment Grade",
        "overall_assessment": "Ma'aden (Saudi Arabian Mining Company) is a strategically important national champion in the mining and fertilizer sectors, with its credit profile supported by steady commodity revenues and government backing.",
        "strengths": [
            "Adequate liquidity (LIQUID=0.0823) provides a working capital buffer for cyclical mining operations",
            "Positive cumulative profitability (CUMPROF=0.0874) reflects consistent earnings retention despite volatile commodity prices",
            "Solid operating returns (PROFITAB=0.0604) indicate efficient mine and plant operations"
        ],
        "weaknesses": [
            "Leverage near parity (LEVERAGE=1.0951) provides limited headroom for additional debt to fund the significant capital expenditure required for mining expansion"
        ],
        "key_risks": [
            "Global phosphate and aluminium price cycles directly impact Ma'aden's revenue and profitability",
            "Ambitious expansion projects (including the phosphate expansion at Wa'ad Al Shamal) require sustained capital investment and carry execution risk"
        ],
        "prediction_analysis": "The model predicts BBB, one notch below Fitch's BBB+. The ratios are consistent with the lower end of the BBB range: LIQUID=0.0823 is adequate, CUMPROF=0.0874 and PROFITAB=0.0604 are positive but not strong, and LEVERAGE=1.0951 is near parity. Fitch's higher rating likely reflects Ma'aden's strategic importance to Saudi Arabia's mining sector and government ownership stake."
    },
}


def main():
    print("=" * 60)
    print("IMPROVING FINE-TUNING VERDICTS")
    print("=" * 60)

    with open(DATASET_FILE) as f:
        examples = [json.loads(line) for line in f]

    print(f"Loaded {len(examples)} examples")
    print(f"Improving {len(IMPROVED_VERDICTS)} verdicts...\n")

    for idx, improved in IMPROVED_VERDICTS.items():
        if idx < len(examples):
            examples[idx]["output"] = json.dumps(improved, indent=2)
            print(f"  [{idx:>2}] {improved['company']:<45} {improved['predicted_rating']:>4} -> improved")

    with open(DATASET_FILE, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nSaved {len(examples)} examples (with {len(IMPROVED_VERDICTS)} improved)")


if __name__ == "__main__":
    main()
