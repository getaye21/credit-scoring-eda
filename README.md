# Credit Scoring EDA – Ethiopian Inclusive Finance

## Objective
Identify key risk factors that predict a borrower’s financial distress (SeriousDlqin2yrs) using historical credit data. Support data‑driven lending in Ethiopia.

## Dataset
- 150,000 borrowers, anonymized.
- Target: `SeriousDlqin2yrs` (1 = distress within 2 years)
- Features: revolving utilization, age, debt ratio, monthly income, past due times, etc.

## How to run
1. Clone repo
2. Install: `pip install -r requirements.txt`
3. Run: `python eda_credit.py`
4. See results in `outputs/` folder.

## Key Findings
- **High revolving utilization (>1)** is a strong default signal.
- **Young (<25) and old (>65)** borrowers have higher default rates.
- **Past due events** (30‑59 days, 60‑89 days, 90+ days) are top predictors.
- Monthly income alone is a weak predictor – must combine with other features.

## Next Steps
- Build a logistic regression / XGBoost model.
- Package as a Model‑as‑a‑Service (MaaS) API for Ethiopian lenders.

## License
MIT
