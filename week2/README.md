# IT vs Banking Sector Returns — A/B Test Analysis

## What
A statistical analysis comparing daily stock returns of India's IT sector vs Banking sector using NIFTY-50 historical data (2000–2021). The goal: determine whether the difference in sector returns is statistically significant or just random noise.

## Why
If an investor notices Banking stocks returning slightly more than IT stocks, should they rebalance their portfolio? This project uses hypothesis testing (t-test) to find out whether the gap is **real** or just **luck**.

## Data
- **Source:** [NIFTY-50 Stock Market Data](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data) (Kaggle)
- **Stocks Used:**
  - IT Sector: INFY, WIPRO, TCS
  - Banking Sector: AXISBANK, HDFCBANK, ICICIBANK
- **Total Records:** 30,669 trading days across all stocks

> **Note:** The dataset is not included in this repo. Download it from the Kaggle link above, and place the CSV files in a `data/` folder inside `week2/`.

### Cleaning Steps
1. **Ticker symbol corrections** — Historical data contained outdated symbols: `INFOSYSTCH` → `INFY` (Infosys rebranded its ticker) and `UTIBANK` → `AXISBANK` (UTI Bank rebranded to Axis Bank ~2007). Without this fix, the sector assignment and row counts would have been wrong.
2. **Date conversion** — `Date` column was stored as `object`; converted to `datetime64` for time-based analysis.
3. **Dropped irrelevant columns** — Removed Series, Open, High, Low, Volume, Turnover, Last, VWAP, Trades, Deliverable Volume, and %Deliverable. Only Date, Symbol, Close, and Prev Close were needed to calculate daily returns.

## Findings

| Metric | IT Sector | Banking Sector |
|--------|-----------|----------------|
| Mean Daily Return | 0.031% | 0.098% |

- **Null Hypothesis:** There is no real difference between IT and Banking daily returns — the observed gap is just noise.
- **Test Used:** Independent samples t-test (`scipy.stats.ttest_ind`)
- **p-value:** 0.04 (threshold: 0.05)
- **Conclusion:** The difference is **statistically significant** (p < 0.05) — we reject the null hypothesis. The gap between sectors is real, not random noise.
- **Practical Significance:** However, the actual difference is just **0.067% per day**. This is too small to justify rebalancing a portfolio based on sector alone. Statistical significance ≠ practical significance.

## How to Run

```bash
# Clone the repo
git clone https://github.com/pugal-mugilan/ai-ml-journey.git
cd ai-ml-journey/week2

# Install dependencies
pip install pandas numpy scipy

# Run the analysis
python data_cleaning.py
```

## Tools Used
- Python 3.x
- Pandas — data loading, cleaning, groupby analysis
- NumPy — conditional column assignment (`np.where`)
- SciPy — independent samples t-test