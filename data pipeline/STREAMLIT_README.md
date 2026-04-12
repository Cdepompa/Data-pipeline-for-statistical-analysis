# BTC Statistical Analysis Dashboard

A comprehensive Streamlit web application for analyzing Bitcoin price data with Fear & Greed Index and holiday effects.

## Features

The dashboard includes five main analysis sections:

### 1. Overview
- Dataset summary statistics
- Basic distributions of BTC returns, volume, and days to holiday
- Key metrics and data preview

### 2. T-Test Analysis
- One-sample t-tests: Fear vs non-fear returns compared to zero
- Welch two-sample t-test: Comparing fear vs non-fear mean returns
- Visualizations: Histograms and distribution plots

### 3. Chi-Square Analysis
- Test of independence between BTC gains and holiday status
- Contingency table analysis
- Percentage gain analysis by holiday status

### 4. Correlation Analysis
- Spearman correlation between BTC volume and days to holiday
- Scatter plot visualization with log-scaled volume

### 5. Variance Test Analysis
- ANOVA F-test for mean differences across day types
- Levene's and Bartlett's tests for variance equality
- Pairwise F-tests between weekdays, weekends, and holidays
- KDE plots, box plots, and variance comparison charts

## Data Requirements

The app uses the processed dataset: `data/gold/btc_fng_with_holidays_api.csv`

Required columns:
- `date`: Date of observation
- `btc_return_pct`: Daily BTC return percentage
- `btc_volume`: BTC trading volume
- `fear`: Boolean indicating fear status
- `is_holiday`: Boolean indicating holiday status
- `days_to_holiday`: Days until nearest holiday
- `is_weekend`: Boolean indicating weekend status

## Installation & Running

1. Ensure you have Python installed
2. Install required packages:
   ```bash
   pip install streamlit pandas matplotlib seaborn scipy
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Open your browser to the provided local URL (typically http://localhost:8501)

## Navigation

Use the sidebar to navigate between different analysis sections. Each section provides:
- Statistical test results with p-values
- Interpretation of results
- Interactive visualizations
- Summary statistics

## Statistical Tests Summary

- **T-tests**: Compare means between fear/non-fear groups
- **Chi-square**: Test relationship between gains and holidays
- **Spearman correlation**: Measure monotonic relationship between volume and holiday proximity
- **Variance tests**: Check if return variances differ by day type (weekday/weekend/holiday)

All tests include appropriate visualizations and clear interpretations of the results.</content>
<parameter name="filePath">c:\Users\Christopher\Documents\data pipeline\STREAMLIT_README.md