# ⚡ GridMind

[![Tests](https://github.com/akashjhajharia123-ui/GridMind/actions/workflows/tests.yml/badge.svg)](https://github.com/akashjhajharia123-ui/GridMind/actions/workflows/tests.yml)

**GridMind** is an end-to-end energy demand forecasting and residual anomaly intelligence system built for leakage-safe time-series evaluation, model selection, reserved-test assessment, and interactive operational diagnostics.

The project combines machine-learning forecasting, expanding-window backtesting, residual-based anomaly detection, and a Streamlit dashboard into a reproducible workflow.

## Live Demo

Explore the deployed GridMind dashboard:

**GridMind Energy Intelligence Dashboard**
https://gridmind-ai.streamlit.app

## Key Features

- Chronological train, validation, and reserved-test splitting
- Leakage-safe lag and rolling-window feature engineering
- Random Forest and HistGradientBoosting challenger models
- Seasonal Naive benchmark comparison
- Expanding-window temporal backtesting
- Frozen final-model selection protocol
- Reserved-test evaluation without post-hoc tuning
- Residual anomaly detection using a frozen `q = 0.99` threshold
- Severity and direction classification for anomaly events
- Interactive Streamlit monitoring dashboard
- Forecast diagnostics and feature-importance analysis
- Automated test suite with **48 passing tests**

## Data

GridMind operates on hourly energy-demand observations with the following required schema:

- `Datetime` — hourly timestamp
- `AEP_MW` — electricity demand in megawatts

The current pipeline reads the dataset from:

```text
data/raw/AEP_hourly.csv
```

Records are parsed chronologically, sorted by timestamp, and deduplicated on `Datetime` before downstream feature engineering and evaluation.

### Dataset Source

This project uses the `AEP_hourly.csv` series from the **Hourly Energy Consumption** dataset published on Kaggle by Rob Mulla (`robikscube`). The dataset collection contains hourly power-consumption data sourced from PJM and measured in megawatts (MW).

- Dataset: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
- Original data source: PJM Interconnection

## Model Performance

### Final Reserved-Test Results

| Model | MAE (MW) | RMSE (MW) | MAPE (%) |
|---|---:|---:|---:|
| Seasonal Naive | 1444.95 | 1910.84 | 9.580 |
| Random Forest | 705.91 | 945.25 | 4.721 |

The selected **Random Forest** model substantially reduced reserved-test error relative to the Seasonal Naive benchmark.

### Backtest Selection

| Model | Mean Backtest MAPE | Fold Variability |
|---|---:|---:|
| Random Forest | 4.634% | 0.265 |
| HistGradientBoosting | 4.639% | 0.251 |

Models within the practical competitiveness margin were resolved using holdout validation performance without reopening the reserved final test set.

## Evaluation Integrity

GridMind separates model development, selection, final evaluation, and anomaly calibration to reduce temporal leakage.

1. **Chronological Splitting**  
   Train, validation, and reserved-test periods preserve temporal order. Random shuffling is not used.

2. **Leakage-Safe Features**  
   Lag and rolling features use historical demand information rather than future target values.

3. **Challenger Evaluation**  
   Random Forest and HistGradientBoosting are compared against a Seasonal Naive benchmark.

4. **Expanding-Window Backtesting**  
   Temporal folds evaluate model stability across multiple historical forecast origins.

5. **Frozen Final Protocol**  
   Final evaluation logic is committed before reserved-test evaluation.

6. **Separate Anomaly Calibration**  
   A train-only calibration model predicts unseen validation data. The `q = 0.99` residual threshold is then frozen for test scoring.

## Anomaly Intelligence

GridMind identifies unusually large forecast residuals using a threshold calibrated independently from the final test set.

Current calibration record:

- Threshold quantile: `q = 0.99`
- Frozen residual threshold: `3,019 MW`
- Full-test anomaly events: `78`
- Full-test anomaly rate: `0.429%`
- Final-test usage: scoring only

Visible anomaly events can be filtered by date range, severity, and direction.

Anomalies are categorized as demand **spikes** or **drops** based on residual direction.

## Forecast Diagnostics

The dashboard includes:

- Window R²
- Mean bias
- Median absolute error
- 95th percentile error
- Residuals over time
- Actual vs predicted calibration view
- Seven-day rolling MAE
- Model comparison
- Feature importance
- Backtest stability
- Highest-severity anomaly events
- Anomaly severity mix
- Event direction intelligence

## Project Structure

```text
GridMind/
├── dashboard/
│   └── app.py
├── scripts/
│   ├── analyze_model.py
│   ├── backtest_models.py
│   ├── compare_models.py
│   ├── detect_anomalies.py
│   ├── final_evaluate.py
│   ├── select_model.py
│   └── train_baseline.py
├── src/
│   ├── __init__.py
│   ├── anomaly.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── features.py
│   ├── forecast_features.py
│   ├── model.py
│   ├── model_io.py
│   └── split.py
├── tests/
├── app.py
├── pytest.ini
├── requirements.txt
└── README.md
```

## Installation

Clone the repository and create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducible Workflow

From the project root, the main pipeline stages can be executed as:

```bash
python scripts/train_baseline.py
python scripts/compare_models.py
python scripts/backtest_models.py
python scripts/select_model.py
python scripts/final_evaluate.py
python scripts/detect_anomalies.py
```

Optional model diagnostics can be generated with:

```bash
python scripts/analyze_model.py
```

The workflow separates baseline training, challenger comparison, temporal backtesting, model selection, reserved-test evaluation, and anomaly scoring into explicit stages.

## Run the Dashboard

```bash
streamlit run dashboard/app.py
```

## Run Tests

```bash
python -m pytest -q
```

Expected result:

```text
48 passed
```

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- Streamlit
- Plotly
- joblib
- pytest

## Design Philosophy

GridMind is built around a simple principle:

> Forecast accuracy is not enough. Evaluation integrity, temporal discipline, reproducibility, and operational interpretability matter too.

The system therefore treats model selection, final evaluation, and anomaly calibration as separate stages with explicit boundaries.

## Author

**Akash Jhajharia**