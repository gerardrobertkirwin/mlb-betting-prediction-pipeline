# MLB Betting Prediction Pipeline

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyMC](https://img.shields.io/badge/PyMC-Bayesian-orange)
![Status](https://img.shields.io/badge/Status-Production-green)

## Overview
This end-to-end system ingests historical MLB data, engineers baseball-specific (Sabermetric) features and trains a Bayesian Hierarchical model to identify market mispricing in sports betting.

**Tested Performance:** +1.39% ROI on out-of-sample data (2023 season, 1055 bets, +5% edge strategy)

## Highlights

### 1. Data Ingestion
**Custom API Wrappers:** API wrappers with backoff to handle potential MLB API rate limits.
**Data Lake Simulation:** Due to limitations with gambling APIs, a Parquet-based feature store was built to handle the nested JSON structures of existing data.
**Complex Merges:** Built around complex baseball-specific issues such as doubleheaders (two games in one day) by creating composite join keys to avoid these many-to-many join errors.

### 2. Bayesian modelling
A Bayesian Linear Model (using PyMC) was chosen because it's more robust than a standard logistic regression model. It accounts for random effects of baseball (of which there are many) and produces a range of probabilities. It creates a "Margin of Safety" where the bets are placed only when the model's confidence is significantly higher than the market odds.

### 3. Leakage-free Feature Engineering
Strict time-series splitting was used for validation along with rolling features to ensure today's predictions were based on yesterday's data. Validated via Unit Testing in `tests/test_features.py`.

## Architecture

The project is architected as a modular Python package (`src/`) rather than a collection of notebooks, ensuring reproducibility and deployment readiness.

```text
├── data/               # Raw JSON/Parquet storage (GitIgnored)
├── notebooks/          # R&D Sandbox (Proof of Concept)
├── src/
│   ├── data_loading.py # ETL Pipeline, API Wrappers, & Resilience Logic
│   ├── features.py     # Feature Store (Rolling Windows, Log5)
│   └── modelling.py     # PyMC Bayesian Inference Engine
├── tests/              # Unit Tests for Data Leakage
└── main.py             # CLI Entry Point
```

## Setup

### Installation
```bash
git clone https://github.com/gerardrobertkirwin/mlb-betting-prediction-pipeline.git
cd mlb-betting-prediction-pipeline
pip install -r requirements.txt
```

### Odds Data Setup:
1. Use the web scraper or download the dataset from: [here](https://github.com/ArnavSaraogi/mlb-odds-scraper)
2. Create the folder data/raw
3. Place the .json file in there (the download will be odds_history.json)

## References

### Data Sources
*   **Betting Data:** Historical odds scraping logic adapted from [mlb-odds-scraper](https://github.com/ArnavSaraogi/mlb-odds-scraper) by Arnav Saraogi.
*   **Game Stats:** Official MLB data via the [MLB-StatsAPI](https://pypi.org/project/MLB-StatsAPI/).

### Sabermetrics & Theory
*   **Log5 Probability:** Bill James' formula for estimating win probability based on two teams' winning percentages. Used as a baseline feature for the model to capture "Theoretical Win Probability."
    *   *Formula:* $P(A) = \frac{p_A(1 - p_B)}{p_A(1 - p_B) + p_B(1 - p_A)}$
*   **Pythagorean Expectation:** A metric to estimate a team's true strength based on Run Differential rather than their actual Win/Loss record.
    *   *Formula:* $Win\% = \frac{Runs^2}{Runs^2 + Allowed^2}$
*   **Performance Metrics:** The model utilizes 10-game rolling averages of **Runs Scored**, **Runs Allowed**, **Hits**, and **Errors** to capture immediate team form rather than season-long averages.
