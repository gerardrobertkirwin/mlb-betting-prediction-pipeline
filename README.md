# MLB Betting Prediction Pipeline

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyMC](https://img.shields.io/badge/PyMC-Bayesian-orange)
![Status](https://img.shields.io/badge/Status-Production-green)

##Overview
This end-to-end system ingests historical MLB data, engineers baseball-specific (sabremetric) features and trains a Bayesian Hierarchical model to identify market mispricing in sports betting.

**Tested Performance** +1.39% ROI on out-of-sample data (2023 season, 1055 bets, +5% edge strategy)

##Architecture

##Setup


Odds Data Setup:
1. Use the web scraper or download the dataset from: [https://github.com/ArnavSaraogi/mlb-odds-scraper][here]
2. Create a folder such as data/raw
3. Place the .json file in there (the download will be odds_history.json)
