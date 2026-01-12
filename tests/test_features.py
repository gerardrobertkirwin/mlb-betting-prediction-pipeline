import sys
import os
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = TEST_DIR.parent

sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pytest
from src.mlb_betting import features

def test_rolling_features_shift_logic():
    """
    CRITICAL: Verifies that rolling features do NOT include the current game.
    If this fails, we have Data Leakage.
    """
    # 1. Create Mock Data (One team, 3 days)
    df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'team': ['NYY', 'NYY', 'NYY'],
        'opponent': ['BOS', 'BOS', 'BOS'],
        'runs_scored': [5, 2, 10],   # Game 1: 5, Game 2: 2
        'runs_allowed': [1, 1, 1],
        'hits': [1,1,1],
        'errors': [0,0,0],
        'is_home': [1,1,1],
        'result': [1,1,1],
        'moneyline_closing': [-110, -110, -110]
    })

    # 2. Run Feature Engineering with window=2
    df_engineered = features.calculate_rolling_features(df, window_size=2)

    # 3. Assertions
    
    # Check Game 1: Should be NaN (No history yet)
    # If this is 5, we forgot to shift!
    assert np.isnan(df_engineered.loc[0, 'rolling_2_runs_scored']), "Game 1 should have NaN rolling stats"

    # Check Game 2: Should match Game 1's runs (5.0)
    # The rolling window (size 2) of the past (Game 1) is just 5.0
    assert df_engineered.loc[1, 'rolling_2_runs_scored'] == 5.0, "Game 2 rolling stat should equal Game 1 actual"

    # Check Game 3: Should match Average(Game 1, Game 2) -> (5 + 2) / 2 = 3.5
    # Crucially, it should NOT include Game 3's runs (10)
    actual_rolling = df_engineered.loc[2, 'rolling_2_runs_scored']
    assert actual_rolling == 3.5, f"Expected 3.5, got {actual_rolling}. Leakage detected!"

    print("Rolling Window Test Passed: No Data Leakage detected.")
