import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from src.mlb_betting.config import PROJECT_ROOT, DATA_DIR
from src.mlb_betting.data_loading import load_and_merge_data
from src.mlb_betting import features
from src.mlb_betting.modeling import BayesianBettingModel, simulate_betting

def run_pipeline():
    print("STARTING PIPELINE")
    
    # --- 1. INGESTION ---
    # We use 2023 data for this demo
    season = 2023
    odds_file = DATA_DIR / "raw" / "odds_history.json"
    
    print(f"\n--- Phase 1: Ingestion (Season {season}) ---")
    if not odds_file.exists():
        print(f"❌ Error: Odds file not found at {odds_file}")
        return

    df_raw = load_and_merge_data(season=season, odds_filepath=str(odds_file))
    print(f"✅ Loaded {len(df_raw)} games.")

    # --- 2. FEATURE ENGINEERING ---
    print("\n--- Phase 2: Feature Engineering ---")
    
    df_long = features.create_team_centric_df(df_raw)
    
    df_rolling = features.calculate_rolling_features(df_long, window=10)
    
    df_adv = features.calculate_advanced_features(df_rolling)
    
    df_train = features.finalize_training_data(df_adv)
    
    print(f"Engineered features. Training set: {len(df_train)} rows.")

    # --- 3. MODEL TRAINING ---
    print("\n--- Phase 3: Model Training ---")
    model_path = DATA_DIR / "models" / "bayesian_v1.nc"
    
    model = BayesianBettingModel(model_path=str(model_path))
    
    
    feature_cols = [
        'is_home', 'rest_days', 'log5_prob', 
        'rolling_10_runs_scored', 'rolling_10_runs_allowed',
        'rolling_10_hits', 'rolling_10_errors',
        'rolling_pythag_win_pct', 'opp_pythag_win_pct',
        'team_code', 'opponent_code'
    ]
    
    
    model.train(df_train, feature_cols=feature_cols, target_col='result')

    # --- 4. EVALUATION ---
    print("\n--- Phase 4: Evaluation ---")
    
    probs = model.predict(df_train, feature_cols=feature_cols)
    df_train['my_prob'] = probs
    
    # Simulate Betting
    results = simulate_betting(df_train, prob_col='my_prob', threshold=0.05)
    
    print(f"\n💰 RESULTS 💰")
    print(f"Bets Placed: {results['total_bets']}")
    print(f"Total Profit: ${results['total_profit']:.2f}")
    print(f"ROI: {results['roi']:.2f}%")
    
    print("\nPipeline Finished Successfully.")

if __name__ == "__main__":
    run_pipeline()
