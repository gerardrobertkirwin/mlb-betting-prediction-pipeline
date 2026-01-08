import pandas as pd

def create_team_centric_df(df_master: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms 1 Game Row (Home vs Away) into 2 Team Rows (Team vs Opponent).
    This allows us to calculate 'Recent Form' for every team easily.
    """
    # 1. Home stats
    df_home = df_master.copy()
    df_home = df_home.rename(columns={
        'home_team_abbr': 'team',
        'away_team_abbr': 'opponent',
        'home_score': 'runs_scored',
        'away_score': 'runs_allowed',
        'home_hits': 'hits',
        'home_errors': 'errors',
        'away_hits': 'opp_hits',
        'away_errors': 'opp_errors',
        'home_moneyline': 'moneyline_closing',
        'away_moneyline': 'moneyline_opp' # We keep opponent odds for reference
    })
    df_home['is_home'] = 1
    df_home['result'] = (df_home['runs_scored'] > df_home['runs_allowed']).astype(int)

    # 2. Away stats
    df_away = df_master.copy()
    df_away = df_away.rename(columns={
        'away_team_abbr': 'team',
        'home_team_abbr': 'opponent',
        'away_score': 'runs_scored',
        'home_score': 'runs_allowed',
        'away_hits': 'hits',
        'away_errors': 'errors',
        'home_hits': 'opp_hits',
        'home_errors': 'opp_errors',
        'away_moneyline': 'moneyline_closing',
        'home_moneyline': 'moneyline_opp'
    })
    df_away['is_home'] = 0
    df_away['result'] = (df_away['runs_scored'] > df_away['runs_allowed']).astype(int)

    # 3. Concatenate and sort
    
    cols_to_keep = [
        'date', 'team', 'opponent', 'is_home', 'result',
        'runs_scored', 'runs_allowed', 'hits', 'errors',
        'moneyline_closing'
    ]

    df_long = pd.concat([df_home[cols_to_keep], df_away[cols_to_keep]])

    df_long = df_long.sort_values(['team', 'date']).reset_index(drop=True)

    return df_long


def calculate_rolling_features(df_long, window_size=10):
    """
    Calculates rolling averages for the last N games.
    """
    df_features = df_long.copy()

    # Metrics we want to average
    metrics = ['runs_scored', 'runs_allowed', 'hits', 'errors']

    for metric in metrics:
        col_name = f'rolling_{window_size}_{metric}'

        df_features[col_name] = (
            df_features.groupby('team')[metric]
            .transform(lambda x: x.shift(1).rolling(window=window_size).mean())
        )

    # Pythagorean Expectation
    rolling_runs = df_features.groupby('team')['runs_scored'].transform(lambda x: x.shift(1).rolling(window_size).sum())
    rolling_allowed = df_features.groupby('team')['runs_allowed'].transform(lambda x: x.shift(1).rolling(window_size).sum())

    df_features['rolling_pythag_win_pct'] = (rolling_runs ** 2) / ((rolling_runs ** 2) + (rolling_allowed ** 2) + 1e-9)

    return df_features

def calculate_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates rest days and Log5 probability
    """
    
    df_adv = df.copy()

    # Rest Days
    # If played yesterday (May 2 - May 1) = 1 day.
    df_adv['rest_days'] = df_adv.groupby('team')['date'].diff().dt.days
    # Fill the first game of the season with huge rest (e.g., 5 days)
    df_adv['rest_days'] = df_adv['rest_days'].fillna(5)

    #Create a small lookup table
    opponent_stats = df_adv[['date', 'team', 'rolling_pythag_win_pct']].copy()
    opponent_stats = opponent_stats.rename(columns={
        'team': 'opponent', # Rename so we can join on this
        'rolling_pythag_win_pct': 'opp_pythag_win_pct'
    })

    df_adv = pd.merge(
        df_adv,
        opponent_stats,
        on=['date', 'opponent'],
        how='left'
    )

    # Log5
    A = df_adv['rolling_pythag_win_pct']
    B = df_adv['opp_pythag_win_pct']

    # We use a safe division formula
    numerator = A * (1 - B)
    denominator = (A * (1 - B)) + (B * (1 - A))

    df_adv['log5_prob'] = numerator / denominator

    return df_adv

def finalize_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops early season NaNs (pre game 10) and encodes team IDs
    """
    df_train = df.copy()

    # 1. Drop the Warm-Up Period
    df_train = df_train.dropna(subset=['rolling_10_runs_scored', 'log5_prob'])
    
    # 2. Encode Team Names for ML process
    df_train['team_code'] = df_train['team'].astype('category').cat.codes
    df_train['opponent_code'] = df_train['opponent'].astype('category').cat.codes

    # 3. Final Feature Selection
    features = [
        'is_home',
        'rest_days',
        'log5_prob',
        'rolling_10_runs_scored',
        'rolling_10_runs_allowed',
        'rolling_10_hits',
        'rolling_10_errors',
        'rolling_pythag_win_pct',
        'opp_pythag_win_pct',
        'team_code',
        'opponent_code',
        'moneyline_closing' # We don't train on this, but we need it for the betting simulation later
    ]

    # The Target
    target = 'result' # 1 = Win, 0 = Loss

    return df_train[features + [target, 'date', 'team', 'opponent']]
