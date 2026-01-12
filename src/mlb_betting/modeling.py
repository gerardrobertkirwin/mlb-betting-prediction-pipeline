import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

class BayesianBettingModel:
  def __init__(self, model_path: str = "data/models/bayesian_model_v1.nc"):
    self.model_path = model_path
    self.trace = None

    self.scaler = StandardScaler()
    self.imputer = SimpleImputer(strategy='mean')
  
  def train(self, df_train: pd.DataFrame, feature_cols: list, target_col: str = 'result'):
    """
    Trains the Bayesian model using PyMC and saves to disc
    """

    #1. Prepare Data
    X = df_train[feature_cols].values
    y = df_train[target_col].values
    
    # 2. Manual Cleaning Pipeline (Impute -> Scale)
    X_imputed = self.imputer.fit_transform(X)
    X_scaled = self.scaler.fit_transform(X_imputed)

    # 2. Define the Model Context
    with pm.Model() as bayesian_model:
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        betas = pm.Normal("betas", mu=0, sigma=1, shape=X_scaled.shape[1])

        mu = alpha + pm.math.dot(X_scaled, betas)
        theta = pm.math.sigmoid(mu)
    
        y_obs = pm.Bernoulli("y_obs", p=theta, observed=y)

        print(f"Sampling (please wait...)")
        self.trace = pm.sample(1000, tune=1000, chains=2, return_inferencedata=True)

    os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
    az.to_netcdf(self.trace, self.model_path)
    print(f"Model saved to {self.model_path}")

  def predict(self, df_new: pd.DataFrame, feature_cols: list) -> np.array:
    """
    Loads the model and generates probability predictions for new data.
    """
    # Load Model if needed
    if self.trace is None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        print(f"Loading model from {self.model_path}...")
        self.trace = az.from_netcdf(self.model_path)

    X = df_new[feature_cols].values
    
    X_imputed = self.imputer.transform(X)
    X_scaled = self.scaler.transform(X_imputed)

    print("Generating Probabilities...")

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        betas = pm.Normal("betas", mu=0, sigma=1, shape=X_scaled.shape[1])

        mu = alpha + pm.math.dot(X_scaled, betas)
        theta = pm.math.sigmoid(mu)
    
        y_obs = pm.Bernoulli("y_obs", p=theta, shape=len(X_scaled))

        ppc = pm.sample_posterior_predictive(self.trace, var_names=["y_obs"])
    
    bayesian_probs = ppc.posterior_predictive['y_obs'].mean(dim=["chain", "draw"]).values


def simulate_betting(df, threshold=0.05, stake=100):
    """
    Simulates betting $100 whenever our model sees an edge > 5%.
    """
    sim = df.copy()

    # 1. Calculate probability from odds
    def us_odds_to_prob(odds):
        if pd.isna(odds): return np.nan
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return (-odds) / (-odds + 100)

    sim['vegas_prob'] = sim['moneyline_closing'].apply(us_odds_to_prob)

    # 2. Find the edge
    sim['edge'] = sim['my_prob'] - sim['vegas_prob']

    # 3. Place bets
    sim['bet_placed'] = sim['edge'] > threshold

    #4. Calculate profit and loss
    def calculate_pnl(row):
        if not row['bet_placed']: return 0

        # Convert US odds to Decimal multiplier for payout
        if row['moneyline_closing'] > 0:
            decimal_odds = 1 + (row['moneyline_closing'] / 100)
        else:
            decimal_odds = 1 + (100 / -row['moneyline_closing'])

        if row['result'] == 1:
            return stake * (decimal_odds - 1)
        else:
            return -stake
 
      
    sim['pnl'] = sim.apply(calculate_pnl, axis=1)
    
    # METRICS
    total_bets = sim['bet_placed'].sum()
    total_profit = sim['pnl'].sum()
    roi = (total_profit / (total_bets * stake)) * 100 if total_bets > 0 else 0
    
    return {
    "total_bets": int(total_bets),
    "total_profit": float(total_profit),
    "roi": float(roi)
    }
    
