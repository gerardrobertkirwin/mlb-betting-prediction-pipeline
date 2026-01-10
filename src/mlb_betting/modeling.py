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
