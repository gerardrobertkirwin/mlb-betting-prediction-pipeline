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
    
