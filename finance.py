from datetime import datetime
import pyodbc
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings
from math import sqrt
from sklearn.metrics import mean_squared_error

def evaluate_models(dataset, p_values, d_values, q_values):
    for p in p_values:
        for d in d_values:
            for q in q_values:
                model = ARIMA(dataset, order=(p, d, q))
                model_fit = model.fit()
                print(model_fit.summary())

def evaluate_models_rmse(dataset, p_values, d_values, q_values, actuals, start_index, end_index):
    """A FUNCTION THAT WILL EVALUATE THE DIFFERENT PARAMETERS BY ROOT MEAN SQUARED ERROR"""
    for p in p_values:
        for d in d_values:
            for q in q_values:
                model = ARIMA(dataset, order=(p, d, q))
                model_fit = model.fit()
                predictions = model_fit.predict(start=start_index, end=end_index)
                error = actuals - predictions
                mse = np.square(error).mean()
                rmse = sqrt(mse)

                print(f"RMSE for Model ({p}, {d}, {q}): {rmse}")