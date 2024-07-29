
import numpy as np
import pandas as pd
import tensorflow as tf
import pymc as pm
from torch import tensor as tt
import altair as alt
from datetime import datetime

from plotting import *

class TimeSeriesModel:
    
    def fit(self, t, y):
        self.model = pm.Model()
        prediction = self.definition(self.model, t, y)
        with self.model:
            error = pm.HalfCauchy('error', 0.5)
            pm.Normal('obs', mu=prediction, sigma=error, observed=y)
            self.trace = pm.sample(tune=200, draws=200, cores=4, chains=4)

        # self.t = t
        # self.y = y
        # self.model = pm.Model()
        # with self.model:
        #     self.alpha = pm.Normal('alpha', mu=0, sigma=1)
        #     self.beta = pm.Normal('beta', mu=0, sigma=1)
        #     self.sigma = pm.HalfNormal('sigma', sigma=1)
        #     self.y_pred = pm.Normal('y_pred', mu=self.alpha + self.beta * self.t, sigma=self.sigma, observed=self.y)
        #     self.trace = pm.sample(1000, tune=1000)
        # return self
        
    def __add__(self, other):
        return AdditiveTimeSeries(self, other)
    
    def __mul__(self, other):
        return MultiplicativeTimeSeries(self, other)

class FourierSeasonality(TimeSeriesModel):
    def __init__(self, p, n):
        self.p = p
        self.n = n
    
    def X(self, t, p=365.25, n=3):
        x = 2 * np.pi * (np.arange(n) + 1) * t[:, None] / p
        return np.concatenate([np.sin(x), np.cos(x)], axis=1)
    
    def definition(self, model, t, y):
        x = self.X(t, p=self.p, n=self.n)
        with model:
            beta = pm.Normal('beta', mu=0, sigma=1, shape=2 * self.n)
            seasonality = pm.Deterministic('seasonality', tt.dot(self.X(t, self.p/len(t)), beta))
        return seasonality

class LinearTrend(TimeSeriesModel):
    def __init__(self, n_changepoints):
        self.n_changepoints = n_changepoints

    def definition(self, model, t, y):
        s = np.linspace(0, np.max(t), self.n_changepoints + 2)[1:-1]
        A = (t[:, None] > s) * 1.
        with model:
            # define priors
            k = pm.Normal('k', mu=0, sigma=1)    
            m = pm.Normal('m', mu=0, sigma=5)
            delta = pm.Laplace('delta', 0, 0.1, shape=self.n_changepoints)
            # calculate trend
            trend = (k + A.dot(delta)) * t[:, None] + m * A.dot(-s * delta)
            
            error = pm.HalfCauchy('error', 0.5)
            pm.Normal('obs', mu=trend, sigma=error, observed=y)
            trace = pm.sample(tune=200, draws=200, cores=4, chains=4)
            
        return trend
    
class AdditiveTimeSeries(TimeSeriesModel):
    def __init__(self, l, r):
        self.l = l
        self.r = r

    def definition(self, model, t, y):
        return (
            self.l.definition(model, t, y) +
            self.r.definition(model, t, y)
        )

class MultiplicativeTimeSeries(TimeSeriesModel):
    def __init__(self, l, r):
        self.l = l
        self.r = r

    def definition(self, model, t, y):
        return (
            self.l.definition(model, t, y) *
            self.r.definition(model, t, y)
        )

def test_models():
    df = (
        
        # pd.DataFrame(np.reshape(pd.read_html('https://www.tradingview.com/symbols/NASDAQ-META/'), (3, 1)))
        # .assign(date=lambda x: pd.to_datetime(x.date))
        # .groupby(['date'])
        # .iloc[0,2]
    )

    # print(df.iloc[0, 2])
    print(df.head())

    # t = np.linspace(0, 10, 100)
    # y = 2 * t + 1 + np.random.normal(0, 1, len(t))
    # model = LinearTrend(n_changepoints=10) + FourierSeasonality(n=3, p=365.25)
    # model.fit(t, y)
    # print(model.trace['seasonality'].shape)
    # print(model.trace['seasonality'].mean(axis=0))
    # print(model.trace['seasonality'].std(axis=0))
    # print(model.trace['seasonality'].quantile(0.025, axis=0))
    # print(model.trace['seasonality'].quantile(0.975, axis=0))
    # return model

class TimeSeriesTesting(tf.test.TestCase):
    
    # def test_something():
    #     test_models()
        
    def test_yf_aapl(self):
        from ..datasets.yf import get_ticker_data
        data = get_ticker_data('AAPL', verbose=1)
        
        pd.read_pickle

        # Check if df is a numpy array and ensure it is 2-dimensional
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                df = pd.DataFrame(data)
            else:
                print(f"Unexpected shape of numpy array: {df.shape}")
                raise ValueError("Expected 2-dimensional array")
        
        # Print the columns of the dataframe
        print("Columns in df:", df.columns)
        
        # Print the first few rows of the dataframe
        print("First few rows of df:\n", df.head())
        
        # Attempt to print the 'symbol' column if it exists
        if 'symbol' in df.columns:
            print(df['symbol'])
        else:
            print("Column 'symbol' does not exist in df")

if __name__ == '__main__':
    tf.test.main()
    # test_models()
