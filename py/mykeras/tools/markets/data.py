
import tensorflow as tf
import keras
import pandas as pd
import numpy as np


def is_yf_data(data):
    return isinstance(data, dict) and 'history' in data and 'symbol' in data and 'data' in data

def supported_indicators():
    return ['SMA', 'EMA', 'RSI', 'MACD', 'BB']

def get_indicator(name='SMA', source_column='Close', nanvalue=0.0):
    # parse the period from SMA20, EMA50, etc.
    if name.upper().startswith('SMA'):
        w = int(name[3:])
        def sma(data):
            if data[source_column].isnull().values.any():
                raise ValueError("Data contains NaN values")
            # return nanvalue for the first w-1 values
            df  = (data[source_column].rolling(window=w).mean())
            df.iloc[:w-1] = nanvalue
            return df
        return sma, w
    elif name.upper().startswith('EMA'):
        s = int(name[3:])
        def ema(data):
            if data[source_column].isnull().values.any():
                raise ValueError("Data contains NaN values")
            df = data[source_column].ewm(span=s, adjust=False).mean()
            df.iloc[:s-1] = nanvalue
            return df
        return ema, s
    elif name.upper().startswith('RSI'):
        w = int(name[3:])
        def rsi(data):
            if data[source_column].isnull().values.any():
                raise ValueError("Data contains NaN values")
            delta = data[source_column].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=w).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=w).mean()
            rs = gain / loss
            df = 100 - (100 / (1 + rs))
            df.iloc[:w-1] = nanvalue
            return df
        return rsi, w
    elif name.upper().startswith('MACD'):
        [s1, s2] = [int(n) for n in name[4:].split('_')]
        def macd(data):
            if data[source_column].isnull().values.any():
                raise ValueError("Data contains NaN values")
            ema_fast = data[source_column].ewm(span=s1, adjust=False).mean()
            ema_slow = data[source_column].ewm(span=s2, adjust=False).mean()
            df = ema_fast - ema_slow
            inv = max(s1, s2)
            df.iloc[:inv] = nanvalue
            if df.isnull().values.any():
                raise ValueError("Data contains NaN values")
            ema_fast.iloc[:inv] = nanvalue
            ema_slow.iloc[:inv] = nanvalue
            if ema_fast.isnull().values.any():
                raise ValueError("Data contains NaN values")
            if ema_slow.isnull().values.any():
                raise ValueError("Data contains NaN values")
            return ema_fast, ema_slow, df
        return macd, (s1, s2)
    elif name.upper().startswith('BB'):
        w = int(name[2:])
        def bb(data):
            if data[source_column].isnull().values.any():
                raise ValueError("Data contains NaN values")
            sma = data[source_column].rolling(window=w).mean()
            std = data[source_column].rolling(window=w).std()
            upper = sma + 2. * std
            lower = sma - 2. * std
            upper.iloc[:w-1] = nanvalue
            lower.iloc[:w-1] = nanvalue
            return upper, lower
        return bb, w
    else:
        raise ValueError(f"Indicator {name} not implemented.")


def get_dataset(data, source_columns=[], indicators=[]):
    if is_yf_data(data):
        data = data['history']
    assert isinstance(data, pd.DataFrame), f"Data must be a DataFrame, got {type(data)} instead."
    assert len(source_columns) > 0, "At least one source column must be specified."
    assert not data.isnull().values.any(), "Data contains NaN values."
    
    # Select source columns
    data = data[source_columns].copy()
    assert not data.isnull().values.any(), "Data contains NaN values."

    # Add indicators
    for indicator in indicators:
        for source_column in source_columns:
            name = f'{indicator}_{source_column}'
            data.loc[:, name] = 0.
            ind, skip = get_indicator(indicator, source_column)
            if indicator.startswith('BB'):
                upper, lower = ind(data)
                data.loc[:, f'{indicator}U_{source_column}'] = upper[skip:]
                data.loc[:, f'{indicator}L_{source_column}'] = lower[skip:]
                assert not data.isnull().values.any(), "Data contains NaN values."
            elif indicator.startswith('MACD'):
                fast, slow, diff = ind(data)
                skip = max(skip[0], skip[1])
                data.loc[:, f'{indicator}F_{source_column}'] = fast[skip:]
                data.loc[:, f'{indicator}S_{source_column}'] = slow[skip:]
                data.loc[:, f'{indicator}D_{source_column}'] = diff[skip:]
                assert not data.isnull().values.any(), "Data contains NaN values."
            else:
                new_data = ind(data)
                print(f"new_data.shape: {new_data.shape} (orig)")
                # print(f"new_data.shape: {new_data[skip:].shape} (now)")
                dlns = data.iloc[:].loc[:, name].shape
                print(f"data[skip:,name].shape: {dlns}")
                # expand data[name] to match new_data+skip
                data.loc[:, name] = new_data[:]
                # replace NaN values with 0
                # data.loc[:, name] = data.loc[:, name].fillna(0)
                assert not data.isnull().values.any(), "Data contains NaN values."
                
    assert not data.isnull().values.any(), "Data contains NaN values."
    # drop rows with NaN values
    data = data.dropna()
    assert data.shape[0] > 0, "Data is empty after dropna()."
    
    print(f"data.head(): {data.head()}")
    print(f"data.tail(): {data.tail()}")
    data.describe()
    print(f"data.shape: {data.shape}")
    assert not data.isnull().values.any(), "Data contains NaN values."
    return data

class TestGetDataset(tf.test.TestCase):
    def test_get_dataset(self):
        from py.mykeras.datasets.yf import get_ticker_data
        data = get_ticker_data('AAPL', start='1970-01-01')

        data = get_dataset(data, 
                           source_columns=['Open', 'Close'],
                           indicators=['SMA20', 'EMA50', 'RSI14', 'MACD12_26', 'BB72'])
        self.assertIsNotNone(data)
        print(f"data.head(): {data.tail()}")
        print(f"data.shaoe: {data.shape}")
        self.assertEqual(data.shape[1], 4)
        self.assertEqual(data.columns[0], 'Close')
        self.assertEqual(data.columns[1], 'Volume')
        self.assertEqual(data.columns[2], 'SMA20')
        self.assertEqual(data.columns[3], 'EMA50')

if __name__ == '__main__':
    tf.test.main()

#
