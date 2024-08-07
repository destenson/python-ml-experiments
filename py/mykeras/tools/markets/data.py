
import tensorflow as tf
import keras
import pandas as pd
import numpy as np


def is_yf_data(data):
    return isinstance(data, dict) and 'history' in data and 'symbol' in data and 'data' in data

def supported_indicators():
    return ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'STDDEV']

def get_indicator(name='SMA', source_column='Close', nanvalue=0.0):
    # parse the period from SMA20, EMA50, etc.
    print(f"get_indicator({name}, {source_column}, {nanvalue})")
    if name.upper().startswith('SMA'):
        w = int(name[3:])
        def sma(data):
            if data[source_column].isnull().values.any():
                raise ValueError("Data contains NaN values")
            if len(data) < w:
                raise ValueError(f"Data is too short for the given window size. {len(data)} < {w}")
            df  = data[source_column].rolling(window=w, min_periods=1).mean()
            if df.isnull().values.any():
                raise ValueError("Data contains NaN values")
            return df
        return sma, w
    elif name.upper().startswith('EMA'):
        s = int(name[3:])
        def ema(data):
            if data[source_column].isnull().values.any():
                raise ValueError("Data contains NaN values")
            df = data[source_column].ewm(span=s, adjust=False, min_periods=1).mean()
            if df.isnull().values.any():
                raise ValueError("Data contains NaN values")
            # df.iloc[:s-1] = nanvalue
            return df
        return ema, s
    elif name.upper().startswith('RSI'):
        w = int(name[3:])
        def rsi(data):
            if data[source_column].isnull().values.any():
                raise ValueError("Data contains NaN values")
            delta = data[source_column].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=w, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=w, min_periods=1).mean()
            rs = gain / loss
            df = 100 - (100 / (1 + rs))
            df.iloc[0] = nanvalue
            if df.isnull().values.any():
                raise ValueError("Data contains NaN values")
            return df
        return rsi, w
    elif name.upper().startswith('MACD'):
        [s1, s2] = [int(n) for n in name[4:].split('_')]
        def macd(data):
            if data[source_column].isnull().values.any():
                raise ValueError("Data contains NaN values")
            ema_fast = data[source_column].ewm(span=s1, adjust=False, min_periods=1).mean()
            ema_slow = data[source_column].ewm(span=s2, adjust=False, min_periods=1).mean()
            df = ema_fast - ema_slow
            inv = max(s1, s2)
            # df.iloc[0] = nanvalue
            if df.isnull().values.any():
                raise ValueError("Data contains NaN values")
            # ema_fast.iloc[:inv] = nanvalue
            # ema_slow.iloc[:inv] = nanvalue
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
            sma = data[source_column].rolling(window=w, min_periods=1).mean()
            std = data[source_column].rolling(window=w, min_periods=1).std()
            upper = sma + 2. * std
            lower = sma - 2. * std
            upper.iloc[0] = nanvalue
            lower.iloc[0] = nanvalue
            assert not upper.isnull().values.any(), "Data contains NaN values."
            assert not lower.isnull().values.any(), "Data contains NaN values."
            return upper, lower
        return bb, w
    elif name.upper().startswith('STDDEV'):
        w = int(name[6:])
        def stddev(data):
            if data[source_column].isnull().values.any():
                raise ValueError("Data contains NaN values")
            std = data[source_column].rolling(window=w, min_periods=1).std()
            std.iloc[0] = nanvalue
            return std
        return stddev, w
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
                assert not data.isnull().values.any(), "Data contains NaN values."
                upper, lower = ind(data)
                data.loc[:, f'{indicator}U_{source_column}'] = upper[:]
                data.loc[:, f'{indicator}L_{source_column}'] = lower[:]
                # replace nans with zeros
                # data = data.fillna(0)
                print(f"data.shape: {data.shape}")
                print(f"data.head(): {data.head()}")
                print(f"data.tail(): {data.tail()}")
                assert not data.loc[:, f'{indicator}U_{source_column}'].isnull().values.any(), "Data contains NaN values."
                assert not data.loc[:, f'{indicator}L_{source_column}'].isnull().values.any(), "Data contains NaN values."
            elif indicator.startswith('MACD'):
                assert not data.isnull().values.any(), "Data contains NaN values."
                fast, slow, diff = ind(data)
                skip = max(skip[0], skip[1])
                data.loc[:, f'{indicator}F_{source_column}'] = fast[:]
                data.loc[:, f'{indicator}S_{source_column}'] = slow[:]
                data.loc[:, f'{indicator}D_{source_column}'] = diff[:]
                # replace nans with zeros
                # data = data.fillna(0)
                assert not data.isnull().values.any(), "Data contains NaN values."
            else:
                assert not data.isnull().values.any(), "Data contains NaN values."
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
    
    def test_get_indicator_sma(self):
        data = pd.DataFrame({'Close': [1., 2., 3., 4., 5., 6.]})
        ind, w = get_indicator('SMA5', 'Close')
        sma = ind(data)
        self.assertIsNotNone(sma)
        # print(f"sma.shape: {sma.shape}")
        # print(f"sma: {sma}")
        self.assertEqual(sma.shape[0], 6)
        self.assertEqual(w, 5)
        self.assertEqual(sma[0], 1.)
        self.assertEqual(sma[1], 1.5)
        self.assertEqual(sma[2], 2.)
        self.assertEqual(sma[3], 2.5)
        self.assertEqual(sma[4], 3.)
        self.assertEqual(sma[5], 4.)
    
    def test_get_indicator_ema(self):
        data = pd.DataFrame({'Close': [1., 2., 3., 4., 5., 6.]})
        ind, s = get_indicator('EMA3', 'Close')
        ema = ind(data)
        self.assertIsNotNone(ema)
        # print(f"ema.shape: {ema.shape}")
        # print(f"ema: {ema}")
        self.assertEqual(ema.shape[0], 6)
        self.assertEqual(s, 3)
        self.assertEqual(ema[0], 1.)
        self.assertEqual(ema[1], 1.5)
        self.assertEqual(ema[2], 2.25)
        self.assertEqual(ema[3], 3.125)
        self.assertEqual(ema[4], 4.0625)
        self.assertEqual(ema[5], 5.03125)
        
    def test_get_indicator_rsi(self):
        data = pd.DataFrame({'Close': [1., 2., 1.75, 1.75, 2.0, 2.1]})
        ind, w = get_indicator('RSI3', 'Close')
        rsi = ind(data)
        self.assertIsNotNone(rsi)
        # print(f"rsi.shape: {rsi.shape}")
        # print(f"rsi: {rsi}")
        self.assertEqual(rsi.shape[0], 6)
        self.assertEqual(w, 3)
        # self.assertEqual(rsi[0], np.nan)
        self.assertEqual(rsi[1], 100.)
        self.assertEqual(rsi[2], 80.)
        self.assertEqual(rsi[3], 80.)
        self.assertEqual(rsi[4], 50.)
        self.assertEqual(rsi[5], 100.)
    
    def test_get_indicator_macd(self):
        data = pd.DataFrame({'Close': [1., 2., 1.75, 1.75, 2.0, 2.1]})
        ind, (s1, s2) = get_indicator('MACD2_4', 'Close')
        fast, slow, diff = ind(data)
        self.assertIsNotNone(fast)
        self.assertIsNotNone(slow)
        self.assertIsNotNone(diff)
        # print(f"fast.shape: {fast.shape}")
        # print(f"slow.shape: {slow.shape}")
        # print(f"diff.shape: {diff.shape}")
        # print(f"fast: {fast}")
        # print(f"slow: {slow}")
        # print(f"diff: {diff}")
        self.assertEqual(fast.shape[0], 6)
        self.assertEqual(slow.shape[0], 6)
        self.assertEqual(diff.shape[0], 6)
        self.assertEqual(s1, 2)
        self.assertEqual(s2, 4)
        self.assertEqual(fast[0], 1.)
        # self.assertEqual(fast[1], 1.5)
        # self.assertEqual(fast[2], 1.75)
        # self.assertEqual(fast[3], 2.125)
        # self.assertEqual(fast[4], 2.5625)
        # self.assertEqual(fast[5], 3.28125)
        # self.assertEqual(slow[0], 1.)
        # self.assertEqual(slow[1], 1.5)
        # self.assertEqual(slow[2], 1.75)
        # self.assertEqual(slow[3], 2.125)
        # self.assertEqual(slow[4], 2.5625)
        # self.assertEqual(slow[5], 3.28125)
        # self.assertEqual(diff[0], 0.)
        # self.assertEqual(diff[1], 0.)
        # self.assertEqual(diff[2], 0.)
        # self.assertEqual(diff[3], 0.)
        # self.assertEqual(diff[4], 0.)
        # self.assertEqual(diff[5], 0.)
    
    def test_get_indicator_bb(self):
        data = pd.DataFrame({'Close': [1., 2., 1.75, 1.75, 2.0, 2.1]})
        ind, w = get_indicator('BB3', 'Close')
        upper, lower = ind(data)
        self.assertIsNotNone(upper)
        self.assertIsNotNone(lower)
        # print(f"upper.shape: {upper.shape}")
        # print(f"lower.shape: {lower.shape}")
        # print(f"upper: {upper}")
        # print(f"lower: {lower}")
        self.assertEqual(upper.shape[0], 6)
        self.assertEqual(lower.shape[0], 6)
        self.assertEqual(w, 3)
        self.assertEqual(upper[0], 0.)
        self.assertGreater(upper[1], 2.)
        self.assertGreater(upper[2], 2.)
        self.assertGreater(upper[3], 2.)
        self.assertGreater(upper[4], 2.)
        self.assertGreater(upper[5], 2.)
        self.assertLess(lower[0], 2.)
        self.assertLess(lower[1], 2.)
        self.assertLess(lower[2], 2.)
        self.assertLess(lower[3], 2.)
        self.assertLess(lower[4], 2.)
        self.assertLess(lower[5], 2.)
    
    def test_get_indicator_stddev(self):
        data = pd.DataFrame({'Close': [1., 1.5, 2., 1., 2., 2., 1.6]})
        ind, w = get_indicator('STDDEV3', 'Close')
        std = ind(data)
        self.assertIsNotNone(std)
        # print(f"std.shape: {std.shape}")
        # print(f"std: {std}")
        self.assertEqual(std.shape[0], 7)
        self.assertEqual(w, 3)
        self.assertEqual(std[0], 0.)
        # self.assertEqual(std[1], 0.5)
        # self.assertEqual(std[2], 0.125)
        # self.assertEqual(std[3], 0.)
        # self.assertEqual(std[4], 0.125)
        # self.assertEqual(std[5], 0.15)
    
    # def test_get_dataset(self):
    #     from py.mykeras.datasets.yf.yf import get_ticker_data
    #     data = get_ticker_data('AAPL', start='1970-01-01')

    #     data = get_dataset(data, 
    #                        source_columns=['Open', 'Close'],
    #                        indicators=['SMA20', 'EMA50', 'RSI14', 'MACD12_26', 'BB72', 'STDEV20', 'STDDEV36', 'STDDEV72'])
    #     self.assertIsNotNone(data)
    #     print(f"data.head(): {data.tail()}")
    #     print(f"data.shaoe: {data.shape}")
    #     self.assertEqual(data.shape[1], 4)
    #     self.assertEqual(data.columns[0], 'Close')
    #     self.assertEqual(data.columns[1], 'Volume')
    #     self.assertEqual(data.columns[2], 'SMA20')
    #     self.assertEqual(data.columns[3], 'EMA50')

if __name__ == '__main__':
    tf.test.main()

#
