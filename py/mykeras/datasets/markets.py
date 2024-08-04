

import tensorflow as tf
import keras
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, InputLayer, Reshape
import pandas as pd
import numpy as np

# get all data in pickle files from data path as a tf.data.Dataset
def get_all_data(data_path='data/'):
    # import os
    data = tf.data.Dataset.list_files(data_path+'*.pickle')
    data = data.map(lambda x: np.array(pd.read_pickle(x)))
    # .map(lambda x: tf.data.Dataset.from_tensor_slices(np.array(pd.read_pickle(x))))
    # .prefetch(tf.data.AUTOTUNE)
    # # data = {}
    # for file in os.listdir(data_path):
    #     if file.endswith('.pickle'):
    #         symbol = file.split('_')[0]
    #         data[symbol] = pd.read_pickle(data_path+file)
    return data

def get_data(symbol, start_date, end_date, features=['Close', 'Volume'], period='1d', source=None):
    data = None
    if source is None or source == 'yf':
        from py.mykeras.datasets.yf import get_ticker_data
        data = get_ticker_data(symbol, start=start_date, end=end_date, interval=period)


    assert data is not None, f"Data source {source} not implemented yet."
    assert data['symbol'] == symbol, f"Data source {source} returned wrong symbol {data['symbol']} instead of {symbol}."
    # assert data['history'][()] == start_date, f"Data source {source} returned wrong start date {data.index[0].date()} instead of {start_date}."

    print(data['history'].keys())    
    # select features
    data = data['history'][features]
    print(f'data.head: {data.head()}')
    print(f'data.tail: {data.tail()}')
    
    return data

class TestGetData(tf.test.TestCase):
    def test0(self):
        data = get_data('AAPL', '2023-01-01', '2023-12-31')
        self.assertIsNotNone(data)
        self.assertEqual(data.shape[1], 2)
        
    def test_get_all_data(self):
        data = get_all_data()
        self.assertIsNotNone(data)
        # self.assertEqual(data.shape[1], 2)

 
if __name__ == '__main__':
    tf.test.main()
