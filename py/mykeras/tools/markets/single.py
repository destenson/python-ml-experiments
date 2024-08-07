
import numpy as np
import tensorflow as tf
import keras
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Input, Reshape, Flatten, BatchNormalization
from py.mykeras.datasets.yf.yf import yahoofinance_data
from py.mykeras.tools.markets.data import get_dataset
import pandas as pd

def get_preprocessed_data(symbol='AAPL', window_size=20, step_size=1):
    data = yahoofinance_data(symbol, verbose=True)[()]
    
    print(f"Data: {len(data)}")
    # print(data)
    
    # examine data
    if isinstance(data, dict):
        print(data.keys())
    print(data['data'].keys())
    print(data['history'].keys())
    print(data['history'].head())
    print(data['history'].tail())

    data = data['history']
    # remove unwanted columns
    data = data.drop(columns=['Dividends', 'Stock Splits', 'Volume'])

    data = get_dataset(data, source_columns=['Open', 'Close'],
                       indicators=['SMA20', 'SMA72', 'EMA50', 'RSI14', 'MACD12_26', 'BB72'])
    # data = data[!["Dividends", "Stock Splits", 'Volume']]
    data.describe()

    windowed_data = []
    for start in range(0, len(data)-window_size+1-step_size, step_size):
        end = start + window_size
        window = data.iloc[start:end]
        # print(f"window: {window}")
        if len(window) != len(windowed_data[-1]) if windowed_data else 0:
            continue
        windowed_data.append(window)
    
    windowed_data = np.array(windowed_data)

    # data = data[['Close', 'Volume']]
    # print("Splits:")
    # for i in range(len(data['data']['splits'])):
    #     print(data['data']['splits'].index[i], data['data']['splits'].iloc[i])

    # print("Dividends:")
    # for i in range(len(data['data']['dividends'])):
    #     print(data['data']['dividends'].index[i], data['data']['dividends'].iloc[i])
    
    # df = (tf.data.Dataset.from_tensor_slices(data)
    #       .rename(columns={'Close': 'close', 'Volume': 'volume'})
    # )

    return data, windowed_data

def desplit_prices(data):
    def unadjust_price(adjusted_price, price_date, splits):
        unadjusted_price = adjusted_price
        for _, split in splits.iterrows():
            if split['date'] > price_date:
                unadjusted_price *= split['ratio']
        return unadjusted_price

    data = data['history']
    # data = data.drop(columns=['Dividends', 'Stock Splits', 'Volume'])
    # use the stock splits to adjust the prices
    # for i in range(len(data['splits'])):
    #     split = data['splits'].iloc[i]
    #     date = data['splits'].index[i]
    #     data['Close'] = data['Close'].apply(lambda x: x/split if x > split else x)
    #     data['Open'] = data['Open'].apply(lambda x: x/split if x > split else x)
    #     data['High'] = data['High'].apply(lambda x: x/split if x > split else x)
    #     data['Low'] = data['Low'].apply(lambda x: x/split if x > split else x)
        
    data = data[['Close', 'Volume']]
    return data

def learn_aapl():
    _data, data = get_preprocessed_data('AAPL')
    data = data[()]
    # Prepare data
    # X = data[:-1]
    # y = data[1:]
    print(f"data.shape: {data.shape}")
    data_shape = (data.shape[-2], data.shape[-1],)
    print(f"data_shape: {data_shape}")
    data =  tf.data.Dataset.from_tensor_slices(data).batch(20).prefetch(tf.data.AUTOTUNE)

    inputs = Input(shape=data_shape)
    x = BatchNormalization()(inputs)
    x = Dense(np.prod(data_shape), activation='sigmoid')(x)
    x = LSTM(64, activation='leaky_relu')(x)
    # x = Flatten()(x)
    x = Dense(np.prod(data_shape), activation='relu')(x)
    outputs = Reshape(data_shape)(x)
    # LSTM model
    # model = Sequential([
    #     Input(shape=data_shape),
    #     LSTM(50),
    #     Dense(np.sum(data_shape)),
    #     Reshape(data_shape),
    # ])
    model = keras.api.models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()

    n = 0
    # Training loop
    for epoch in range(10):
        for d in data.as_numpy_iterator():
            # print(f"d: {d}")
            X = d[:-1]
            y = d[1:]
            verbose = 1 if n % 10 == 0 else 0
            if len(X) == 0 or len(y) == 0:
                print("len(X) or len(y) == 0")
                break
            model.fit(X, y, epochs=1, initial_epoch=epoch, verbose=verbose, validation_split=0.6)
            n = n + 1
        
    print("Training complete. (n={n})")
    return model


if __name__ == '__main__':
    # import os
    # import sys
    # sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    # get_preprocessed_data('AAPL')
    model = learn_aapl()
    model.summary()
