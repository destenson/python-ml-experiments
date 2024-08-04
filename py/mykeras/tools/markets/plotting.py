
import tensorflow as tf
# import keras
import numpy as np
from keras.api.layers import Dense, Dropout, Input
from keras.api.layers import Conv1D, MaxPooling1D, Flatten

import matplotlib.pyplot as plt 
import pandas as pd 
import matplotlib.dates as mpl_dates 
import datetime 
from mpl_finance import candlestick_ohlc 

import os

from py.mykeras.tools.video import create_video

def plot_first_market(path='data/'):

    # load the first .pickle file in `path`
    for f in os.listdir(path):
        if f.endswith('.pickle'):
            print(f"Found file in cache: {f}") #if verbose > 0 else None
            loaded_filename = f'{path}{f}'
            df = pd.read_pickle(loaded_filename)
            break

    if df is None:
        raise ValueError(f"No .pickle files found in {path}")

    return plot_history(df['history'], path)

def plot_history(history, path=None):
    print(history.head())
    print(history.tail())
    print(history.shape)
    print(history.info())
    print(history.describe())
    fig, ax = plt.subplots()
    close: pd.Series = history['Close']
    date: pd.DatetimeIndex = history.index
    print(f"close = {type(close)}")
    print(f"date = {type(date)}")
    # draw close and date
    ax.plot(date, close)
    # fig.draw([close, date])
    if path is not None:
        plt.savefig(f'{path}stock_prices.png')
        # import os
        # plt.savefig(os.path.join(path, 'stock_prices.png'))
    plt.show()
    return tf.convert_to_tensor(fig.get_rasterized())    

def plot_candlesticks(history, i=0, bars=5, show_plot=True, use_pyplot=True):
    # Defining a dataframe showing stock prices of a week
    date = history.index[history.index.weekday < 5]
    open = history['Open'][history.index.weekday < 5]
    high = history['High'][history.index.weekday < 5]
    low = history['Low'][history.index.weekday < 5]
    close = history['Close'][history.index.weekday < 5]
    volume = history['Volume'][history.index.weekday < 5]

    stock_prices = pd.DataFrame({'date':date, 'open':open, 'high':high,
                                 'low':low, 'close':close,'volume':volume})

    ohlc = stock_prices.loc[:, ['date', 'open', 'high', 'low', 'close', 'volume']]
    ohlc['date'] = pd.to_datetime(ohlc['date'])
    ohlc['date'] = ohlc['date'].apply(mpl_dates.date2num)
    ohlc = ohlc.astype(float)
    
    if use_pyplot:
        return plot_candlesticks_with_matplotlib(ohlc, i=i, bars=bars, show_plot=show_plot)
    else:
        return plot_candlesticks_with_finplot(ohlc, i=i, bars=bars, show_plot=show_plot)

def plot_candlesticks_with_finplot(ohlc, i=0, bars=5, show_plot=True):
    import finplot as fplt
    ax = fplt.create_plot('Stock Prices', maximize=False)
    fplt.candle_bull_color = fplt.candle_bear_color = fplt.candle_bear_body_color = '#000'
    fplt.volume_bull_color = fplt.volume_bear_color = '#333'
    fplt.candle_bull_body_color = fplt.volume_bull_body_color = '#fff'
    
    # make ochl from ohlc
    ochl = ohlc[['date','open', 'close', 'high', 'low']]
    
    fplt.candlestick_ochl(ochl.values[i:i+bars])
    
    axo = ax.overlay()
    # fplt.volume_ocv(ochl.values[i:i+bars], ax=axo)
    # fplt.plot('Volume', ochl.values[i:i+bars], ax=axo, color=1)
    if show_plot:
        fplt.show()
    else:
        fplt.close()
    
    # convert image to tensor
    image = np.frombuffer(fplt.canvas.tostring_rgb(), dtype=np.uint8)
    print(f"image.shape = {image.shape}")
    shape = (0,
        int(fig.get_dpi()*fig.get_figwidth()), 4)
    shape = (image.shape[0] // shape[1] // shape[2],
             shape[1], shape[2]) 
    image = image.reshape(shape)# / 255. - 0.5
    # image = tf.data.Dataset.from_tensor_slices(image).prefetch(tf.data.AUTOTUNE)
    return image

def plot_candlesticks_with_matplotlib(ohlc, i=0, bars=5, show_plot=True):
    
    # Creating Subplots
    fig, ax = plt.subplots() 

    candlestick_ohlc(ax, ohlc.values[i:i+bars], width=0.6, colorup='blue', 
                    colordown='green', alpha=0.4) 

    # Setting labels & titles
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    fig.suptitle(f'Stock Prices of a {"week" if bars == 5 else f"{bars} days"}', y=0.95, fontsize=16)

    # Formatting Date (but don't plot weekends or days with no data)    
    date_format = mpl_dates.DateFormatter('%d-%m-%Y', usetex=True)
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    fig.tight_layout()

    plt.show()
        
    # convert image to tensor
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    if not show_plot:
        plt.close()
    print(f"image.shape = {image.shape}")
    shape = (0,
        int(fig.get_dpi()*fig.get_figwidth()), 4)
    shape = (image.shape[0] // shape[1] // shape[2],
             shape[1], shape[2]) 
    image = image.reshape(shape)# / 255. - 0.5
    # image = tf.data.Dataset.from_tensor_slices(image).prefetch(tf.data.AUTOTUNE)
    return image

def plot_candlesticks_movie(history, i=0, bars=5, frames=10, fps=2.2, filename='output.mp4', use_pyplot=False):
    images = np.array([
        plot_candlesticks(history, i=i+n, bars=bars, show_plot=True, use_pyplot=use_pyplot)
        for n in range(frames)
    ])#[:,:,:,:3]
    # duplicate the last frame to make the video end on a still frame
    images = np.concatenate([images, np.expand_dims(images[-1], axis=0)])
    # create video
    create_video(images, filename=filename, fps=fps)

# # not really working
# def get_market_history_generator(symbol='AAPL', path='data/', N=5, infinite=False):
#     import os
#     # load the symbol's .pickle file from `path`
#     for f in os.listdir(path):
#         if f.startswith(symbol) and f.endswith('.pickle'):
#             print(f"Found file in cache: {f}") #if verbose > 0 else None
#             loaded_filename = f'{path}{f}'
#             df = pd.read_pickle(loaded_filename)
#             break
    
#     if df is None:
#         raise ValueError(f"No .pickle files found in {path}")
    
#     if isinstance(df, pd.DataFrame):
#         df = tf.data.Dataset.from_tensors(df['history'])
#     elif isinstance(df, list):
#         df = tf.data.Dataset.from_tensors(df)
#     elif isinstance(df, np.ndarray):
#         df = tf.data.Dataset.from_tensors(df)
#     elif isinstance(df, dict) and 'history' in df.keys():
#         # print(f"df.keys() = {df.keys()}")
#         # raise ValueError(f"df type {type(df)} not supported")
#         # pass
#         df = tf.data.Dataset.from_tensors(df['history'])
#     elif isinstance(df, tf.data.Dataset):
#         pass
#     else:
#         raise ValueError(f"df type {type(df)} not supported")

#     df = (
#         df
#         # .map(lambda x: x['history'])
#         .window(N)
#     )
#     if infinite:
#         df = df.repeat()
#     df = df.prefetch(tf.data.AUTOTUNE)

#     def generator():
#         # create a generator that yields the history of the symbol in chunks of N days
#         if infinite:
#             while True:
#                 yield df.take(N)
#         else:
#             for i in range(0, len(df), N):
#                 yield df.take(N)
    
#     return generator
        

class PlottingTest(tf.test.TestCase):
    # def test_plot_first_market(self):
    #     plot_first_market()
        
    # def test_plot_history(self):
    #     import os
    #     path = 'data/'
    #     # load the first .pickle file in `path`
    #     for f in os.listdir(path):
    #         if f.endswith('.pickle'):
    #             print(f"Found file in cache: {f}")
    #             loaded_filename = f'{path}{f}'
    #             df = pd.read_pickle(loaded_filename)
    #             break
        
    #     if df is None:
    #         raise ValueError(f"No .pickle files found in {path}")
        
    #     plot_history(df['history'], f"{path}stock_prices.png")
    
    # # why is this so slow? (sped up dramatically using `tf.data.Dataset`)
    # def test_plot_candlesticks(self):
    #     import os
    #     path = 'data/'
    #     # load the first .pickle file in `path`
    #     for f in os.listdir(path):
    #         if f.endswith('.pickle'):
    #             print(f"Found file in cache: {f}") #if verbose > 0 else None
    #             loaded_filename = f'{path}{f}'
    #             df = pd.read_pickle(loaded_filename)
    #             break

    #     if df is None:
    #         raise ValueError(f"No .pickle files found in {path}")

    #     print(df['history'].head())
    #     images = tf.data.Dataset.from_tensors([
    #         plot_candlesticks(df['history'], i, show_plot=True) for i in range(4, 7, 1)])
    #     print(f"images = {len(images)} {images.element_spec}")
    #     # plot_candlesticks(df['history'])
    #     print(images)
    #     plt.plot(images.as_numpy_iterator())
    #     plt.show()

    def test_plot_candlesticks_movie(self):
        import os
        path = 'data/'
        # load the first .pickle file in `path`
        for f in os.listdir(path):
            if f.endswith('.pickle'):
                print(f"Found file in cache: {f}") #if verbose > 0 else None
                loaded_filename = f'{path}{f}'
                df = pd.read_pickle(loaded_filename)
                break

        if df is None:
            raise ValueError(f"No .pickle files found in {path}")

        print(df['history'].head())
        plot_candlesticks_movie(df['history'], frames=5, fps=2.5, filename='test-output.mp4')

    # # not working
    # def test_get_market_history_generator(self):
    #     generator = get_market_history_generator()
    #     print(f"generator = {generator}")
    #     i = 0
    #     for history in generator():
    #         print(f"history = {history}")
    #         plot_history(history)
    #         i = i + 1
    #         if i > 3:
    #             break
    #     plot_history(next(generator()))

    # def test_finplot(self):
    #     import finplot as fplt
    #     import yfinance

    #     df = yfinance.download('AAPL')
    #     fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']])
    #     fplt.show()
    
if __name__ == '__main__':
    tf.test.main()
       
# Function to create a neural network that can be used
# to classify market price data into different regimes:
# 1. Bullish
# 2. Bearish
# 3. Ranging
# 4. Flat
# The model is a Convolutional Neural Network (CNN) with
# a variable number of hidden layers and dropout regularization.
# The model is created using the Keras Functional API.
# The model is compiled with the Adam optimizer and the
# categorical crossentropy loss function.

#

def create_convolution_model(input_shape, hidden_layers=2, n_outputs=3, dropout=0.2,
                             loss='categorical_crossentropy',
                             optimizer='adam',
                             verbose=False) -> tf.keras.Model:
    # needs_padding = False
    if isinstance(hidden_layers, int):
        hidden_layers = [64 for _ in range(hidden_layers)]
    print(f"Creating model with input shape: {input_shape}") if verbose > 0 else None
    inputs = Input(shape=input_shape)
    print(f"Inputs: {inputs}") if verbose > 1 else None
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    print(f"Conv1D: {x}") if verbose > 1 else None
    x = MaxPooling1D()(x)
    print(f"MaxPooling1D: {x}") if verbose > 1 else None
    print(f"Hidden layers: {hidden_layers}") if verbose > 0 else None
    if isinstance(hidden_layers, list):
        for filters in hidden_layers:
            x = Conv1D(filters=filters, kernel_size=3, activation='relu', padding='same')(x)
            print(f"Conv1D: {x}") if verbose > 1 else None
            x = MaxPooling1D()(x)
            print(f"MaxPooling1D: {x}") if verbose > 1 else None

    if dropout:
        print(f"Dropout {dropout}") if verbose > 1 else None
        x = Dropout(dropout)(x)
    x = Flatten()(x)
    outputs = Dense(n_outputs, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary() if verbose > 0 else None
    
    # compile the model
    model.compile(loss=loss, optimizer=optimizer,
                  metrics= ['accuracy',
                            tf.keras.metrics.AUC(),
                            tf.keras.metrics.Precision(),
                            tf.keras.metrics.Recall(),
                            tf.keras.metrics.FalsePositives(),
                            tf.keras.metrics.TruePositives(),
                            tf.keras.metrics.FalseNegatives(),
                            tf.keras.metrics.TrueNegatives(),
                           ])
    
    return model


# class ConvolutionModelTest(tf.test.TestCase):    
#     def test_create_convolution_model(self):
#         input_shape = (4, 1)
#         model = create_convolution_model(input_shape, n_outputs=3)
#         self.assertEqual(model.input_shape, (None, *input_shape))
#         self.assertEqual(model.output_shape, (None, 3))
#         model.summary()
        
#         # input_shape = (28, 3)
#         # model = create_convolution_model(input_shape)
#         # self.assertEqual(model.input_shape, (None, 28, 3))
#         # self.assertEqual(model.output_shape, (None, 3))
        
#         # input_shape = (28, 3)
#         # model = create_convolution_model(input_shape, hidden_layers=3)
#         # self.assertEqual(model.input_shape, (None, 28, 3))
#         # self.assertEqual(model.output_shape, (None, 3))
        
#         # input_shape = (28, 3)
#         # model = create_convolution_model(input_shape, hidden_layers=[32, 64, 128])
#         # self.assertEqual(model.input_shape, (None, 28, 3))
#         # self.assertEqual(model.output_shape, (None, 3))
        
#         # input_shape = (28, 3)
#         # model = create_convolution_model(input_shape, hidden_layers=[32, 64, 128], dropout=0.5)
#         # self.assertEqual(model.input_shape, (None, 28, 3))
#         # self.assertEqual(model.output_shape, (None, 3))
        
#         # input_shape = (28, 3)
#         # model = create_convolution_model(input_shape, hidden_layers=[32, 64, 128], dropout=0.5, verbose=1)
#         # self.assertEqual(model.input_shape, (None, 28, 3))
#         # self.assertEqual(model.output_shape, (None, 3))
        
#         # input_shape = (28, 3)
#         # model = create_convolution_model(input_shape, hidden_layers=[32, 64, 128], dropout=0.5, verbose=2)
#         # self.assertEqual(model.input_shape, (None, 28, 3))
#         # self.assertEqual(model.output_shape, (None, 3))
