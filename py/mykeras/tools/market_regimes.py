
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

import matplotlib.pyplot as plt 
import pandas as pd 
import matplotlib.dates as mpl_dates 
import datetime 
from mpl_finance import candlestick_ohlc 

from py.mykeras.tools.video import create_video

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


class ConvolutionModelTest(tf.test.TestCase):
    
    def test_create_convolution_model(self):
        input_shape = (4, 1)
        model = create_convolution_model(input_shape, n_outputs=3)
        self.assertEqual(model.input_shape, (None, *input_shape))
        self.assertEqual(model.output_shape, (None, 3))
        model.summary()
        
        # input_shape = (28, 3)
        # model = create_convolution_model(input_shape)
        # self.assertEqual(model.input_shape, (None, 28, 3))
        # self.assertEqual(model.output_shape, (None, 3))
        
        # input_shape = (28, 3)
        # model = create_convolution_model(input_shape, hidden_layers=3)
        # self.assertEqual(model.input_shape, (None, 28, 3))
        # self.assertEqual(model.output_shape, (None, 3))
        
        # input_shape = (28, 3)
        # model = create_convolution_model(input_shape, hidden_layers=[32, 64, 128])
        # self.assertEqual(model.input_shape, (None, 28, 3))
        # self.assertEqual(model.output_shape, (None, 3))
        
        # input_shape = (28, 3)
        # model = create_convolution_model(input_shape, hidden_layers=[32, 64, 128], dropout=0.5)
        # self.assertEqual(model.input_shape, (None, 28, 3))
        # self.assertEqual(model.output_shape, (None, 3))
        
        # input_shape = (28, 3)
        # model = create_convolution_model(input_shape, hidden_layers=[32, 64, 128], dropout=0.5, verbose=1)
        # self.assertEqual(model.input_shape, (None, 28, 3))
        # self.assertEqual(model.output_shape, (None, 3))
        
        # input_shape = (28, 3)
        # model = create_convolution_model(input_shape, hidden_layers=[32, 64, 128], dropout=0.5, verbose=2)
        # self.assertEqual(model.input_shape, (None, 28, 3))
        # self.assertEqual(model.output_shape, (None, 3))

def plot_first_market(path='data/'):
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    # import numpy as np
    # from sklearn.preprocessing import MinMaxScaler

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
    # convert close and date to a suitable format to draw
    ax.plot(date, close)
    # fig.draw([close, date])
    if path is not None:
        plt.savefig(f'{path}stock_prices.png')
        # import os
        # plt.savefig(os.path.join(path, 'stock_prices.png'))
    plt.show()
    return tf.convert_to_tensor(fig.get_rasterized())
    

# class PlotFirstMarketTest(tf.test.TestCase):
#     def test_plot_first_market(self):
#         plot_first_market()


def plot_candlesticks(history, i=0, bars=5, show_plot=False):
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

    if show_plot:
        plt.show()
    # convert image to tensor
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return image

def plot_candlesticks_movie(history, i=0, bars=5, frames=10, fps=2.2, filename='output.mp4'):
    images = np.concatenate([
        plot_candlesticks(history, i=i+n, bars=bars, show_plot=False).numpy()
        for n in range(frames)
    ])
    # create video
    create_video(images, filename=filename, fps=fps)

class PlotCandlesticksTest(tf.test.TestCase):
    def test_plot_candlesticks(self):
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
        images = []
        for i in range(3):
            images.append(plot_candlesticks(df['history'], i))
        # plot_candlesticks(df['history'])
        print(images)
        plt.plot(images)
        plt.show()

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
        plot_candlesticks_movie(df['history'])

if __name__ == '__main__':
    tf.test.main()
       
#
