
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

import matplotlib.pyplot as plt 
import pandas as pd 
import matplotlib.dates as mpl_dates 
import datetime 
from mpl_finance import candlestick_ohlc 

import os

   
# Function to create a neural network that can be used
# to classify market price data into different regimes:
# 1. Bullish
# 2. Bearish
# 3. Ranging
# 4. Flat
