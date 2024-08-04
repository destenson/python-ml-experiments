
import tensorflow as tf

# from tensorflow.keras import layers, models

from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector
from tensorflow.keras.layers import Reshape, Flatten
from tensorflow.keras.models import Model

import numpy as np
import pandas as pd

def build_model(input_shape, output_shape):
    inputs = Input(shape=input_shape)
    if input_shape[-1] == 1:
        inputs = Reshape((input_shape[0], input_shape[1]))(inputs)

    encoded = LSTM(32)(inputs)
    decoded = RepeatVector(output_shape[0])(encoded)
    decoded = LSTM(32, return_sequences=True)(decoded)

    # req_output_shape = output_shape
    # if len(output_shape) == 1:
    #     req_output_shape = (output_shape[0], 1)
    decoded = Dense(output_shape[-1], activation='linear')(decoded)
    # if len(output_shape) == 1:
    #     decoded = Reshape(output_shape)(decoded)

    model = Model(inputs, decoded)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'], run_eagerly=True)

    return model

class AddingModelTest(tf.test.TestCase):
    def test_build_model(self):
        model = build_model((28, 28, 1), (10,))
        self.assertIsInstance(model, Model)
        model.summary()
        
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
        train_x = train_x / 255.0
        test_x = test_x / 255.0
        train_x = train_x.reshape(-1, 28, 28, 1)
        test_x = test_x.reshape(-1, 28, 28, 1)
        train_y = tf.keras.utils.to_categorical(train_y, 10)
        test_y = tf.keras.utils.to_categorical(test_y, 10)
        
        # fit the model
        results = model.fit(
            train_x, train_y, epochs=1, batch_size=32, verbose=1,
            validation_data=(test_x, test_y))
        

if __name__ == '__main__':
    tf.test.main()
