
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten


# Function to create a neural network that can be used
# to classify market price data into different regimes:
# 1. Bullish
# 2. Bearish
# 3. Sideways
# The model is a Convolutional Neural Network (CNN) with
# a variable number of hidden layers and dropout regularization.
# The model is created using the Keras Functional API.
# The model is compiled with the Adam optimizer and the
# categorical crossentropy loss function.

def create_convolution_model(input_shape, hidden_layers=2, dropout=0.2,
                             loss='categorical_crossentropy',
                             optimizer='adam',
                             verbose=False):
    print(f"Creating model with input shape: {input_shape}") if verbose > 0 else None
    inputs = Input(input_shape=input_shape)
    print(f"Inputs: {inputs}") if verbose > 1 else None
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    print(f"Conv1D: {x}") if verbose > 1 else None
    x = MaxPooling1D()(x)
    print(f"MaxPooling1D: {x}") if verbose > 1 else None
    print(f"Hidden layers: {hidden_layers}") if verbose > 0 else None
    if isinstance(hidden_layers, int):
        for _ in range(hidden_layers):
            x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
            print(f"Conv1D: {x}") if verbose > 1 else None
            x = MaxPooling1D()(x)
            print(f"MaxPooling1D: {x}") if verbose > 1 else None
    elif isinstance(hidden_layers, list):
        for filters in hidden_layers:
            x = Conv1D(filters=filters, kernel_size=3, activation='relu')(x)
            print(f"Conv1D: {x}") if verbose > 1 else None
            x = MaxPooling1D()(x)
            print(f"MaxPooling1D: {x}") if verbose > 1 else None

    if dropout:
        print(f"Dropout {dropout}") if verbose > 1 else None
        x = Dropout(dropout)(x)
    x = Flatten()(x)
    outputs = Dense(3, activation='softmax')(x)
    
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
