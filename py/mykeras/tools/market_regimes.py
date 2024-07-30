
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten


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


if __name__ == '__main__':
    tf.test.main()
       
#
