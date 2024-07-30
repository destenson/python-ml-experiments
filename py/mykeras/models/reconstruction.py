
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



def get_reconstruction_model(input_shape: tuple, verbose: int|bool=False) -> keras.Model:
    assert len(input_shape) == 2 or (len(input_shape)==3 and input_shape[0] is None), "Input shape must be a tuple of 2 integers"

    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(32, 7, padding="same", strides=2, activation="relu")(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(16, 7, padding="same", strides=2, activation="relu")(x)
    encoder_outputs = x
    x = layers.Conv1DTranspose(16, 7, padding="same", strides=2, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1DTranspose(32, 7, padding="same", strides=2, activation="relu")(x)
    outputs = layers.Conv1DTranspose(input_shape[-1], 7, padding="same")(x)
    encoder = keras.Model(inputs=inputs, outputs=encoder_outputs)
    decoder = keras.Model(inputs=encoder_outputs, outputs=outputs)
    autoencoder = keras.Model(inputs=inputs, outputs=outputs)

    # model = keras.Sequential(
    #     [
    #         layers.Input(shape=input_shape),
    #         layers.Conv1D(filters=32, kernel_size=7,
    #                       padding="same", strides=2,
    #                       activation="relu"),
    #         layers.Dropout(rate=0.2),
    #         layers.Conv1D(filters=16, kernel_size=7, 
    #                       padding="same", strides=2,
    #                       activation="relu"),
    #         layers.Conv1DTranspose(filters=16, kernel_size=7,
    #                                padding="same", strides=2,
    #                                activation="relu"),
    #         layers.Dropout(rate=0.2),
    #         layers.Conv1DTranspose(filters=32, kernel_size=7,
    #                                padding="same", strides=2,
    #                                activation="relu"),
    #         layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    #     ]
    # )
    encoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    encoder.summary() if verbose > 0 else None
    decoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    decoder.summary() if verbose > 0 else None
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    autoencoder.summary() if verbose > 0 else None
    return autoencoder, encoder, decoder

class ReconstructionModelTest(tf.test.TestCase):
    def test_get_reconstruction_model(self):
        input_shape = (28, 1)
        autoencoder, encoder, decoder = get_reconstruction_model(input_shape)
        self.assertEqual(autoencoder.input_shape, (None, 28, 1))
        self.assertEqual(autoencoder.output_shape, (None, 28, 1))
        self.assertEqual(encoder.input_shape, (None, 28, 1))
        self.assertEqual(encoder.output_shape, (None, 7, 16))
        self.assertEqual(decoder.input_shape, (None, 7, 16))
        self.assertEqual(decoder.output_shape, (None, 28, 1))
        
        input_shape = (28, 3)
        autoencoder, encoder, decoder = get_reconstruction_model(input_shape)
        self.assertEqual(autoencoder.input_shape, (None, 28, 3))
        self.assertEqual(autoencoder.output_shape, (None, 28, 3))
        self.assertEqual(encoder.input_shape, (None, 28, 3))
        self.assertEqual(encoder.output_shape, (None, 7, 16))
        self.assertEqual(decoder.input_shape, (None, 7, 16))
        self.assertEqual(decoder.output_shape, (None, 28, 3))

if __name__ == "__main__":
    tf.test.main()
