
import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
import pandas as pd

def get_all_weights(model):
    # for layer in model.layers:
    #     print(layer.name, layer.input, layer.output)
    #     weights = np.zeros(img_shape)
    #     if hasattr(layer, 'get_weights'):
    #         layer_weights = layer.get_weights()
    #         if layer_weights:
    #             np.concatenate(weights, (np.concatenate([w.flatten() for w in layer_weights])))
    #     layer_weights[layer.name] = weights
    weights = {}
    for layer in model.layers:
        weights[layer.name] = layer.get_weights()
    return weights


class GetWeightsTest(tf.test.TestCase):
    
    def test_get_all_weights(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(4, 1)),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(3),
        ])
        model.build()
        weights = get_all_weights(model)
        self.assertEqual(len(weights), 2)
        self.assertEqual(len(weights['lstm']), 3)
        self.assertEqual(len(weights['dense']), 2)

    
if __name__ == '__main__':
    tf.test.main()

#
