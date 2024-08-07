
import tensorflow as tf
from tensorflow import keras
import numpy as np


class LogisticEndpoint(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.accuracy_metric = keras.metrics.BinaryAccuracy(name="accuracy")

    def call(self, logits, targets=None, sample_weight=None):
        if targets is not None:
            # Compute the training-time loss value and add it
            # to the layer using `self.add_loss()`.
            loss = self.loss_fn(targets, logits, sample_weight)
            self.add_loss(loss)

            # Log the accuracy as a metric (we could log arbitrary metrics,
            # including different metrics for training and inference.)
            self.accuracy_metric.update_state(targets, logits, sample_weight)

        # Return the inference-time prediction tensor (for `.predict()`).
        return tf.nn.softmax(logits)

class LogisticEndpointTest(tf.test.TestCase):
    def test_logistic_endpoint(self):
        inputs = keras.Input((764,), name="inputs")
        logits = keras.layers.Dense(1)(inputs)
        targets = keras.Input((1,), name="targets")
        sample_weight = keras.Input((1,), name="sample_weight")
        preds = LogisticEndpoint()(logits, targets, sample_weight)
        model = keras.Model([inputs, targets, sample_weight], preds)

        data = {
            "inputs": np.random.random((1000, 764)),
            "targets": np.random.random((1000, 1)),
            "sample_weight": np.random.random((1000, 1)),
        }

        model.compile(keras.optimizers.Adam(1e-3))
        model.fit(data, epochs=2)


if __name__ == '__main__':
    tf.test.main()

#
