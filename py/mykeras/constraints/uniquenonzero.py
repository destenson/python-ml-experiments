
import tensorflow as tf
# import tensorflow.keras as keras
import numpy as np
import math
import keras

def get_custom_objects():
    return {
        "MyKeras>UniqueNonZero": UniqueNonZero,
    }

@keras.utils.register_keras_serializable(package="MyKeras")
class UniqueNonZero(tf.keras.constraints.Constraint):
    '''
    A constraint that ensures unique non-zero values in a tensor.   
    '''
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, w):
        # Ensure values are positive and non-zero (add a small epsilon)
        w_non_zero = tf.abs(w) + self.epsilon

        # Make values unique by adding a small increment to duplicates
        w_unique = tf.nn.relu(w_non_zero - tf.reduce_min(w_non_zero, axis=-1, keepdims=True))
        w_unique += tf.cast(tf.range(tf.shape(w_unique)[-1]), w_unique.dtype) * self.epsilon

        return w_unique

    def get_config(self):
        config = super().get_config()
        config.update({
            "epsilon": self.epsilon
        })
        return config

class TestUniqueNonZero(tf.test.TestCase):
    
    def setUp(self):
        super().setUp()
        self.constraint = UniqueNonZero()

    def test_call(self):
        weights = np.array([1.0, 1.0, 0.0])
        w_unique = self.constraint(weights)
        exp_unique = [1.0, 1.0 + 1e-7, 1e-7]
        self.assertAllClose(w_unique, exp_unique, atol=0.99e-7)

    def test_get_config(self):
        config = self.constraint.get_config()
        self.assertEqual(config, {"epsilon": 1e-7})


if __name__ == '__main__':
    tf.test.main()
