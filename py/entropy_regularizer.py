from typing_extensions import Self
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

class EntropyRegularizer(tf.keras.regularizers.Regularizer):
    def __call__(self, x):
        probabilities = EntropyRegularizer.normalize_weights(x.numpy())
        entropy = EntropyRegularizer.calculate_entropy(probabilities)
        return entropy

    def get_config(self):
        return {}

    @ staticmethod
    def normalize_weights(weights):
        exp_weights = np.exp(weights - np.max(weights))  # For numerical stability
        probabilities = exp_weights / np.sum(exp_weights)
        return probabilities

    @ staticmethod
    def calculate_entropy(probabilities):
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))  # Adding small value to avoid log(0)
        return entropy


class TestEntropyRegularizer(tf.test.TestCase):
    
    def setUp(self):
        super(TestEntropyRegularizer, self).setUp()
        self.regularizer = EntropyRegularizer()

    def test_normalize_weights(self):
        weights = np.array([1.0, -2.0, 7.0])
        probabilities = self.regularizer.normalize_weights(weights)
        expected_probabilities = np.exp(weights - np.max(weights)) / np.sum(np.exp(weights - np.max(weights)))
        self.assertAllClose(probabilities, expected_probabilities, atol=1e-6)

    def test_calculate_entropy(self):
        probabilities = np.array([0.1, 0.2, 0.7])
        entropy = self.regularizer.calculate_entropy(probabilities)
        expected_entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        self.assertAllClose(entropy, expected_entropy, atol=1e-6)

    def test_call(self):
        weights = tf.constant([1.0, 2.0, 3.0])
        entropy = self.regularizer(weights)
        probabilities = self.regularizer.normalize_weights(weights.numpy())
        expected_entropy = self.regularizer.calculate_entropy(probabilities)
        self.assertAllClose(entropy, expected_entropy, atol=1e-6)

    def test_get_config(self):
        config = self.regularizer.get_config()
        self.assertEqual(config, {})

    # TODO: test that this actually regularizes entropy of a model's weights
    

if __name__ == '__main__':
    tf.test.main()
