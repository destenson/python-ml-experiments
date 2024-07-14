import tensorflow as tf
import tensorflow_probability as tfp

import tensorflow.keras.models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Input, ConvLSTM1D, Convolution1D
from tensorflow.keras.layers import Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import Flatten, Reshape
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Bidirectional
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Conv3D, MaxPooling3D, AveragePooling3D
from tensorflow.keras.optimizers import Adam

import numpy as np
import keras

# print(tf.__version__)
# keras.saving.get_custom_objects().clear()

def get_custom_objects():
    return {
        "HMMNeuronLayer": HMMNeuronLayer,
        "Custom>HMMNeuronLayer": HMMNeuronLayer,
    }

@keras.utils.register_keras_serializable(package="HMMLayers")
class HMMNeuronLayer(Layer):
    def __init__(self, units, num_hmm_states, hmm_params=None, **kwargs):
        super(HMMNeuronLayer, self).__init__(**kwargs)
        print(f"HMMNeuronLayer({locals()})")
        self.units = units
        self.num_hmm_states = num_hmm_states
        self.hmm = None
        self.hmm_params = hmm_params or self.add_weight(
            shape=(self.units, self.num_hmm_states, self.num_hmm_states),
            initializer='random_normal',
            trainable=True,
            name='hmm_params'
        )

    def build(self, input_shape):
        print(f"HMMNeuronLayer.build({locals()})")
        super(HMMNeuronLayer, self).build(input_shape)

    def call(self, inputs):
        print(f"HMMNeuronLayer.call({locals()})")
        if self.hmm is None:
            transitions = tfp.distributions.Categorical(
                probs=self.hmm_params[0])

            batch_size = tf.shape(inputs)[0]

            # create a normal distribution with batch shape of transitions
            observations = tfp.distributions.Normal(
                loc=tf.zeros((batch_size, self.num_hmm_states)),
                scale=tf.ones((batch_size, self.num_hmm_states))
            )

            self.hmm = tfp.distributions.HiddenMarkovModel(
                initial_distribution=tfp.distributions.Categorical(
                    probs=self.hmm_params[0, 0]),
                transition_distribution=transitions,
                observation_distribution=observations,
                num_steps=inputs.shape[1]
            )

        # Use HMM to process inputs
        hmm_output = self.hmm.posterior_mode(inputs)
        return hmm_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "num_hmm_states": self.num_hmm_states,
            "hmm_params": self.hmm_params.numpy(),
        })
        return config

    @classmethod
    def from_config(cls, config):
        self = cls(**config)
        self.hmm_params = tf.Variable(config["hmm_params"])
        return self

    # def compute_output_shape(self, input_shape):
    #     # TODO: verify this is correct
    #     return (input_shape[0], self.output_dim)

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def call_and_return_dict(self, inputs):
        return {"output": self.call(inputs)}



# def test_hmm_neuron_layer():
#     # Create an instance of HMMNeuronLayer
#     hmm_neuron_layer = HMMNeuronLayer(units=1, num_hmm_states=5)
#     print(hmm_neuron_layer)
#     print(hmm_neuron_layer(tf.random.normal((2, 2, 2))))

# test_hmm_neuron_layer()
