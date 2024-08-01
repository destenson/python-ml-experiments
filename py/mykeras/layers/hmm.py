import tensorflow as tf
import tensorflow_probability as tfp

# import tensorflow.keras.models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, ConvLSTM1D, Convolution1D
# from tensorflow.keras.layers import Dropout, BatchNormalization, Activation
# from tensorflow.keras.layers import Flatten, Reshape
# from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Bidirectional
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
# from tensorflow.keras.layers import Conv3D, MaxPooling3D, AveragePooling3D
from tensorflow.keras.optimizers import Adam

import numpy as np
import keras

## prints the version of tensorflow & clears custom objects
# print(tf.__version__)
# keras.saving.get_custom_objects().clear()

def get_custom_objects():
    return {
        "MyKeras>HMMNeuronLayer": HMMNeuronLayer,
    }

@keras.utils.register_keras_serializable(package="MyKeras")
class HMMNeuronLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_hmm_states, hmm_params=None, **kwargs):
        super().__init__(**kwargs)
        # print(f"HMMNeuronLayer({locals()})")
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
        # print(f"HMMNeuronLayer.build({locals()})")
        super().build(input_shape)

    def get_config(self):
        # print(f"HMMNeuronLayer.get_config({locals()})")
        config = super().get_config()
        config.update({
            "units": self.units,
            "num_hmm_states": self.num_hmm_states,
            "hmm_params": self.hmm_params.numpy(),
        })
        # print(f"HMMNeuronLayer.get_config: {config}")
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

    def call(self, inputs):
        # print(f"HMMNeuronLayer.call({locals()})")
        if self.hmm is None:
            transitions = tfp.distributions.Categorical(
                probs=self.hmm_params[0])

            batch_size = tf.shape(inputs)[0]

            # create a normal distribution with batch shape of transitions
            observations = tfp.distributions.Normal(
                loc=tf.zeros((batch_size, self.num_hmm_states)),
                scale=tf.ones((batch_size, self.num_hmm_states))
            )

            # print(f"HMMNeuronLayer.call(inputs.shape={inputs.shape})")
            self.hmm = tfp.distributions.HiddenMarkovModel(
                initial_distribution=tfp.distributions.Categorical(
                    probs=self.hmm_params[0, 0]),
                transition_distribution=transitions,
                observation_distribution=observations,
                num_steps=inputs.shape[-1] #if len(inputs.shape) > 1 else 1
            )

        # Use HMM to process inputs
        hmm_output = self.hmm.posterior_mode(inputs)
        return hmm_output



class TestHmmLayer(tf.test.TestCase):
    
    def test_hmm_neuron_layer(self):
        # Create an instance of HMMNeuronLayer
        hmm_neuron_layer = HMMNeuronLayer(units=7, num_hmm_states=15, name='hmm_neuron_layer_test')
        # print(hmm_neuron_layer)
        self.assertEqual(hmm_neuron_layer.units, 7)
        self.assertEqual(hmm_neuron_layer.num_hmm_states, 15)
        print(hmm_neuron_layer(tf.random.normal((22, 1))))

    def setUp(self):
        super().setUp()
        self.hmm_layer = HMMNeuronLayer(units=1, num_hmm_states=5, name='hmm_layer')
        # print(self.hmm_layer)
        # print(self.hmm_layer(tf.random.normal((2, 2, 2))))
        self.assertEqual(self.hmm_layer.units, 1)
        self.assertEqual(self.hmm_layer.num_hmm_states, 5)

    def test_call(self):
        weights = np.array([1.0, 1.0, 0.0])
        w_unique = self.hmm_layer(weights)
        exp_unique = [1.0, 1.0 + 1e-7, 1e-7]
        self.assertNotAllClose(w_unique, exp_unique, atol=1e-7)
        
    def test_model(self):
        
        input_dim = 28
        output_dim = 10

        input_shape = (28, input_dim,1)

        # tf.debugging.disable_traceback_filtering()
        model = Sequential([
            Input(shape=input_shape),
            Conv2D(32, 3, activation='relu'),
            # MaxPooling2D((2, 2)),
            # Conv2D(64, (3, 3), activation='relu'),
            # MaxPooling2D((2, 2)),
            # Flatten(),
            # Dense(8, activation='relu'),
            HMMNeuronLayer(num_hmm_states=4, units=1, name='hmm_neuron_layer_test'),
            # Dense(8, activation='relu'),
            Dense(output_dim, activation='softmax')
        ])
        model.summary(expand_nested=True, show_trainable=True)
        compile_kwargs = {
            'optimizer': Adam(learning_rate=0.001),
            'loss': 'sparse_categorical_crossentropy',
            'metrics': ['accuracy']
        }
        model.compile(**compile_kwargs)
        model.summary()


    def test_get_config(self):
        config = self.hmm_layer.get_config()
        self.assertEqual({
            'units': 1,
            'num_hmm_states': 5,
            'hmm_params': self.hmm_layer.hmm_params.numpy(),
            'name': 'hmm_layer',
            'dtype': { # TODO: do this right
                'module': 'keras',
                'class_name': 'DTypePolicy',
                'config': {'name': 'float32'},
                'registered_name': None
            },
            'trainable': True,
        }, config)
        

if __name__ == '__main__':
    tf.test.main()


#
