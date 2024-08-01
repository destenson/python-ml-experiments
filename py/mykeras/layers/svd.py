
from py.mykeras.constraints.uniquenonzero import UniqueNonZero

import tensorflow as tf
# import tensorflow.keras as keras
import numpy as np
import math
import keras
# from py.mykeras.constraints.uniquenonzero import UniqueNonZero





@keras.utils.register_keras_serializable()
class SVDLayer(keras.layers.Layer):
    """
    A layer that keeps its weights Singular Value Decomposed.

    Attributes:
      output_dim: dimensions of the output space
      rank: rank of the SVD decomposition
      activation: activation function to apply to the output
      seed: seed for any random features
      U: rotation weights matrix
      S: singular values matrix
      V: rotation weights matrix
    """
    def __init__(self, output_dim,
                 rank=None, low_rank=None, activation='relu',
                 s_regularizer=None, s_initializer=None,
                 u_regularizer=None, u_initializer=None,
                 v_regularizer=None, v_initializer=None,
                 b_regularizer=None, b_initializer=None,
                 seed=None,
                 **kwargs):
        super().__init__(**kwargs)
        if output_dim is None:
            # TODO: allow automatically configuring the output dimension
            raise ValueError("output_dim must be specified")
        self.output_dim = output_dim
        self.rank = rank if rank is not None else output_dim
        self.low_rank = low_rank if low_rank is not None else self.rank
        self.seed = seed

        self.activation = tf.keras.activations.get(activation) or tf.keras.activations.linear

        self.s_initializer = tf.keras.initializers.get(s_initializer) or tf.keras.initializers.Ones()
        self.u_initializer = tf.keras.initializers.get(u_initializer) or tf.keras.initializers.Orthogonal()
        self.v_initializer = tf.keras.initializers.get(v_initializer) or tf.keras.initializers.Orthogonal()
        self.b_initializer = tf.keras.initializers.get(b_initializer) or tf.keras.initializers.Zeros()

        self.s_regularizer = tf.keras.regularizers.get(s_regularizer)
        self.u_regularizer = tf.keras.regularizers.get(u_regularizer) or tf.keras.regularizers.orthogonal_regularizer(0.01, mode='columns')
        self.v_regularizer = tf.keras.regularizers.get(v_regularizer) or tf.keras.regularizers.orthogonal_regularizer(0.01, mode='rows')
        self.b_regularizer = tf.keras.regularizers.get(b_regularizer)

    def compute_output_shape(self, input_shape):
        # TODO: verify this is correct
        return (input_shape[0], self.output_dim)

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def call_and_return_dict(self, inputs):
        return {"output": self.call(inputs)}

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "rank": self.rank,
            "low_rank": self.low_rank,
            "seed": self.seed,
            "activation": tf.keras.activations.serialize(self.activation),
            "s_regularizer": tf.keras.regularizers.serialize(self.s_regularizer),
            "s_initializer": tf.keras.initializers.serialize(self.s_initializer),
            "u_regularizer": tf.keras.regularizers.serialize(self.u_regularizer),
            "u_initializer": tf.keras.initializers.serialize(self.u_initializer),
            "v_regularizer": tf.keras.regularizers.serialize(self.v_regularizer),
            "v_initializer": tf.keras.initializers.serialize(self.v_initializer),
            "b_regularizer": tf.keras.regularizers.serialize(self.b_regularizer),
            "b_initializer": tf.keras.initializers.serialize(self.b_initializer),
        })
        return config

    def get_weights(self):
        return [self.U.numpy(), self.S.numpy(), self.V.numpy(), self.b.numpy()]

    def set_weights(self, weights):
        self.U.assign(weights[0])
        self.S.assign(weights[1])
        self.V.assign(weights[2])
        self.b.assign(weights[3])

    def build(self, input_shape):
        # print(f"SVDLayer.build(): input_shape={input_shape}")
        # TODO: add support for other, such as, complex types

        if self.rank is None:
            self.rank = math.min(input_shape[-1], self.output_dim)
            print(f"SVDLayer        : rank={self.rank}")

        if self.low_rank is None or self.low_rank > self.rank:
            self.low_rank = self.rank
            print(f"SVDLayer        : low_rank={self.low_rank}")

        # Initialize U, Σ, and V
        self.U = self.add_weight(name="U", #dtype=dtype,
                                 shape=(input_shape[-1], self.rank),
                                 initializer=self.u_initializer,
                                 regularizer=self.u_regularizer,
                                 constraint=tf.keras.constraints.UnitNorm(axis=0),
                                 trainable=True)

        self.S = self.add_weight(name="S", #dtype=dtype,
                                 shape=(self.rank,),
                                 initializer=self.s_initializer,
                                 regularizer=self.s_regularizer,
                                 constraint=UniqueNonZero(),
                                 trainable=True)

        self.V = self.add_weight(name="V", #dtype=dtype,
                                 shape=(self.rank, self.output_dim),
                                 initializer=self.v_initializer,
                                 regularizer=self.v_regularizer,
                                 constraint=tf.keras.constraints.UnitNorm(axis=1),
                                 trainable=True)

        self.b = self.add_weight(name="bias", #dtype=dtype,
                                 shape=(self.output_dim,),
                                 initializer=self.b_initializer,
                                 regularizer=self.b_regularizer,
                                 trainable=True)

    def call(self, inputs):
        # # Flatten input if it's not 2D
        # original_shape = tf.shape(inputs)
        # if len(inputs.shape) > 2:
        #     inputs = tf.reshape(inputs, (-1, inputs.shape[-1]))

        # Reconstruct the weight matrix (or a low-rank approximation)
        U = self.U[:, :self.low_rank]
        S = self.S[:self.low_rank]
        V = self.V[:self.low_rank, :]

        w = keras.ops.matmul(keras.ops.multiply(U, S), V, transpose_b=True)

        result = keras.ops.add(keras.ops.matmul(inputs, w), self.b)

        # # Reshape result if input was not 2D
        # if len(original_shape) > 2:
        #     new_shape = tf.concat([original_shape[:-1], [self.output_dim]], axis=0)
        #     result = tf.reshape(result, new_shape)

        return self.activation(result)


def make_svd_layer_from_dense_layer(dense_layer, rank=None, low_rank=None, verbose=True):
    weights, biases = dense_layer.get_weights()
    input_dim, output_dim = weights.shape
    s, u, v = tf.linalg.svd(weights)
    svd_layer = SVDLayer(output_dim, rank=rank, low_rank=low_rank)
    assert input_dim == 576
    svd_layer.build((input_dim,))
    svd_layer.set_weights([u[:,:rank],s[:rank],v[:rank,:],biases])
    print(f"make_svd_layer_from_dense_layer: dense_layer={dense_layer}, svd_layer={svd_layer}") if verbose > 0 else None
    return svd_layer

    # def perform_svd(self):
    #     w = tf.keras.ops.matmul(tf.matmul(self.U, tf.linalg.diag(self.S)), self.V, transpose_b=True)
    #     s, u, v = tf.linalg.svd(w)
    #     return s, u, v

def get_custom_objects():
    return {
        "SVDLayer": SVDLayer,
    }

class SvdLayerTests(tf.test.TestCase):
    def test_simple_svd_layer(self):
        svdlayer = SVDLayer(10, rank=5, low_rank=3)
        svdlayer.build((10,))
        self.assertEqual(svdlayer.output_dim, 10)
        self.assertEqual(svdlayer.rank, 5)
        self.assertEqual(svdlayer.low_rank, 3)

        self.assertEqual(svdlayer.U.shape, (10, 5))
        self.assertEqual(svdlayer.S.shape, (5,))
        self.assertEqual(svdlayer.V.shape, (5, 10))

        # self.assertAlmostEqual(svdlayer.s_initializer, tf.keras.initializers.Ones())
        # self.assertEqual(svdlayer.u_initializer, tf.keras.initializers.Orthogonal())
        # self.assertEqual(svdlayer.v_initializer, tf.keras.initializers.Orthogonal())
        # self.assertEqual(svdlayer.b_initializer, tf.keras.initializers.Zeros())

        self.assertIsNone(svdlayer.s_regularizer)
        self.assertIsNotNone(svdlayer.u_regularizer)
        self.assertIsNotNone(svdlayer.v_regularizer)
        self.assertIsNone(svdlayer.b_regularizer)
        self.assertEqual(svdlayer.activation, tf.keras.activations.relu)
        self.assertIsNone(svdlayer.seed)
        self.assertEqual(svdlayer.compute_output_shape((None, 10)), (None, 10))
        
    def test_copy_from_dense(self):
        from py.mykeras.datasets.mnist import load_mnist_cnn
        model = load_mnist_cnn('cnn-functional-99.3.keras')
        dense_layer = model.layers[-1]
        # dense_layer.build((10,))
        svdlayer = make_svd_layer_from_dense_layer(dense_layer, rank=5, low_rank=3)
        svdlayer.build((576,))
        self.assertEqual(svdlayer.output_dim, 10)
        self.assertEqual(svdlayer.rank, 5)
        self.assertEqual(svdlayer.low_rank, 3)

        self.assertEqual(svdlayer.U.shape, (576, 5))
        self.assertEqual(svdlayer.S.shape, (5,))
        self.assertEqual(svdlayer.V.shape, (5, 10))

        # self.assertEqual(svdlayer.s_initializer, tf.keras.initializers.Ones())
        # self.assertEqual(svdlayer.u_initializer, tf.keras.initializers.Orthogonal())
        # self.assertEqual(svdlayer.v_initializer, tf.keras.initializers.Orthogonal())
        # self.assertEqual(svdlayer.b_initializer, tf.keras.initializers.Zeros())

        self.assertIsNone(svdlayer.s_regularizer)
        self.assertIsNotNone(svdlayer.u_regularizer)
        self.assertIsNotNone(svdlayer.v_regularizer)
        self.assertIsNone(svdlayer.b_regularizer)
        self.assertEqual(svdlayer.activation, tf.keras.activations.relu)
        self.assertIsNone(svdlayer.seed)
        self.assertEqual(svdlayer.compute_output_shape((None, 10)), (None, 10))

        weights, biases = dense_layer.get_weights()
        svd_weights = svdlayer.get_weights()
        svd_biases = svd_weights[-1]
        svd_weights = (svd_weights[0] @ svd_weights[1]) @ svd_weights[2].transpose()
        self.assertAllClose(weights, svd_weights)
        self.assertAllClose(biases, svd_biases)

# svdlayer = make_svd_layer_from_dense_layer(keras.layers.Dense(10), rank=5, low_rank=3)
# print(f"{svdlayer}")

if __name__ == "__main__":
    tf.test.main()
