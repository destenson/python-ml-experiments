import tensorflow as tf
import keras

def get_custom_objects():
    return {
        "GlobalContextPool": GlobalContextPool,
    }

@keras.utils.register_keras_serializable()
class GlobalContextPool(tf.keras.layers.Layer):
    def __init__(self, activation='relu', **kwargs):
        super(GlobalContextPool, self).__init__(**kwargs)
        self.max_pool = tf.keras.layers.GlobalMaxPooling1D(keepdims=True)
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        super(GlobalContextPool, self).build(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2] * 2)

    def get_config(self):
        scfg = super(GlobalContextPool, self).get_config()
        config = {
            "activation": tf.keras.activations.serialize(self.activation),
            "max_pool": tf.keras.layers.serialize(self.max_pool),
        }
        return {**scfg, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        # save the max of each feature and append the global max to each
        # data point, doubling the number of features at each data frame
        max_pooled = self.max_pool(inputs)
        # max_pooled = tf.expand_dims(max_pooled, axis=1)
        # concatted = tf.concat([inputs, tf.broadcast_to(max_pooled, tf.shape(inputs))], axis=-1)
        # return concatted
        max_pooled = tf.tile(max_pooled, [1, tf.shape(inputs)[1], 1])
        if self.activation is not None:
            return self.activation(tf.concat([inputs, max_pooled], axis=1))
        else:
            return tf.concat([inputs, max_pooled], axis=1)

# class GlobalContextLayer(layers.Layer):
#     def call(self, inputs):
#         # Apply global max pooling
#         global_max_pooled = tf.reduce_max(inputs, axis=1, keepdims=True)
#         # Tile the global max pooled features to match the input shape
#         max_pooled = tf.tile(global_max_pooled, [1, tf.shape(inputs)[1], 1])
#         # Concatenate the global max pooled features with the input
#         return tf.concat([inputs, max_pooled], axis=1)
