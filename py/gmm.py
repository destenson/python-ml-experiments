import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow_probability import distributions as tfd
import keras


@keras.utils.register_keras_serializable(package="Custom")
class GMMLayer(Layer):
    def __init__(self, num_components, **kwargs):
        super(GMMLayer, self).__init__(**kwargs)
        self.num_components = num_components

    def build(self, input_shape):
        self.means = self.add_weight(shape=(self.num_components, input_shape[-1]),
                                     initializer='random_normal',
                                     trainable=True)
        self.logits = self.add_weight(shape=(self.num_components,),
                                      initializer='random_normal',
                                      trainable=True)
        self.scales = self.add_weight(shape=(self.num_components, input_shape[-1]),
                                      initializer='random_normal',
                                      trainable=True)
        
    def get_config(self):
        config = super(GMMLayer).get_config()
        config.update({
            "num_components": self.num_components,
        })
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])
        

    def call(self, inputs):
        gm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=self.logits),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=self.means,
                scale_diag=tf.nn.softplus(self.scales)
            )
        )
        return gm.log_prob(inputs)

# Example usage
input_dim = 5
inputs = tf.keras.Input(shape=(input_dim,))
gmm_layer = GMMLayer(num_components=3)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=gmm_layer)
model.summary()


input_dim = 10  # Example input dimension
num_components = 3  # Number of GMM components

inputs = Input(shape=(input_dim,input_dim,))
gmm_output = GMMLayer(num_components=num_components)(inputs)
hidden = Dense(64, activation='relu')(gmm_output)
output = Dense(1, activation='sigmoid')(hidden)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()
