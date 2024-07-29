
import keras.src
import tensorflow as tf
import tensorflow_probability as tfp
import keras

tfd = tfp.distributions

def get_custom_objects():
    return {
        "MyKeras>EKFLayer": EKFLayer,
    }

class ObservationFunction(keras.Function):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, x):
        return x

@keras.utils.register_keras_serializable(package="MyKeras")
class EKFLayer(tf.keras.layers.Layer):
    def __init__(self,
                 transition_fn, observation_fn,
                 transition_jacobian_fn, observation_jacobian_fn,
                 state_prior_mu = None, state_prior_sigma=None,
                 mu_initializer = 'normal', sigma_initializer='ones', **kwargs):
        super().__init__(**kwargs)
        self.input_shape = None
        self.transition_fn = transition_fn
        self.observation_fn = observation_fn
        self.transition_jacobian_fn = transition_jacobian_fn
        self.observation_jacobian_fn = observation_jacobian_fn
        self.state_prior_mu = state_prior_mu
        self.state_prior_sigma = state_prior_sigma
        self.mu_initializer = tf.keras.initializers.get(mu_initializer)
        self.sigma_initializer = tf.keras.initializers.get(sigma_initializer)
    
    def build(self, input_shape):
        self.input_shape = input_shape
        shape = (input_shape[-1],)
        
        self.state_prior_mu = self.state_prior_mu or self.add_weight(
            shape=shape,
            initializer=self.mu_initializer,
            regularizer=None,
            dtype=tf.float32,
            trainable=True,
            name='prior_mu'
        )
        
        self.state_prior_sigma = self.state_prior_sigma or self.add_weight(
            shape=shape,
            initializer=self.sigma_initializer,
            dtype=tf.float32,
            trainable=True,
            name='prior_sigma'
        )
        
        super().build(input_shape)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "transition_fn": self.transition_fn,
            "observation_fn": self.observation_fn,
            "transition_jacobian_fn": self.transition_jacobian_fn,
            "observation_jacobian_fn": self.observation_jacobian_fn,
            "state_prior_mu": self.state_prior_mu.numpy(),
            "state_prior_sigma": self.state_prior_sigma.numpy()
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        self = cls(**config)
        self.state_prior_mu = tf.Variable(config["state_prior_mu"])
        self.state_prior_sigma = tf.Variable(config["state_prior_sigma"])        
        return self

    def call(self, inputs):
        # TODO: should initial_state be removed or used?
        observations, initial_state = inputs
        state_prior = tfd.MultivariateNormalDiag(
            loc=self.state_prior_mu, scale_diag=self.state_prior_sigma)
        results = tfp.experimental.sequential.extended_kalman_filter(
            observations=observations,
            initial_state_prior=state_prior,
            transition_fn=self.transition_fn,
            observation_fn=self.observation_fn,
            transition_jacobian_fn=self.transition_jacobian_fn,
            observation_jacobian_fn=self.observation_jacobian_fn
        )
        # TODO: should we return the full results or just the means?
        return results.mean()

def test_ekf_layer(transition_fn, observation_fn, transition_jacobian_fn, observation_jacobian_fn):
    # Define the transition and observation functions as before
    # ...
    if transition_fn is None:
        transition_fn = lambda x, t: x
    if observation_fn is None:
        observation_fn = lambda x: x
    if transition_jacobian_fn is None:
        transition_jacobian_fn = lambda x, t: tf.eye(x.shape[-1])
    if observation_jacobian_fn is None:
        observation_jacobian_fn = lambda x: tf.eye(x.shape[-1])

    # Create the custom EKF layer
    ekf_layer = EKFLayer(transition_fn, observation_fn, transition_jacobian_fn, observation_jacobian_fn)

    # Example usage in a Keras model
    inputs = tf.keras.Input(shape=(None, 2))
    initial_state = tf.keras.Input(shape=(2,))
    filtered_means = ekf_layer([inputs, initial_state])
    model = tf.keras.Model(inputs=[inputs, initial_state], outputs=filtered_means)
    model.summary()

test_ekf_layer(None, None, None, None)
