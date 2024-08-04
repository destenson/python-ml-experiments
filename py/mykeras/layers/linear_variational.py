
import keras.api
import tensorflow as tf
import keras
import keras.api.backend as K
from keras.api.layers import LSTM, RepeatVector, Reshape, Flatten, Dense, Input, Layer

# import torch
# import torch.nn as nn

@keras.saving.register_keras_serializable(package='MyKeras')
class LinearVariational(Layer):
    """
    Mean field approximation of a linear layer.
    """

    def __init__(self, in_features, out_features, parent, n_batches, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.include_bias = bias
        self.parent = parent
        self.n_batches = n_batches

        if getattr(parent, 'accumulated_kl_div', None) is None:
            parent.accumulated_kl_div = 0

        # Initialize the variational parameters.
        # 𝑄(𝑤)=N(𝜇_𝜃,𝜎2_𝜃)
        # Do some random initialization with 𝜎=0.001
        self.w_mu = self.add_weight(
            shape=(in_features, out_features),
            initializer='glorot_normal',
            trainable=True,
        )
        # proxy for variance
        # log(1 + exp(ρ))◦ eps
        self.w_p = self.add_weight(
            shape=(in_features, out_features),
            initializer='glorot_normal',
            trainable=True,
        )
        if self.include_bias:
            self.b_mu = self.add_weight(
                shape=(out_features,),
                initializer='zeros',
                trainable=True,
            )
            # proxy for variance
            self.b_p = self.add_weight(
                shape=(out_features,),
                initializer='zeros',
                trainable=True,
            )

    def reparameterize(self, mu, p):
        sigma = keras.ops.log(1 + keras.ops.exp(p))
        eps = keras.random.normal(sigma)
        # sigma = torch.log(1 + torch.exp(p)) 
        # eps = torch.randn_like(sigma)
        return mu + (eps * sigma)
    
    def kl_divergence(self, z, mu_theta, p_theta, prior_sd=1):
        log_prior = K.random_normal(0, prior_sd).log_prob(z)
        log_p_q = K.random_normal(mu_theta, K.log(1 + K.exp(p_theta))).log_prob(z)
        # log_prior = dist.Normal(0, prior_sd).log_prob(z) 
        # log_p_q = dist.Normal(mu_theta, torch.log(1 + torch.exp(p_theta))).log_prob(z) 
        return (log_p_q - log_prior).sum() / self.n_batches
    
    def call(self, x):
        w = self.reparameterize(self.w_mu, self.w_p)
        
        if self.include_bias:
            b = self.reparameterize(self.b_mu, self.b_p)
        else:
            b = 0
            
        z = x @ w + b
        
        self.parent.accumulated_kl_div += self.kl_divergence(w, 
                                                             self.w_mu,
                                                             self.w_p, 
                                                             )
        if self.include_bias:
            self.parent.accumulated_kl_div += self.kl_divergence(b, 
                                                                 self.b_mu, 
                                                                 self.b_p,
                                                                 )
        return z

class LinearVariationalTest(tf.test.TestCase):
    def test_linear_variational_layer(self):
        model = keras.models.Sequential([
            LinearVariational(10, 10, self, 100),
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'], run_eagerly=True)
        model.summary()
        
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
        train_x = train_x / 255.0
        test_x = test_x / 255.0
        train_x = train_x.reshape(-1, 28, 28, 1)
        test_x = test_x.reshape(-1, 28, 28, 1)
        train_y = keras.utils.to_categorical(train_y, 10)
        test_y = keras.utils.to_categorical(test_y, 10)
        
        # fit the model
        results = model.fit(
            train_x, train_y, epochs=1, batch_size=32, verbose=1,
            validation_data=(test_x, test_y))

# class LinearVariational(nn.Module):
#     """
#     Mean field approximation of nn.Linear
#     """
#     def __init__(self, in_features, out_features, parent, n_batches, bias=True):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.include_bias = bias        
#         self.parent = parent
#         self.n_batches = n_batches
        
#         if getattr(parent, 'accumulated_kl_div', None) is None:
#             parent.accumulated_kl_div = 0
            
#         # Initialize the variational parameters.
#         # 𝑄(𝑤)=N(𝜇_𝜃,𝜎2_𝜃)
#         # Do some random initialization with 𝜎=0.001
#         self.w_mu = nn.Parameter(
#             torch.FloatTensor(in_features, out_features).normal_(mean=0, std=0.001)
#         )
#         # proxy for variance
#         # log(1 + exp(ρ))◦ eps
#         self.w_p = nn.Parameter(
#             torch.FloatTensor(in_features, out_features).normal_(mean=-2.5, std=0.001)
#         )
#         if self.include_bias:
#             self.b_mu = nn.Parameter(
#                 torch.zeros(out_features)
#             )
#             # proxy for variance
#             self.b_p = nn.Parameter(
#                 torch.zeros(out_features)
#             )
        
#     def reparameterize(self, mu, p):
#         sigma = torch.log(1 + torch.exp(p)) 
#         eps = torch.randn_like(sigma)
#         return mu + (eps * sigma)
    
#     def kl_divergence(self, z, mu_theta, p_theta, prior_sd=1):
#         log_prior = dist.Normal(0, prior_sd).log_prob(z) 
#         log_p_q = dist.Normal(mu_theta, torch.log(1 + torch.exp(p_theta))).log_prob(z) 
#         return (log_p_q - log_prior).sum() / self.n_batches

#     def forward(self, x):
#         w = self.reparameterize(self.w_mu, self.w_p)
        
#         if self.include_bias:
#             b = self.reparameterize(self.b_mu, self.b_p)
#         else:
#             b = 0
            
#         z = x @ w + b
        
#         self.parent.accumulated_kl_div += self.kl_divergence(w, 
#                                                              self.w_mu,
#                                                              self.w_p, 
#                                                              )
#         if self.include_bias:
#             self.parent.accumulated_kl_div += self.kl_divergence(b, 
#                                                                  self.b_mu, 
#                                                                  self.b_p,
#                                                                  )
#         return z
            


if __name__ == '__main__':
    tf.test.main()
