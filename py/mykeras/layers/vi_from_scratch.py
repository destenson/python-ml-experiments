import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

import unittest
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Optional: Allow PyTorch as a backend
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class VariationalInference:
    def __init__(self, encoder, decoder, use_torch=False):
        self.encoder = encoder
        self.decoder = decoder
        self.use_torch = use_torch and TORCH_AVAILABLE
        
        if self.use_torch:
            import torch.nn as nn
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = keras.losses.MeanSquaredError()

    def gaussian_parameters(self, h):
        if self.use_torch:
            import torch
            mean = h[:, :h.shape[1]//2]
            h = h[:, h.shape[1]//2:]
            std = torch.exp(h / 2)
        else:
            mean = h[:, :h.shape[1]//2]
            h = h[:, h.shape[1]//2:]
            std = K.exp(h / 2)
        return mean, std

    def sample_z(self, args):
        mean, std = args
        if self.use_torch:
            import torch
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            eps = K.random_normal(shape=K.shape(std))
            return mean + std * eps

    def kl_divergence(self, mean, std):
        if self.use_torch:
            import torch
            return -0.5 * torch.sum(1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2), dim=1)
        else:
            return -0.5 * K.sum(1 + K.log(K.square(std)) - K.square(mean) - K.square(std), axis=1)

    def elbo_loss(self, x, x_hat, mean, std):
        reconstruction_loss = self.loss_fn(x, x_hat)
        kl_loss = self.kl_divergence(mean, std)
        
        if self.use_torch:
            return reconstruction_loss + kl_loss.mean()
        else:
            return reconstruction_loss + K.mean(kl_loss)

    def forward(self, x):
        h = self.encoder(x)
        mean, std = self.gaussian_parameters(h)
        z = self.sample_z((mean, std))
        x_hat = self.decoder(z)
        loss = self.elbo_loss(x, x_hat, mean, std)
        return x_hat, mean, std, z, loss

# Define the encoder and decoder models using Keras
def create_encoder(input_dim, latent_dim):
    return keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dense(latent_dim * 2)
    ])

def create_decoder(latent_dim, output_dim):
    return keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(latent_dim,)),
        keras.layers.Dense(output_dim, activation='sigmoid')
    ])

def test_vi(data):
    # Example usage
    input_dim = 784  # for MNIST
    latent_dim = 2
    output_dim = 784
    num_epochs = 10

    encoder = create_encoder(input_dim, latent_dim)
    decoder = create_decoder(latent_dim, output_dim)

    vi = VariationalInference(encoder, decoder, use_torch=False)

    # Training loop (simplified example)
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    @tf.function
    def train_step(x):
        with tf.GradientTape() as tape:
            _, _, _, _, loss = vi.forward(x)
        gradients = tf.scalar_mul(2., tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables))
        # gradients = tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables) * 2.
        optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))
        return loss

    # Assuming 'data' is your training dataset
    for epoch in range(num_epochs):
        for batch in data:
            loss = train_step(batch)
        print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")


# Assuming the VariationalInference, create_encoder, and create_decoder functions are already defined

class TestVariationalInference(tf.test.TestCase):

    def setUp(self):
        self.input_dim = 784
        self.latent_dim = 2
        self.output_dim = 784
        self.encoder = create_encoder(self.input_dim, self.latent_dim)
        self.decoder = create_decoder(self.latent_dim, self.output_dim)
        self.vi = VariationalInference(self.encoder, self.decoder, use_torch=False)

    def test_gaussian_parameters(self):
        h = np.random.randn(1, self.latent_dim * 2).astype(np.float32)
        mean, std = self.vi.gaussian_parameters(h)
        self.assertEqual(mean.shape, (1, self.latent_dim))
        self.assertEqual(std.shape, (1, self.latent_dim))
        self.assertTrue(np.all(std > 0))

    def test_sample_z(self):
        mean = np.random.randn(1, self.latent_dim).astype(np.float32)
        std = np.random.randn(1, self.latent_dim).astype(np.float32)
        z = self.vi.sample_z((mean, std))
        self.assertEqual(z.shape, (1, self.latent_dim))

    def test_kl_divergence(self):
        mean = np.random.randn(1, self.latent_dim).astype(np.float32)
        std = np.random.randn(1, self.latent_dim).astype(np.float32)
        kl_div = self.vi.kl_divergence(mean, std)
        self.assertEqual(kl_div.shape, (1,))
        self.assertTrue(np.all(kl_div >= 0))

    def test_elbo_loss(self):
        x = np.random.randn(1, self.input_dim).astype(np.float32)
        x_hat = np.random.randn(1, self.output_dim).astype(np.float32)
        mean = np.random.randn(1, self.latent_dim).astype(np.float32)
        std = np.random.randn(1, self.latent_dim).astype(np.float32)
        loss = self.vi.elbo_loss(x, x_hat, mean, std)
        self.assertTrue(loss.numpy() >= 0)

    def test_forward(self):
        x = np.random.randn(1, self.input_dim).astype(np.float32)
        x_hat, mean, std, z, loss = self.vi.forward(x)
        self.assertEqual(x_hat.shape, (1, self.output_dim))
        self.assertEqual(mean.shape, (1, self.latent_dim))
        self.assertEqual(std.shape, (1, self.latent_dim))
        self.assertEqual(z.shape, (1, self.latent_dim))
        self.assertTrue(loss.numpy() >= 0)
        
    def test_vi(self):
        (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, self.input_dim) / 255.0
        x_test = x_test.reshape(-1, self.input_dim) / 255.0
        data = tf.data.Dataset.from_tensor_slices(
            x_train).batch(32).prefetch(tf.data.AUTOTUNE)
        test_vi(data)
        
        

if __name__ == '__main__':
    tf.test.main()
