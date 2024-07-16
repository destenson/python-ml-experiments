import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse


import numpy as np
import math
import keras


def create_lagrangian_network(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    output = Dense(1)(x)
    return Model(inputs, output)

input_shape = (2,)  # Assuming input is [price, price_derivative]
lagrangian_model = create_lagrangian_network(input_shape)

@tf.function
def euler_lagrange(model, q, q_dot):
    with tf.GradientTape() as tape:
        tape.watch(q)
        tape.watch(q_dot)
        L = model(tf.concat([q, q_dot], axis=-1))
    dL_dq = tape.gradient(L, q)
    dL_dq_dot = tape.gradient(L, q_dot)
    
    with tf.GradientTape() as tape2:
        tape2.watch(q)
        tape2.watch(q_dot)
        dL_dq_dot = tape.gradient(L, q_dot)
    d_dt_dL_dq_dot = tape2.gradient(dL_dq_dot, q)
    
    return d_dt_dL_dq_dot - dL_dq



class LNN(tf.keras.Model):
    def __init__(self):
        super(LNN, self).__init__()
        self.lagrangian = create_lagrangian_network(input_shape)
    
    def call(self, inputs):
        q, q_dot = tf.split(inputs, 2, axis=-1)
        return euler_lagrange(self.lagrangian, q, q_dot)


def train_a_model(num_epochs, dataset):
    model = LNN()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    @tf.function
    def train_step(q, q_dot, targets):
        with tf.GradientTape() as tape:
            predictions = model(tf.concat([q, q_dot], axis=-1))
            loss = tf.reduce_mean(tf.square(predictions - targets))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    # Assuming `dataset` is a tf.data.Dataset object with (q, q_dot, targets) tuples
    for epoch in range(num_epochs):
        for q_batch, q_dot_batch, target_batch in dataset:
            loss = train_step(q_batch, q_dot_batch, target_batch)
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")
    
    return model


class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(latent_dim + latent_dim)  # mean and log variance
        ])
        self.decoder = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(2)  # reconstruct [price, price_derivative]
        ])
    
    def encode(self, x):
        z_mean, z_log_var = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return z_mean, z_log_var
    
    def reparameterize(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=z_mean.shape)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def decode(self, z):
        return self.decoder(z)
    
    def call(self, inputs):
        z_mean, z_log_var = self.encode(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decode(z)
        return reconstructed, z_mean, z_log_var

latent_dim = 2
vae = VAE(latent_dim)

# Loss function
def vae_loss(inputs, reconstructed, z_mean, z_log_var):
    reconstruction_loss = mse(inputs, reconstructed)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return reconstruction_loss + kl_loss

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

@tf.function
def train_step(inputs):
    with tf.GradientTape() as tape:
        reconstructed, z_mean, z_log_var = vae(inputs)
        loss = vae_loss(inputs, reconstructed, z_mean, z_log_var)
    gradients = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
    return loss


batch_size = 32
num_epochs = 1000

for dataset in data:
    print(f"dataset.shape: {dataset.shape}")
    dataset = dataset.dropna(inplace=False, axis=0)
    print(f"dataset.shape: {dataset.shape}")
    # Training loop
    for epoch in range(num_epochs):
        if epoch % 31 == 0:
            print(f"Epoch {epoch} beginning")
            print(f"Using dataset: {dataset}")
        for batch in dataset.values:
            # print(f"batch: {batch}")
            loss = train_step(batch.reshape(-1, 2))
        if epoch % 31 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy()}")

print(f"dataset: {data[0]}")
# Training loop
loss_history = []
k=0
for d in data:
    d = tf.convert_to_tensor(d.dropna(inplace=False))
    batch_count = len(d) // batch_size
    rem = len(d) % batch_size

    ticker_loss_history = []
    print(f"Training")
    print(f"Using data: {d}")
    for epoch in range(num_epochs):
        epoch_loss_history = []
        if epoch % 31 == 0:
            print(f"Epoch {epoch+1} beginning")
        if epoch == num_epochs-1:
            print("Last epoch beginning")
        # split d into batch_size batches
        losses = []
        for n in range(batch_count):
            batch = d[n * batch_size:(n + 1) * batch_size]
            loss = train_step(batch)
            losses.append(np.dot(loss, loss))
            epoch_loss_history.append(loss)
        rms = np.sqrt(np.mean(losses))

        ticker_loss_history.append(epoch_loss_history)

        if epoch % 31 == 0 or epoch == num_epochs-1:
            print(f"Epoch {epoch+1}, RMS Loss: {rms:.4}")
            # print(f"d[3000:+1]      = {d.iloc[3000:3001]}")
            # reconstructed, z_mean, z_log_var = vae(d.iloc[3000:3001].values)
            # print(f"vae(d[3000:+1]) = {reconstructed}, {z_mean}, {z_log_var}")
            # print(f"d[3002]         = {d.iloc[3002]}")
            # pt = predict_trajectory(vae, d[3000:3010], 10, .1)
            # print(f"pt = {pt}")
            # print(f"pt[0] = {pt[0]}")
            # print(f"pt[1] = {pt[1]}")

    loss_history.append(ticker_loss_history)
    # %mkdir -p lnn
    vae.save(f"lnn/vae-{k:02}-{epoch+1}.keras")
    k = k+1



# class LNN_VAE(tf.keras.Model):
#     def __init__(self, vae, lnn):
#         super(LNN_VAE, self).__init__()
#         self.vae = vae
#         self.lnn = lnn
    
#     def call(self, inputs):
#         _, z_mean, _ = self.vae(inputs)
#         return self.lnn(z_mean)

# lnn_vae = LNN_VAE(vae, LNN())

# # Training loop for LNN_VAE
# for epoch in range(num_epochs):
#     for batch in dataset:
#         print(f"batch: {batch}")
#         loss = train_step(batch)
#     print(f"Epoch {epoch}, Loss: {loss.numpy()}")


# def predict_trajectory(model, initial_state, steps, dt=None):
#     trajectory = [initial_state]
#     state = initial_state
#     dt = 1./steps if dt is None else dt
#     for _ in range(steps):
#         acceleration = model(state)
#         # Update state using simple Euler integration
#         new_q = state[:, :1] + state[:, 1:] * dt
#         new_q_dot = state[:, 1:] + acceleration * dt
#         state = tf.concat([new_q, new_q_dot], axis=-1)
#         trajectory.append(state.numpy())
#     return np.stack(trajectory)

# # Example usage
# initial_state = tf.constant([data[200:201]])  # Example initial state [price, price_derivative]
# predicted_trajectory = predict_trajectory(model, initial_state, steps=100, dt=0.01)
