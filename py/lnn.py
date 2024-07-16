import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

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


def predict_trajectory(model, initial_state, steps, dt=None):
    trajectory = [initial_state]
    state = initial_state
    dt = 1/steps if dt is None else dt
    for _ in range(steps):
        acceleration = model(state)
        # Update state using simple Euler integration
        new_q = state[:, :1] + state[:, 1:] * dt
        new_q_dot = state[:, 1:] + acceleration * dt
        state = tf.concat([new_q, new_q_dot], axis=-1)
        trajectory.append(state)
    return tf.stack(trajectory)

# Example usage
initial_state = tf.constant([[100.0, 1.0]])  # Example initial state [price, price_derivative]
predicted_trajectory = predict_trajectory(model, initial_state, steps=100, dt=0.01)


