import streamlit as st
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Reshape
from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import plotly.graph_objects as go
import numpy as np

import datetime

import os
import io
import tempfile

# Function to create a simple neural network
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Function to create 3D visualization of the network
def visualize_network(model):
    layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    
    x, y, z = [], [], []
    edges_x, edges_y, edges_z = [], [], []
    
    for i, layer in enumerate(layers):
        layer_size = layer.units
        layer_x = [i] * layer_size
        layer_y = list(range(layer_size))
        layer_z = [0] * layer_size
        
        x.extend(layer_x)
        y.extend(layer_y)
        z.extend(layer_z)
        
        if i > 0:
            for j in range(layers[i-1].units):
                for k in range(layer_size):
                    edges_x.extend([i-1, i, None])
                    edges_y.extend([j, k, None])
                    edges_z.extend([0, 0, None])

    # Create the 3D scatter plot for nodes
    nodes = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=6, color=z, colorscale='Viridis', opacity=0.8),
        hoverinfo='text',
        text=[f'Layer {i}, Node {j}' for i, j in zip(x, y)]
    )

    # Create the 3D line plot for edges
    edges = go.Scatter3d(
        x=edges_x, y=edges_y, z=edges_z,
        mode='lines',
        line=dict(color='rgba(125,125,125,0.2)', width=1),
        hoverinfo='none'
    )

    # Create the layout
    layout = go.Layout(
        title='3D Neural Network Visualization',
        scene=dict(
            xaxis=dict(title='Layer'),
            yaxis=dict(title='Node'),
            zaxis=dict(title=''),
            aspectmode='manual',
            aspectratio=dict(x=2, y=1, z=0.5)
        ),
        margin=dict(r=20, l=10, b=10, t=40),
        showlegend=False,
    )

    # Create the figure and return it
    fig = go.Figure(data=[edges, nodes], layout=layout)
    return fig

# Streamlit app
st.title('Neural Network Visualizer')

# Create a model
model = create_model()

# Visualize the model
fig = visualize_network(model)

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# Display model summary
st.subheader('Model Summary')
model.summary(print_fn=lambda x: st.text(x))


# import tensorflow as tf
# import tensorflow.keras as keras
# import numpy as np
# # import math

@keras.utils.register_keras_serializable()
class UniqueNonZero(tf.keras.constraints.Constraint):
    def __init__(self, epsilon=1e-7):
        super(UniqueNonZero, self).__init__()
        self.epsilon = epsilon

    def __call__(self, w):
        # Ensure values are positive and non-zero (add a small epsilon)
        w_non_zero = tf.abs(w) + self.epsilon

        # Make values unique by adding a small increment to duplicates
        w_unique = tf.nn.relu(w_non_zero - tf.reduce_min(w_non_zero, axis=-1, keepdims=True))
        w_unique += tf.cast(tf.range(tf.shape(w_unique)[-1]), w_unique.dtype) * self.epsilon

        return w_unique

    def get_config(self):
        config = super(UniqueNonZero, self).get_config()
        config.update({
            "epsilon": self.epsilon
        })
        return config


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
        super(SVDLayer, self).__init__(**kwargs)
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

        w = keras.ops.matmul(keras.ops.multiply(U, S), V)

        result = keras.ops.add(keras.ops.matmul(inputs, w), self.b)

        # # Reshape result if input was not 2D
        # if len(original_shape) > 2:
        #     new_shape = tf.concat([original_shape[:-1], [self.output_dim]], axis=0)
        #     result = tf.reshape(result, new_shape)

        return self.activation(result)



    # def perform_svd(self):
    #     w = tf.keras.ops.matmul(tf.matmul(self.U, tf.linalg.diag(self.S)), self.V, transpose_b=True)
    #     s, u, v = tf.linalg.svd(w)
    #     return s, u, v




# File uploader for Keras model
uploaded_file = st.file_uploader("Upload a Keras (.keras or .h5) model", type=["keras", "h5"])

if uploaded_file is not None:
    # Load the model
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras" if uploaded_file.name.endswith(".keras") else ".h5") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    try:
        model = load_model(temp_file_path)
        
        # Visualize the model
        fig = visualize_network(model)
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Display model summary
        st.subheader('Model Summary')
        model.summary(print_fn=lambda x: st.text(x))
    except ImportError as e:
        st.error(f"Error loading model: {e}. This might be due to a version mismatch between the model and the installed Keras version.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)
else:
    st.write("Please upload a Keras (.keras or .h5) model file to visualize.")
