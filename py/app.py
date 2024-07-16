import streamlit as st
import tensorflow as tf
# import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Reshape
from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import keras

import plotly.graph_objects as go
import numpy as np

import math
import datetime

import os
import io
import tempfile

import gcp as gcp_py
import hmm_layer as hmm_layer_py
import losses as losses_py
import svd_layer as svd_layer_py

from svd_layer import SVDLayer, UniqueNonZero
from hmm_layer import HMMNeuronLayer

# @st.cache_data
# Function to create a simple neural network
def create_simple_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)) ,
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# @st.cache_data
def create_custom_model(input_shape, hidden_layers=2):
    inputs = tf.keras.layers.Input(input_shape=input_shape)


# Function to create 3D visualization of the network
def visualize_network(model, name):
    layers = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)
    ]
    
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
        title=f'3D Neural Network Visualization of {name}',
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
model = create_simple_model()

# Visualize the model
fig = visualize_network(model, 'a simple model')

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# Display model summary
st.subheader('Model Summary')
model.summary(print_fn=st.text)

#  custom_objects={"CustomLayer": CustomLayer, "custom_fn": custom_fn}
 
def load_model_file(path, name=None):
    # Pass the custom objects dictionary to a custom object scope and place
    # the `keras.models.load_model()` call within the scope.
    custom_objects = {}
    custom_objects.update(gcp_py.get_custom_objects())
    custom_objects.update(hmm_layer_py.get_custom_objects())
    custom_objects.update(losses_py.get_custom_objects())
    custom_objects.update(svd_layer_py.get_custom_objects())

    with keras.saving.custom_object_scope(custom_objects):
        model = keras.models.load_model(path)

    # Visualize the model
    fig = visualize_network(model, path if not name else name)
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Display model summary
    st.subheader('Model Summary')
    model.summary(print_fn=st.text)
    return model

 
# List of sample models
model_files = ["None","rcnn.h5","rcnn.keras","model1.keras", "model2.keras", "model3.h5"]  # Add your model file names here

# Dropdown menu to select a model
selected_model = st.selectbox("Select a model to visualize", model_files)

if selected_model and selected_model != 'None':
    # Load the selected model
    model = load_model_file(selected_model)
else:
    st.write("Please select a model file to visualize.")

# File uploader for Keras model
uploaded_file = st.file_uploader("Upload a Keras (.keras or .h5) model", type=["keras", "h5"])

if uploaded_file is not None:
    # Load the model
    # Save the uploaded file temporarily
    print(f"Uploaded filename: {uploaded_file.name}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras" if uploaded_file.name.endswith(".keras") else ".h5") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    try:
        model = load_model_file(temp_file_path, uploaded_file.name)
    # except ImportError as e:
    #     st.error(f"Error loading model: {e}. This might be due to a version mismatch between the model and the installed Keras version.")
    # except Exception as e:
    #     st.error(f"Error loading model: {e}")
    # except:
    #     st.error(f"Error loading model (unknown reason)")
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)
else:
    st.write("Please upload a Keras (.keras or .h5) model file to visualize.")
