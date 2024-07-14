
import streamlit as st
import tensorflow as tf
import numpy as np
import plotly.graph_objects as go

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