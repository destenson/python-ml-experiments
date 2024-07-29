import numpy as np
import matplotlib.pyplot as plt

# sparse distributed representation using columns of neurons

def make_column_mask(depth, n_neurons, sparsity, verbose=False):
    column = np.zeros((depth, n_neurons))
    for i in range(n_neurons):
        column[np.random.choice(depth, int(depth * sparsity), replace=False), i] = 1
    print(column) if verbose > 0 else None
    return column

# make_column_mask(10, 5, 0.2)
# make_column_mask(10, 5, 0.2, verbose=True)

def show_column(column, cmap='gray'):
    plt.imshow(column, cmap=cmap)
    plt.show()
    return column

def make_neuron(column, threshold=0.5, verbose=False):
    show_column(column) if verbose > 0 else None
    neuron = np.sum(column, axis=1) > threshold
    print(neuron) if verbose > 0 else None
    return neuron

def show_neuron(neuron, cmap='gray'):
    plt.imshow([neuron], cmap=cmap)
    plt.show()
    return neuron

# show_neuron(make_neuron(make_column_mask(10, 5, 0.2), 0.5, verbose=1))

def make_column(depth, n_neurons, sparsity, threshold=0.5, verbose=False):
    # column = make_column_mask(depth, n_neurons, sparsity, verbose)
    column = np.ones((depth, n_neurons))
    neurons = np.array([make_neuron(column, threshold, verbose) for _ in range(n_neurons)]).T
    print(neurons) if verbose > 0 else None
    return neurons

show_column(make_column(10, 3, 0.2, 0.5, verbose=1))

def make_layer(depth, n_neurons, sparsity, threshold=0.5, verbose=False):
    column = make_column_mask(depth, n_neurons, sparsity, verbose)
    layer = np.array([make_neuron(column, threshold, verbose) for _ in range(n_neurons)]).T
    print(layer) if verbose > 0 else None
    return layer

def show_layer(layer, cmap='gray'):
    plt.imshow(layer, cmap=cmap)
    plt.show()
    return layer
