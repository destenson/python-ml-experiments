import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow_probability import distributions as tfd
import numpy as np
import matplotlib.pyplot as plt

# Once you have extracted the weights, you can perform various
# statistical analyses to understand their distribution and
# identify potential pruning candidates:
# - Calculate basic statistics: Compute mean, median, standard 
#   deviation, and percentiles of the weight values.
# - Visualize weight distributions: Create histograms or kernel
#   density plots to visualize the distribution of weights 
#   across different gates (input, forget, cell, and output).
# - Analyze weight magnitudes: Look at the absolute values of
#   weights to identify those with the smallest magnitudes,
#   which are often candidates for pruning.
#
# Importance scoring:
# Develop a scoring mechanism to rank the importance of
# weights. Some approaches include:
# - Magnitude-based scoring: Rank weights based on their
#   absolute values.
# - Gradient-based scoring: Use gradients during training to
#   identify which weights contribute most to the model's
#   performance.
# - Activity-based scoring: Analyze the activation patterns
#   of neurons to identify less active connections.
#
# Pruning strategy:
# Based on your analysis, you can develop a pruning strategy:
# - Global threshold: Set a global threshold and prune all
#   weights below it.
# - Per-gate thresholds: Set different thresholds for each
#   gate based on their individual statistics.
# - Gradual pruning: Implement an iterative pruning approach,
#   where you prune a small percentage of weights, retrain
#   the model, and repeat the process.
# - Soft pruning with Squeeze-Excitation-Pruning (SEP): 
#   Instead of hard pruning, which permanently removes
#   weights, consider using a soft pruning approach like
#   SEP. This method selectively excludes some kernels during
#   forward and backward propagations based on a learned
#   pruning scheme, allowing for data-dependent pruning.
# LSTM-guided pruning:
# An advanced approach is to use another LSTM to learn the
# hierarchical characteristics of your network and generate
# a global pruning scheme. This method can help identify
# which layers are more suitable for pruning, potentially
# leading to better complexity reduction with less
# performance drop.
#
# Evaluation and fine-tuning:
# After pruning, it's crucial to evaluate the model's
# performance and fine-tune if necessary. You may need to
# iterate through the pruning process multiple times to
# find the optimal balance between model size and
# performance.

def get_LSTM_gate_weights(weights):
    '''
    Extract the weights of the LSTM gates from the LSTM layer weights.
    The weights are in the form of a list of 3 numpy arrays.
    The function returns a dictionary of dictionaries with the keys being the
    gates of the LSTM cell and the values being dictionaries with the keys
    'W', 'U', and 'b' for the weights of the input, recurrent, and bias
    respectively.
    
    The weights come from the LSTM layer such as the following:

        import tensorflow as tf
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(units))
        weights = model.layers[0].get_weights()
    '''
    W, U, b = weights
    gates = ["i", "f", "c", "o"]
    hunit = U.shape[0]
    gate_weights = {}
    for i, gate in enumerate(gates):
        gate_weights[gate] = {
            'W': W[:, i*hunit:(i+1)*hunit],
            'U': U[:, i*hunit:(i+1)*hunit],
            'b': b[i*hunit:(i+1)*hunit]
        }
    return gate_weights



def lstm_weight_statistic(weights):
    '''
    Calculate basic statistics of the LSTM weights.
    '''
    gate_weights = get_LSTM_gate_weights(weights)
    statistics = {}
    for gate, weights in gate_weights.items():
        W, U, b = weights['W'], weights['U'], weights['b']
        statistics[gate] = {
            'W': {
                'mean': W.mean(),
                'median': np.median(W),
                'std': W.std(),
                'percentile': np.percentile(W, [25, 50, 75])
            },
            'U': {
                'mean': U.mean(),
                'median': np.median(U),
                'std': U.std(),
                'percentile': np.percentile(U, [25, 50, 75])
            },
            'b': {
                'mean': b.mean(),
                'median': np.median(b),
                'std': b.std(),
                'percentile': np.percentile(b, [25, 50, 75])
            }
        }
    return statistics

def simple_lstm(units, input_shape=(None, 1)):
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(shape=input_shape),
        tf.keras.layers.LSTM(units)
    ])

def visualize_weight_distribution(weights):
    gate_weights = get_LSTM_gate_weights(weights)
    for gate, weights in gate_weights.items():
        W, U, b = weights['W'], weights['U'], weights['b']
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].hist(W.flatten(), bins=50, alpha=0.7, color='b')
        ax[0].set_title(f"{gate} W")
        ax[1].hist(U.flatten(), bins=50, alpha=0.7, color='r')
        ax[1].set_title(f"{gate} U")
        ax[2].hist(b.flatten(), bins=50, alpha=0.7, color='g')
        ax[2].set_title(f"{gate} b")
        plt.show()
    
    # show histogram of all weights
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    W = np.concatenate([gate_weights[gate]['W'].flatten() for gate in gate_weights])
    U = np.concatenate([gate_weights[gate]['U'].flatten() for gate in gate_weights])
    b = np.concatenate([gate_weights[gate]['b'].flatten() for gate in gate_weights])
    ax[0].hist(W, bins=50, alpha=0.7, color='b')
    ax[0].set_title(f"All W")
    ax[1].hist(U, bins=50, alpha=0.7, color='r')
    ax[1].set_title(f"All U")
    ax[2].hist(b, bins=50, alpha=0.7, color='g')
    ax[2].set_title(f"All b")
    plt.show()

def lstm_weight_pruning(weights, threshold=0.1):
    gate_weights = get_LSTM_gate_weights(weights)
    pruned_weights = {}
    for gate, weights in gate_weights.items():
        W, U, b = weights['W'], weights['U'], weights['b']
        W_pruned = np.where(np.abs(W) < threshold, 0, W)
        U_pruned = np.where(np.abs(U) < threshold, 0, U)
        b_pruned = np.where(np.abs(b) < threshold, 0, b)
        pruned_weights[gate] = {
            'W': W_pruned,
            'U': U_pruned,
            'b': b_pruned
        }
    return pruned_weights    

class LstmFunctionTests(tf.test.TestCase):    
    def test_get_LSTM_gate_weights(self):
        l = simple_lstm(20, (20, 10))
        weights = l.layers[0].get_weights()
        gate_weights = []
        for weight in weights:
            # print(weight.shape)
            # print(weight)
            # assign random values to the weights
            weight = np.random.randn(*weight.shape, 4*20)
            gate_weights.append(weight)

        # l.weights = [np.random.randn(10, 4*20), np.random.randn(20, 4*20), np.random.randn(4*20)]
        gate_weights = get_LSTM_gate_weights(weights)
        self.assertEqual(len(gate_weights), 4)
        for gate, weights in gate_weights.items():
            self.assertEqual(weights['W'].shape, (10, 20))
            self.assertEqual(weights['U'].shape, (20, 20))
            self.assertEqual(weights['b'].shape, (20,))
    
    def test_lstm_weight_statistic(self):
        weights = [np.random.randn(10, 4*20), np.random.randn(20, 4*20), np.random.randn(4*20)]
        stats = lstm_weight_statistic(weights)
        self.assertEqual(len(stats), 4)
        for gate, statistics in stats.items():
            self.assertEqual(len(statistics), 3)
            for weight_type, stats in statistics.items():
                self.assertEqual(len(stats), 4)
                self.assertIn('mean', stats)
                self.assertIn('median', stats)
                self.assertIn('std', stats)
                self.assertIn('percentile', stats)
                
                

if __name__ == "__main__":
    if False:
        tf.test.main()
    else:
        print(tf.__version__)
        # model = simple_lstm(10)
        model = tf.keras.models.load_model('mnist_lstm.keras')
        print(model.summary())
        # weights = model.layers[0].get_weights()
        # stats = lstm_weight_statistic(weights)
        # print(f"stats: {stats}")
        # visualize_weight_distribution(weights)
