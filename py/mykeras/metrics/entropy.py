
import tensorflow as tf
import tensorflow.keras.backend as K

@tf.function
def entropy_metric(y_true, y_pred):
    return entropy(y_pred)

def entropy(y):
    # Apply softmax to get probabilities
    y_softmax = tf.nn.softmax(y)

    # Calculate entropy
    entropy = -tf.reduce_sum(
        y_softmax * tf.math.log(y_softmax + K.epsilon()),
        axis=-1)

    return K.mean(entropy)
