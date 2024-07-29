import tensorflow as tf

def mse_loss(y_pred, y):
    return tf.reduce_mean(tf.square(y_pred - y))
