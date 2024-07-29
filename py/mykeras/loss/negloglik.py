
import tensorflow as tf

# def negloglik(y, rv_y):
#     return -rv_y.log_prob(y)

def negloglik(x, y, model_fn, axis=-1):
  """Negative log-likelihood."""
  return -tf.reduce_mean(model_fn(x).log_prob(y), axis=axis)
