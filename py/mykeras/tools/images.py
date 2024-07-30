
import tensorflow as tf

# Residual Block
def ResBlock(inputs, filters=64, kernel_size=3):
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = tf.keras.layers.Add()([inputs, x])
    return x

# Upsampling Block
def Upsampling(inputs, filters=64, kernel_size=3, factor=2, **kwargs):
    x = tf.keras.layers.Conv2D(filters * (factor ** 2), kernel_size, padding="same", **kwargs)(inputs)
    x = tf.nn.depth_to_space(x, block_size=factor)
    x = tf.keras.layers.Conv2D(filters * (factor ** 2), kernel_size, padding="same", **kwargs)(x)
    x = tf.nn.depth_to_space(x, block_size=factor)
    return x
