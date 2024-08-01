
import tensorflow as tf

# Residual Block
def ResBlock(inputs, filters=64, kernel_size=3):
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = tf.keras.layers.Add()([inputs, x])
    return x

# Upsampling Block
def Upsampling(inputs, filters=64, kernel_size=3, factor=2, **kwargs):
    x = tf.keras.layers.Conv2D(int(filters * (factor ** 2)), kernel_size, padding="same", **kwargs)(inputs)
    x = tf.keras.layers.Lambda(lambda inp: tf.nn.depth_to_space(inp, block_size=factor))(x)
    # x = tf.keras.layers.Conv2D(filters * (factor ** 2), kernel_size, padding="same", **kwargs)(x)
    # x = tf.keras.layers.Lambda(lambda inp: tf.nn.depth_to_space(inp, block_size=factor))(x)
    return x

class ResBlockTest(tf.test.TestCase):
    def test_ResBlock(self):
        inputs = tf.keras.Input(shape=(28, 28, 1))
        x = ResBlock(inputs)
        model = tf.keras.Model(inputs, x)
        model.summary()
        self.assertEqual(model.output_shape, (None, 28, 28, 64))

class UpsamplingTest(tf.test.TestCase):
    def test_Upsampling(self):
        inputs = tf.keras.Input(shape=(28, 28, 1))
        x = Upsampling(inputs)
        model = tf.keras.Model(inputs, x)
        model.summary()
        self.assertEqual(model.output_shape, (None, 56, 56, 64))

    def test_Upsampling0(self):
        inputs = tf.keras.Input(shape=(28, 28, 1))
        x = Upsampling(inputs, factor = 4)
        model = tf.keras.Model(inputs, x)
        model.summary()
        self.assertEqual(model.output_shape, (None, 112, 112, 64))

if __name__ == '__main__':
    tf.test.main()
    

#
