import tensorflow as tf
import keras

def get_custom_objects():
    return {
        "MyKeras>DynamicOutputLayer": DynamicOutputLayer,
    }

@keras.utils.register_keras_serializable(package="MyKeras")
class DynamicOutputLayer(tf.keras.layers.Layer):
    def __init__(self, n=64, initial_classes=10,
                 kernel_initializer='random_normal', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 weights=None, biases=None, **kwargs):
        super(DynamicOutputLayer, self).__init__(**kwargs)
        self.n = n
        self.initial_classes = initial_classes
        self.W = weights
        self.b = biases
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    
    def build(self, input_shape):
        n = self.n
        initial_classes = self.initial_classes
        self.W = self.add_weight(
            shape=(n*input_shape[-1], initial_classes),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True)
        self.b = self.add_weight(
            shape=(initial_classes,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            trainable=True)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "n": self.n,
            "initial_classes": self.initial_classes,
            "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": tf.keras.regularizers.serialize(self.bias_regularizer),
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        self = cls(**config)
        self.initial_classes = config["initial_classes"]
        self.n = config["n"]
        self.kernel_initializer = tf.keras.initializers.deserialize(config["kernel_initializer"])
        self.kernel_regularizer = tf.keras.regularizers.deserialize(config["kernel_regularizer"])
        self.bias_initializer = tf.keras.initializers.deserialize(config["bias_initializer"])
        self.bias_regularizer = tf.keras.regularizers.deserialize(config["bias_regularizer"])
        return self
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.W.shape[1])

    def add_class(self):
        new_W = tf.Variable(initial_value=tf.random.normal([self.W.shape[0], 1]), trainable=True)
        self.W = tf.concat([self.W, new_W], axis=1)
        new_b = tf.Variable(initial_value=tf.zeros([1]), trainable=True)
        self.b = tf.concat([self.b, new_b], axis=0)

    def call(self, inputs):
        input = tf.reshape(inputs, [-1, self.W.shape[0]])
        # print(f"inputs.shape={inputs.shape}, input.shape={input.shape}, W.shape={self.W.shape}, b.shape={self.b.shape}")
        return tf.matmul(input, self.W) + self.b


def build_dynamic_model(input_shape=(28, 28, 1)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = DynamicOutputLayer()(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.AUC(name='auc'),
                           tf.keras.metrics.FalsePositives(name='fp'),
                           tf.keras.metrics.TruePositives(name='tp'),
                           tf.keras.metrics.FalseNegatives(name='fn'),
                           tf.keras.metrics.TrueNegatives(name='tn'),
                           tf.keras.metrics.Precision(name='prc'),
                           tf.keras.metrics.Recall(name='rcl'),
                          ])
    return model

# model = build_dynamic_model()

class DynamicOutputLayerTest(tf.test.TestCase):
    # def setUp(self):
    #     super().setUp()
    #     self.output_layer = DynamicOutputLayer(n=10, initial_classes=5)

    def test_dynamic_output_layer_0(self):
        layer = DynamicOutputLayer(n=2, initial_classes=10)
        layer.build(input_shape=(28, 28, 15))
        self.assertEqual(layer.W.shape, (30, 10))
        self.assertEqual(layer.b.shape, (10,))
        layer.add_class()
        self.assertEqual(layer.W.shape, (30, 11))
        self.assertEqual(layer.b.shape, (11,))

    def test_dynamic_output_layer_1(self):
        layer = DynamicOutputLayer(n=128, initial_classes=6)
        layer.build(input_shape=(28, 1))
        self.assertEqual(layer.W.shape, (128, 6))
        self.assertEqual(layer.b.shape, (6,))
        layer.add_class()
        self.assertEqual(layer.W.shape, (128, 7))
        self.assertEqual(layer.b.shape, (7,))

    def test_dynamic_output_layer_2(self):
        layer = DynamicOutputLayer(n=64, initial_classes=7)
        layer.build(input_shape=(32, 2))
        self.assertEqual(layer.W.shape, (128, 7))
        self.assertEqual(layer.b.shape, (7,))
        layer.add_class()
        self.assertEqual(layer.W.shape, (128, 8))
        self.assertEqual(layer.b.shape, (8,))

    def test_dynamic_output_layer_serialization(self):
        layer = DynamicOutputLayer(n=64, initial_classes=10)
        config = layer.get_config()
        new_layer = DynamicOutputLayer.from_config(config)
        self.assertAllEqual(layer.W, new_layer.W)
        self.assertAllEqual(layer.b, new_layer.b)
    
    def test_dynamic_output_layer_add_class(self):
        layer = DynamicOutputLayer(n=64, initial_classes=10)
        layer.build(input_shape=(32, 1))
        layer.add_class()
        self.assertEqual(layer.W.shape, (64, 11))
        self.assertEqual(layer.b.shape, (11,))
        layer.add_class()
        self.assertEqual(layer.W.shape, (64, 12))
        self.assertEqual(layer.b.shape, (12,))
    
    # def test_dynamic_output_layer_call(self):
    #     layer = DynamicOutputLayer(n=64, initial_classes=10)
    #     layer.build(input_shape=(None, 64, 1))
    #     inputs = tf.random.normal([32, 64, 1])
    #     outputs = layer(inputs)
    #     self.assertEqual(outputs.shape, (32, 10))
    
    def test_model_with_dynamic_output(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(16, 7)),
            DynamicOutputLayer(n=16, initial_classes=5),
        ])
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        model.fit(tf.random.normal([39, 16, 7]), tf.random.normal([39, 5]), epochs=1)
        model.evaluate(tf.random.normal([32, 16, 7]), tf.random.normal([32, 5]))
        model.predict(tf.random.normal([32, 16, 7]))
        model.save('test_model0.keras')
        model = tf.keras.models.load_model('test_model0.keras', custom_objects=get_custom_objects())
        model.summary()
        model.evaluate(tf.random.normal([15, 16, 7]), tf.random.normal([15, 5]))
        model.predict(tf.random.normal([15, 16, 7]))

    # def test_build_dynamic_model(self):
    #     model = build_dynamic_model((32, 64, 2))
    #     model.summary()
    #     model.fit(tf.random.normal([32, 64, 2]), tf.random.normal([32, 10]), epochs=1)
    #     model.evaluate(tf.random.normal([32, 64, 2]), tf.random.normal([32, 10]))
    #     model.predict(tf.random.normal([32, 64, 2]))
    #     # model.save('test_model1.keras')
    #     # model = tf.keras.models.load_model('test_model1.keras', custom_objects=get_custom_objects())
    #     # model.summary()
    #     # model.evaluate(tf.random.normal([32, 64]), tf.random.normal([32, 10]))
    #     # model.predict(tf.random.normal([32, 64]))


        
if __name__ == '__main__':
    tf.test.main()
