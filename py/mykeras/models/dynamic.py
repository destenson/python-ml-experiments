
import tensorflow as tf
import keras
import numpy as np
import pandas as pd


def get_custom_objects():
    return {
        "MyKeras>DynamicSequentialModel": DynamicSequentialModel,
    }

# Dynamic sequential model that can handle 1D, 2D, 3D, and 4D output shapes.
@keras.utils.register_keras_serializable(package="MyKeras")
class DynamicSequentialModel(tf.keras.Model):

    def __init__(self, output_shape=(10,), use_bias=True,
                 output_activation='sigmoid', name='dynamic_model',
                 output_layer=None, dynamic_layers: list=[],
                 *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

        self.output_shape = output_shape
        self.dynamic_layers = dynamic_layers

        if output_shape[0] is None:
            self.output_shape = output_shape[1:]
        self.output_layer = output_layer
        if self.output_layer is not None:
            self.output_shape = self.output_layer.output_shape
        else:
            if len(self.output_shape) == 1:
                self.output_layer = tf.keras.layers.Dense(
                    self.output_shape[-1], activation=output_activation, use_bias=use_bias))
            elif len(self.output_shape) == 2:
                self.output_layer = (tf.keras.layers.Dense(
                    self.output_shape[-2]*self.output_shape[-1], activation=output_activation, use_bias=use_bias))
                self.output_layer = (tf.keras.layers.Reshape(self.output_shape))
            elif len(self.output_shape) == 3:
                self.output_layer = (tf.keras.layers.Dense(
                    self.output_shape[-3]*self.output_shape[-2]*self.output_shape[-1], activation=output_activation, use_bias=use_bias))
                self.output_layer = (tf.keras.layers.Reshape(self.output_shape))
            elif len(self.output_shape) == 4:
                self.output_layer = (tf.keras.layers.Dense(
                    self.output_shape[-4]*self.output_shape[-3]*self.output_shape[-2]*self.output_shape[-1], activation=output_activation, use_bias=use_bias))
                self.output_layer = (tf.keras.layers.Reshape(self.output_shape))
            else: # pehaps we can support any output shape
                self.output_layer = (tf.keras.layers.Dense(
                    tf.reduce_prod(self.output_shape), activation=output_activation, use_bias=use_bias))
                self.output_layer = (tf.keras.layers.Reshape(self.output_shape))

    def call(self, inputs):
        for layer in self.get_layers():
            inputs = layer(inputs)
        return inputs

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        return tf.TensorShape((input_shape[0], *self.output_shape))
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # TODO: consider modifification of gradients as needed
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, inputs):
        # Adding dummy dimension using tf.expand_dims and converting to float32 using tf.cast
        x = tf.cast(tf.expand_dims(inputs, axis=0), tf.float32)
        for layer in self.get_layers():
            x = layer(x)
        return x

    def get_layers(self):
        layers = self.dynamic_layers[:]
        layers.append(self.output_layer)
        return layers
    
    def add_layer(self,
                  units = None, # None uses the same number of units as the previous layer's output
                  activation='relu',
                  use_bias = True,
                  kernel_regularizer = None,
                  bias_regularizer = None,
                  activity_regularizer = None,
                  kernel_constraint = None,
                  bias_constraint = None,
                  lora_rank = None):
        layer = tf.keras.layers.Dense(
            self.get_layers()[-1].output_shape,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer='identity',
            bias_initializer='zeros',
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            lora_rank=lora_rank
            )
        dynamic_layers = self.dynamic_layers[:]
        dynamic_layers.append(layer)
        return DynamicSequentialModel(output_layer=self.output_layer,
                                      dynamic_layers=dynamic_layers)
    
class DynamicModelTest(tf.test.TestCase):

    def test_create_dynamic_model(self):
        model = DynamicSequentialModel(output_shape=(2,2))
        self.assertIsInstance(model, tf.keras.Model)
        model.build((None, 2, 2))
        model.summary()
    
    def test_compile_dynamic_model(self):
        model = DynamicSequentialModel(output_shape=(1,))
        model.compile(optimizer='adam', loss='mse')
        model.build((None, 1))
        model.summary()

    def test_train_dynamic_model(self):
        model = DynamicSequentialModel(output_shape=(1,))
        model.compile(optimizer='adam', loss='mse')
        # model.build((None, 1))
        model.fit(tf.random.normal((100, 1)), tf.random.normal((100, 1)), epochs=10)
        model.summary()
        print(f'{model.get_weights()}')

## Dynamic layers (which add or remove weights) are not supported in keras.
## The following code doesn't really work.

# @keras.utils.register_keras_serializable(package="MyKeras")
# class DynamicOutputLayer(tf.keras.layers.Layer):
#     def __init__(self, n=64, initial_classes=10,
#                  kernel_initializer='random_normal', bias_initializer='zeros',
#                  kernel_regularizer=None, bias_regularizer=None, **kwargs):
#         super(DynamicOutputLayer, self).__init__(**kwargs)
#         self.n = n
#         self.initial_classes = initial_classes
#         self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
#         self.bias_initializer = tf.keras.initializers.get(bias_initializer)
#         self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
#         self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
#         self.W = None
#         self.b = None
    
#     def build(self, input_shape):
#         n = self.n
#         initial_classes = self.initial_classes
#         self.W = self.add_weight(
#             shape=(n*input_shape[-1], initial_classes),
#             initializer=self.kernel_initializer,
#             regularizer=self.kernel_regularizer,
#             trainable=True)
#         self.b = self.add_weight(
#             shape=(initial_classes,),
#             initializer=self.bias_initializer,
#             regularizer=self.bias_regularizer,
#             trainable=True)
        
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "n": self.n,
#             "initial_classes": self.initial_classes,
#             "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
#             "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
#             "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),
#             "bias_regularizer": tf.keras.regularizers.serialize(self.bias_regularizer),
#         })
#         return config
    
#     @classmethod
#     def from_config(cls, config):
#         self = cls(**config)
#         self.initial_classes = config["initial_classes"]
#         self.n = config["n"]
#         self.kernel_initializer = tf.keras.initializers.deserialize(config["kernel_initializer"])
#         self.kernel_regularizer = tf.keras.regularizers.deserialize(config["kernel_regularizer"])
#         self.bias_initializer = tf.keras.initializers.deserialize(config["bias_initializer"])
#         self.bias_regularizer = tf.keras.regularizers.deserialize(config["bias_regularizer"])
#         return self
    
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.b.shape[0])

#     def add_class(self):
#         classes = self.W.shape[1]
#         new_w = self.add_weight(
#             shape=(self.W.shape[0], classes + 1),
#             initializer=self.kernel_initializer,
#             regularizer=self.kernel_regularizer,
#             trainable=True
#         )
#         new_b = self.add_weight(
#             shape=(classes + 1,),
#             initializer=self.bias_initializer,
#             regularizer=self.bias_regularizer,
#             trainable=True)
        
#         # Copy over old values
#         new_w[:, :classes].assign(self.W)
#         new_b[:classes].assign(self.b)
        
#         # Replace old variables
#         self.W = new_w
#         self.b = new_b
        
#     # @tf.function(reduce_retracing=True)
#     def call(self, inputs):
#         input = tf.reshape(inputs, [-1, self.W.shape[0]])
#         # print(f"inputs.shape={inputs.shape}, input.shape={input.shape}, W.shape={self.W.shape}, b.shape={self.b.shape}")
#         return tf.matmul(input, self.W) + self.b


# def build_dynamic_model(input_shape=(28, 28, 1), n=10, initial_classes=5):
#     inputs = tf.keras.layers.Input(shape=input_shape)
#     x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
#     x = tf.keras.layers.MaxPooling2D((2, 2))(x)
#     x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
#     x = tf.keras.layers.MaxPooling2D((2, 2))(x)
#     x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
#     x = tf.keras.layers.MaxPooling2D((2, 2))(x)
#     x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(x)
#     x = tf.keras.layers.MaxPooling2D((2, 2))(x)
#     x = tf.keras.layers.Conv2D(4, (3, 3), activation='relu')(x)
#     x = tf.keras.layers.MaxPooling2D((2, 2))(x)
#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(n*2, activation='relu')(x)
#     outputs = DynamicOutputLayer(n=n, initial_classes=initial_classes)(x)
    
#     model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
#     model.compile(optimizer='adam', loss='binary_crossentropy',
#                   metrics=['accuracy',
#                            tf.keras.metrics.AUC(name='auc'),
#                            tf.keras.metrics.FalsePositives(name='fp'),
#                            tf.keras.metrics.TruePositives(name='tp'),
#                            tf.keras.metrics.FalseNegatives(name='fn'),
#                            tf.keras.metrics.TrueNegatives(name='tn'),
#                            tf.keras.metrics.Precision(name='prc'),
#                            tf.keras.metrics.Recall(name='rcl'),
#                           ])
#     return model

# model = build_dynamic_model()

# class DynamicOutputLayerTest(tf.test.TestCase):

#     # def test_dynamic_output_layer_0(self):
#     #     layer = DynamicOutputLayer(n=2, initial_classes=10)
#     #     layer.build(input_shape=(28, 28, 15))
#     #     self.assertEqual(layer.W.shape, (30, 10))
#     #     self.assertEqual(layer.b.shape, (10,))
#     #     layer.add_class()
#     #     self.assertEqual(layer.W.shape, (30, 11))
#     #     self.assertEqual(layer.b.shape, (11,))

#     # def test_dynamic_output_layer_1(self):
#     #     layer = DynamicOutputLayer(n=128, initial_classes=6)
#     #     layer.build(input_shape=(28, 1))
#     #     self.assertEqual(layer.W.shape, (128, 6))
#     #     self.assertEqual(layer.b.shape, (6,))
#     #     layer.add_class()
#     #     self.assertEqual(layer.W.shape, (128, 7))
#     #     self.assertEqual(layer.b.shape, (7,))

#     # def test_dynamic_output_layer_2(self):
#     #     layer = DynamicOutputLayer(n=64, initial_classes=7)
#     #     layer.build(input_shape=(32, 2))
#     #     self.assertEqual(layer.W.shape, (128, 7))
#     #     self.assertEqual(layer.b.shape, (7,))
#     #     layer.add_class()
#     #     self.assertEqual(layer.W.shape, (128, 8))
#     #     self.assertEqual(layer.b.shape, (8,))

#     # def test_dynamic_output_layer_serialization(self):
#     #     layer = DynamicOutputLayer(n=64, initial_classes=10)
#     #     config = layer.get_config()
#     #     new_layer = DynamicOutputLayer.from_config(config)
#     #     self.assertAllEqual(layer.W, new_layer.W)
#     #     self.assertAllEqual(layer.b, new_layer.b)
    
#     # def test_dynamic_output_layer_add_class(self):
#     #     layer = DynamicOutputLayer(n=64, initial_classes=10)
#     #     layer.build(input_shape=(32, 1))
#     #     layer.add_class()
#     #     self.assertEqual(layer.W.shape, (64, 11))
#     #     self.assertEqual(layer.b.shape, (11,))
#     #     self.assertEqual((32, 11), layer.compute_output_shape((32, 1)))
#     #     layer.add_class()
#     #     self.assertEqual(layer.W.shape, (64, 12))
#     #     self.assertEqual(layer.b.shape, (12,))
#     #     self.assertEqual((32, 12), layer.compute_output_shape((32, 1)))
    
#     # def test_dynamic_output_layer_call(self):
#     #     layer = DynamicOutputLayer(n=64, initial_classes=10)
#     #     layer.build(input_shape=(None, 64, 1))
#     #     inputs = tf.random.normal([32, 64, 1])
#     #     outputs = layer(inputs)
#     #     self.assertEqual(outputs.shape, (32, 10))
    
#     # def test_model_with_dynamic_output(self):
#     #     model = tf.keras.Sequential([
#     #         tf.keras.layers.Input(shape=(16, 7)),
#     #         DynamicOutputLayer(n=16, initial_classes=5),
#     #     ])
#     #     model.compile(optimizer='adam', loss='mse')
#     #     # model.summary()
#     #     model.fit(tf.random.normal([39, 16, 7]), tf.random.normal([39, 5]), epochs=1)
#     #     model.evaluate(tf.random.normal([32, 16, 7]), tf.random.normal([32, 5]))
#     #     model.predict(tf.random.normal([32, 16, 7]))
#     #     model.save('test_model0.keras')
#     #     model = tf.keras.models.load_model('test_model0.keras', custom_objects=get_custom_objects())
#     #     # model.summary()
#     #     model.evaluate(tf.random.normal([15, 16, 7]), tf.random.normal([15, 5]))
#     #     model.predict(tf.random.normal([15, 16, 7]))
#     #     model.summary()

#     # def test_build_dynamic_model(self):
#     #     model = build_dynamic_model((1024, 768, 3), n=1, initial_classes=5)
#     #     # model.summary()
#     #     model.fit(tf.random.normal([2, 1024, 768, 3]), tf.random.normal([2, 5]), epochs=1)
#     #     model.evaluate(tf.random.normal([4, 1024, 768, 3]), tf.random.normal([4, 5]))
#     #     model.predict(tf.random.normal([1, 1024, 768, 3]))
#     #     # model.summary()
#     #     model.save('test_model1.keras')
#     #     model = tf.keras.models.load_model('test_model1.keras', custom_objects=get_custom_objects())
#     #     # model.summary()
#     #     model.evaluate(tf.random.normal([1, 1024, 768, 3]), tf.random.normal([1, 5]))
#     #     model.predict(tf.random.normal([1, 1024, 768, 3]))
#     #     model.summary()

#     # def test_build_dynamic_model1(self):
#     #     model = build_dynamic_model((1024, 768, 3), n=1, initial_classes=5)
#     #     initial_output = model.predict(tf.random.normal([1, 1024, 768, 3]))
#     #     self.assertEqual(initial_output.shape[1], 5)
#     #     model.summary()
        
#     #     model.layers[-1].add_class()
#     #     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
#     #     new_output = model.predict(tf.random.normal([1, 1024, 768, 3]))
#     #     self.assertEqual(initial_output.shape[0], new_output.shape[0])
#     #     self.assertEqual(new_output.shape[1], 6)
#     #     self.assertEqual(model.layers[-1].compute_output_shape((1, 1024, 768, 3))[1], 6)
#     #     model.summary()

#     # def test_build_dynamic_model2(self):
#     #     model = build_dynamic_model((1024, 768, 3), n=1, initial_classes=5)
#     #     model.summary()
#     #     model.fit(tf.random.normal([2, 1024, 768, 3]), tf.random.normal([2, 5]), epochs=1)
#     #     model.evaluate(tf.random.normal([4, 1024, 768, 3]), tf.random.normal([4, 5]))
#     #     initial_output = model.predict(tf.random.normal([1, 1024, 768, 3]))
#     #     self.assertEqual(initial_output.shape, (1, 5))

#     #     model.layers[-1].add_class()
#     #     model.compile(optimizer='adam', loss='binary_crossentropy',
#     #                 metrics=['accuracy',
#     #                         tf.keras.metrics.AUC(name='auc'),
#     #                         tf.keras.metrics.FalsePositives(name='fp'),
#     #                         tf.keras.metrics.TruePositives(name='tp'),
#     #                         tf.keras.metrics.FalseNegatives(name='fn'),
#     #                         tf.keras.metrics.TrueNegatives(name='tn'),
#     #                         tf.keras.metrics.Precision(name='prc'),
#     #                         tf.keras.metrics.Recall(name='rcl'),
#     #                         ])
#     #     model.summary()
#     #     model.fit(tf.random.normal([2, 1024, 768, 3]), tf.random.normal([2, 6]), epochs=1)
#     #     model.evaluate(tf.random.normal([1, 1024, 768, 3]), tf.random.normal([1, 6]))
#     #     new_output = model.predict(tf.random.normal([1, 1024, 768, 3]))
#     #     self.assertEqual(new_output.shape, (1, 6))
#     #     self.assertEqual(model.layers[-1].compute_output_shape((1, 1024, 768, 3))[1], 6)
#     #     model.summary()


if __name__ == '__main__':
    tf.test.main()   
    
#
            