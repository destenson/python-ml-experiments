
import tensorflow as tf


# Dynamic Model that can handle 1D, 2D, 3D, and 4D output shapes.
class DynamicModel(tf.keras.Model):

    def __init__(self, output_shape=(10,), internal_activations='relu', use_bias=True, output_activation='sigmoid', name='dynamic_model', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.output_shape = output_shape
        self.output_layers = []
        if output_shape[0] is None:
            self.output_shape = output_shape[1:]
        if len(self.output_shape) == 1:
            self.output_layers.append(tf.keras.layers.Dense(
                self.output_shape[-1], activation=output_activation, use_bias=use_bias))
        elif len(self.output_shape) == 2:
            self.output_layers.append(tf.keras.layers.Dense(
                self.output_shape[-2]*self.output_shape[-1], activation=output_activation, use_bias=use_bias))
            self.output_layers.append(tf.keras.layers.Reshape(self.output_shape))
        elif len(self.output_shape) == 3:
            self.output_layers.append(tf.keras.layers.Dense(
                self.output_shape[-3]*self.output_shape[-2]*self.output_shape[-1], activation=output_activation, use_bias=use_bias))
            self.output_layers.append(tf.keras.layers.Reshape(self.output_shape))
        elif len(self.output_shape) == 4:
            self.output_layers.append(tf.keras.layers.Dense(
                self.output_shape[-4]*self.output_shape[-3]*self.output_shape[-2]*self.output_shape[-1], activation=output_activation, use_bias=use_bias))
            self.output_layers.append(tf.keras.layers.Reshape(self.output_shape))
        else:
            raise ValueError('The output shape must be 1D, 2D, 3D, or 4D')

    def call(self, inputs):
        for layer in self.output_layers:
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
        # # Passing low resolution image to model
        # super_resolution_img = self(x, training=False)
        # # Clips the tensor from min(0) to max(255)
        # super_resolution_img = tf.clip_by_value(super_resolution_img, 0, 255)
        # # Rounds the values of a tensor to the nearest integer
        # super_resolution_img = tf.round(super_resolution_img)
        # # Removes dimensions of size 1 from the shape of a tensor and converting to uint8
        # super_resolution_img = tf.squeeze(
        #     tf.cast(super_resolution_img, tf.uint8), axis=0
        # )
        # return super_resolution_img
        for layer in self.output_layers:
            x = layer(x)
        return x
    
class DynamicModelTest(tf.test.TestCase):

    def test_create_dynamic_model(self):
        model = DynamicModel(output_shape=(2,2))
        self.assertIsInstance(model, tf.keras.Model)
        model.build((None, 2, 2))
        model.summary()
    
    def test_compile_dynamic_model(self):
        model = DynamicModel(output_shape=(1,))
        model.compile(optimizer='adam', loss='mse')
        model.build((None, 1))
        model.summary()

    def test_train_dynamic_model(self):
        model = DynamicModel(output_shape=(1,))
        model.compile(optimizer='adam', loss='mse')
        # model.build((None, 1))
        model.fit(tf.random.normal((100, 1)), tf.random.normal((100, 1)), epochs=10)
        model.summary()
        print(f'{model.get_weights()}')

if __name__ == '__main__':
    tf.test.main()   
    
#
            