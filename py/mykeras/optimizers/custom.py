import tensorflow as tf
import keras
from keras.optimizers import Optimizer

class CustomOptimizer(Optimizer):
    def __init__(self, learning_rate=0.001, decay_rate=0.01,
                 name="CustomOptimizer", **kwargs):
        super().__init__(learning_rate, name=name, **kwargs)
        self.decay_rate = decay_rate
        self.iterations = self.add_weight(
            name="iterations", initializer="zeros", dtype=tf.int32)
        
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "hessian_inv")
            self.add_slot(var, "grad_sum_squares")
            self.add_slot(var, "grad_mean")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self.learning_rate
        decay_t = self.decay_rate
        iter_t = tf.cast(self.iterations + 1, var_dtype)

        hessian_inv = self.get_slot(var, "hessian_inv")
        grad_sum_squares = self.get_slot(var, "grad_sum_squares")
        grad_mean = self.get_slot(var, "grad_mean")

        grad_squared = tf.square(grad)
        grad_sum_squares.assign_add(grad_squared)
        grad_mean.assign_add(grad)

        sum_grad_squares = grad_sum_squares / iter_t
        mean_grad = grad_mean / iter_t

        grad_adjustment = sum_grad_squares / (mean_grad + 1e-8)
        hessian_inv_update = (1 / (1 + decay_t * iter_t)) * hessian_inv + grad_adjustment

        hessian_inv.assign(hessian_inv_update)
        
        var_update = var - lr_t * tf.linalg.inv(
            hessian_inv + tf.eye(tf.shape(hessian_inv)[0]) * 1e-8) @ grad
        var.assign(var_update)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self.learning_rate,
            "decay_rate": self.decay_rate,
        })
        return config


class CustomOptimizerTest(tf.test.TestCase):
    def test_custom_optimizer(self):
        
        # Example usage
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        optimizer = CustomOptimizer(learning_rate=0.001, decay_rate=0.01)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(60000, 784).astype('float32') / 255
        x_test = x_test.reshape(10000, 784).astype('float32') / 255
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_train, 10)

        # Assuming you have training data in `x_train` and `y_train`
        results = model.fit(x_train, y_train, epochs=10, return_dict=True)

        # Test the optimizer
        var = tf.Variable([1.0, 2.0, 3.0])
        grad = tf.Variable([0.1, 0.2, 0.3])
        optimizer.apply_gradients(zip([grad], [var]))
        
        

if __name__ == "__main__":
    tf.test.main()


#
