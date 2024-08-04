import keras.api
import tensorflow as tf
import keras
# from tensorflow.keras.optimizers import Optimizer
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.datasets import mnist
from keras.api.optimizers import Optimizer
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.datasets import mnist

# from tensorflow.keras.optimizers import Optimizer
from keras import backend as K

class CustomOptimizer(Optimizer):
    def __init__(self, shape, learning_rate=0.001, decay_rate=0.01, name="CustomOptimizer", **kwargs):
        super().__init__(learning_rate, name=name, **kwargs)
        self.decay_rate = self.add_variable(shape=(),
            name="decay_rate", initializer='zeros', dtype=tf.float32)
        self.iterations = {}
        self.hessian_inv = {}
        self.grad_sum_squares = {}
        self.grad_mean = {}

    def build(self, vars) -> None:
        super().build(vars)
        print(f"CustomOptimizer.build({vars})")
        for var in vars:
            print(f"CustomOptimizer.build var = {var._shared_name}")
            print(f"type(var) = {type(var)}")
            # assert isinstance(var, keras.ResourceVariable)
            assert 'path' in var.__dict__, "Variable does not have 'path' attribute"
            # print(f"CustomOptimizer.build var = {var}")
            path = var._shared_name.replace("/", "_")
            self.iterations[path] = self.add_variable(var.shape,name=f"{path}_iterations")
            self.hessian_inv[path] = self.add_variable(var.shape, name=f"{path}_hessian_inv")
            self.grad_sum_squares[path] = self.add_variable(var.shape, name=f"{path}_grad_sum_squares")
            self.grad_mean[path] = self.add_variable(var.shape, name=f"{path}_grad_mean")

    @tf.function
    def update_step(self, grad, var, apply_state=None):
        lr_t = self.learning_rate
        decay_t = self.decay_rate
        
        path = var._shared_name.replace("/", "_")
        self.iterations[path].assign_add(1)
        iter_t = self.iterations[path]

        hessian_inv = self.hessian_inv[path]
        grad_sum_squares = self.grad_sum_squares[path]
        grad_mean = self.grad_mean[path]

        grad_squared = tf.square(grad)
        grad_sum_squares.assign_add(grad_squared)
        grad_mean.assign_add(grad)

        sum_grad_squares = grad_sum_squares / iter_t
        mean_grad = grad_mean / iter_t

        grad_adjustment = sum_grad_squares / (mean_grad + K.epsilon())
        hessian_inv_update = (1 / (1 + decay_t * iter_t)) * hessian_inv + grad_adjustment

        hessian_inv.assign(hessian_inv_update)
        
        hessian_inv_regularized = hessian_inv + tf.eye(tf.shape(hessian_inv)[0]) * K.epsilon()
        var_update = var - lr_t * tf.linalg.solve(hessian_inv_regularized, tf.expand_dims(grad, -1))[:, 0]
        var.assign(var_update)

        return var_update

    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay_rate": self._serialize_hyperparameter("decay_rate"),
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomOptimizerTest(tf.test.TestCase):
    def test_custom_optimizer(self):

        # Example usage
        model = Sequential([
            Dense(10, activation='relu', input_shape=(784,)),
            Dense(10, activation='softmax')
        ])
        shape = model.layers[0].weights[0].shape
        optimizer = CustomOptimizer(shape, learning_rate=0.001, decay_rate=0.01)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                      metrics=['accuracy'], run_eagerly=True)

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(60000, 784).astype('float32') / 255
        # x_test = x_test.reshape(10000, 784).astype('float32') / 255
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_train, 10)

        print(f"X_train: {x_train.shape}, Y_train: {y_train.shape}")

        results = model.fit(x_train, y_train, validation_split=0.2, epochs=10)
        print(f"Results: {results}")

        # Test the optimizer
        var = tf.Variable([1.0, 2.0, 3.0])
        grad = tf.Variable([0.1, 0.2, 0.3])
        optimizer.apply_gradients(zip([grad], [var]))
        
        

if __name__ == "__main__":
    tf.test.main()


#
