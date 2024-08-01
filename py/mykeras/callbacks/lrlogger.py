import tensorflow as tf
import keras
import numpy as np
import pandas as pd

def get_custom_objects():
    return {
        "MyKeras>LearningRateLogger": LearningRateLogger,
    }

@keras.utils.register_keras_serializable(package="MyKeras")
class LearningRateLogger(tf.keras.callbacks.Callback):
    '''
    Callback to log the learning rate at the end of each epoch.
    '''
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations) # should this be epoch?
        print(f"Epoch {epoch+1}: Learning rate is {tf.keras.backend.get_value(lr)}") if self.verbose else None
    
    def get_config(self):
        return {
            'verbose': self.verbose,
        }
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class LearningRateLoggerTest(tf.test.TestCase):
    def test_learning_rate_logger(self):
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train.reshape(-1, 784).astype('float32') / 255
        X_test = X_test.reshape(-1, 784).astype('float32') / 255
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(784,)),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(10, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit(X_train, y_train, epochs=2, callbacks=[LearningRateLogger()])
        model.evaluate(X_test, y_test)
        model.predict(X_test)
        model.summary()
        model.save("model.keras")
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model("model.keras", custom_objects=get_custom_objects())
        model.summary()
        # raise ValueError("Done")

if __name__ == "__main__":
    tf.test.main()
