import tensorflow as tf
import keras

def get_custom_objects():
    return {
        "MyKeras>LearningRateLogger": LearningRateLogger,
    }

@keras.utils.register_keras_serializable(package="MyKeras")
class LearningRateLogger(tf.keras.callbacks.Callback):
    '''
    Callback to log the learning rate at the end of each epoch.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations) # should this be epoch?
        print(f"Epoch {epoch+1}: Learning rate is {tf.keras.backend.get_value(lr)}")
    
    def get_config(self):
        return {}

# Example usage
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
# model.compile(optimizer='adam', loss='binary_crossentropy')
# model.fit(X_train, y_train, epochs=10, callbacks=[LearningRateLogger()])
# model.evaluate(X_test, y_test)
# model.predict(X_test)
# model.summary()
# model.save("model.h5")
# model = tf.keras.models.load_model("model.h5", custom_objects=get_custom_objects())

if __name__ == "__main__":
    pass
