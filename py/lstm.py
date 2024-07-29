import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images to the range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape the images to be sequences of pixels
# Each image is 28x28 pixels, so we reshape to (28, 28)
x_train = x_train.reshape(-1, 28*28, 1)
x_test = x_test.reshape(-1, 28*28, 1)

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"x: {x_train.shape}, y: {y_train.shape}")
print(f"x: {x_test.shape}, y: {y_test.shape}")

# Build the model
model = Sequential([
    tf.keras.layers.InputLayer(input_shape=(28*28, 1)),
    LSTM(128, return_sequences=True),
    LSTM(128),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy',
                       tf.keras.metrics.AUC(name='auc'),
                       tf.keras.metrics.FalsePositives(name='fp'),
                       tf.keras.metrics.TruePositives(name='tp'),
                       tf.keras.metrics.FalseNegatives(name='fn'),
                       tf.keras.metrics.TrueNegatives(name='tn'),
                       tf.keras.metrics.Precision(name='prc'),
                       tf.keras.metrics.Recall(name='rcl'),
                      ])
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')
