import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.keras.models import Model

import numpy as np

import pandas as pd

# tfk = tf.keras
# tfkl = tf.keras.layers
# tfpl = tfp.layers
# tfd = tfp.distributions

def mnist_digits(batch_size=32):
    datasets, datasets_info = tfds.load(name='mnist', with_info=True, as_supervised=False)
    def _preprocess(sample):
        image = tf.cast(sample['image'], tf.float32) / 255.0  # Scale to unit interval.
        return image, image < tf.random.uniform(tf.shape(image))

    assert 'train' in datasets
    assert 'test' in datasets

    train_dataset = (datasets['train']
                    .map(_preprocess)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE)
                    .shuffle(int(10e3)))

    eval_dataset = (datasets['test']
                    .map(_preprocess)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE))
    
    return train_dataset, eval_dataset

def mnist_fashion():
    datasets, datasets_info = tfds.load(name='fashion_mnist', with_info=True, as_supervised=False)
    def _preprocess(sample):
        image = tf.cast(sample['image'], tf.float32) / 255.0  # Scale to unit interval.
        return image, image < tf.random.uniform(tf.shape(image))

    assert 'train' in datasets
    assert 'test' in datasets

    train_dataset = (datasets['train']
                    .map(_preprocess)
                    .batch(256)
                    .prefetch(tf.data.AUTOTUNE)
                    .shuffle(int(10e3)))

    eval_dataset = (datasets['test']
                    .map(_preprocess)
                    .batch(256)
                    .prefetch(tf.data.AUTOTUNE))
    
    return train_dataset, eval_dataset

def mnist_data():
    return {
        "digits": mnist_digits(),
        "fashion": mnist_fashion(),
    }

# def get_custom_objects():
#     return {
#         "MyKeras>MNISTDigits": mnist_digits,
#         "MyKeras>MNISTFashion": mnist_fashion,
#         "MyKeras>MNISTData": mnist_data,
#     }

def infinite_mnist_generator(batch_size=32):
    # Load the MNIST dataset
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    # Normalize the data
    x_train = x_train.astype('float32') / 255.0

    # Reshape the data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    # Calculate the number of batches per epoch
    batches_per_epoch = x_train.shape[0] // batch_size

    while True:
        # Shuffle the dataset at the start of each epoch
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]

        for i in range(batches_per_epoch):
            batch_x = x_shuffled[i*batch_size:(i+1)*batch_size]
            batch_y = y_shuffled[i*batch_size:(i+1)*batch_size]
            yield batch_x, batch_y

def test_generator():
    # Usage example
    generator = infinite_mnist_generator(batch_size=64)

    # Get batches from the generator
    for _ in range(5):  # Get 5 batches as an example
        batch_x, batch_y = next(generator)
        print(f"Batch shape: {batch_x.shape}, {batch_y.shape}")


class MnistDataTest(tf.test.TestCase):
    def test_mnist_digits(self):
        train_dataset, eval_dataset = mnist_digits()
        self.assertIsInstance(train_dataset, tf.data.Dataset)
        self.assertIsInstance(eval_dataset, tf.data.Dataset)
        
    def test_mnist_fashion(self):
        train_dataset, eval_dataset = mnist_fashion()
        self.assertIsInstance(train_dataset, tf.data.Dataset)
        self.assertIsInstance(eval_dataset, tf.data.Dataset)
    
    def test_mnist_data(self):
        data = mnist_data()
        self.assertIsInstance(data, dict)
        self.assertIn("digits", data)
        self.assertIn("fashion", data)
        self.assertIsInstance(data["digits"], tuple)
        self.assertIsInstance(data["fashion"], tuple)
        self.assertIsInstance(data["digits"][0], tf.data.Dataset)
        self.assertIsInstance(data["digits"][1], tf.data.Dataset)
        self.assertIsInstance(data["fashion"][0], tf.data.Dataset)
        self.assertIsInstance(data["fashion"][1], tf.data.Dataset)

    def test_infinite_mnist_generator(self):
        generator = infinite_mnist_generator(batch_size=64)
        for _ in range(5):
            batch_x, batch_y = next(generator)
            self.assertEqual(batch_x.shape, (64, 28, 28, 1))
            self.assertEqual(batch_y.shape, (64, 10))

    def test_generator(self):
        test_generator()

def load_mnist_cnn(path='mnist_cnn-80k-99.1.keras'):
    return tf.keras.models.load_model(path)

def plot_first_n(n, x, y, model, title, image_size=28):
    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(x[i].reshape(image_size, image_size), cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()
    y_pred = tf.argmax(model.predict(x[:n]), axis=1)
    y_act = tf.argmax(y[:n], axis=1)
    print(f"y_pred: {y_pred}  y_act: {y_act}  match: {y_pred == y_act}")

import matplotlib.pyplot as plt

def plot_conv_weights(weights, title):
    weight_slice = weights[:, :, 0, :]
    num_filters = weight_slice.shape[2]
    grid_size = int(np.ceil(np.sqrt(num_filters)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    fig.suptitle(title)
    
    for i in range(num_filters):
        ax = axes[i // grid_size, i % grid_size]
        ax.imshow(weight_slice[:, :, i], cmap='viridis')
        ax.axis('off')
    
    plt.show()

class MnistCnnModelTest(tf.test.TestCase):
    def test_open_model_file(self):
        model = load_mnist_cnn()
        self.assertIsInstance(model, tf.keras.Model)
        print(model.summary())
    
    def test_verify_results(self):
        model = load_mnist_cnn()
        # self.verify_results(model, mnist_digits(batch_size=64))
        # self.verify_results(model, infinite_mnist_generator(batch_size=64))
        
    def analyze_variables(self):
        model = load_mnist_cnn('cnn-functional-99.3.keras')
        for layer in model.layers:
            if hasattr(layer, 'weights') and len(layer.weights) > 0:
                weights = layer.weights
                for i, weight in enumerate(weights):
                    print(f"Layer: {layer.name}, Weight {i}")
                    print(f"  Shape: {weight.shape}")
                    print(f"  Mean: {weight.numpy().mean()}")
                    print(f"  Std: {weight.numpy().std()}")
                    print(f"  Min: {weight.numpy().min()}")
                    print(f"  Max: {weight.numpy().max()}")

        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                weights = layer.get_weights()[0]
                plot_conv_weights(weights, f"Filters of {layer.name}")
                
    def plot_activations(self):
        model = load_mnist_cnn('cnn-functional-99.3.keras')
                
        # Create a model that outputs activations of all layers
        layer_outputs = [layer.output for layer in model.layers]
        activation_model = Model(inputs=model.input, outputs=layer_outputs)

        # Get activations for a single image
        img = x_test[0:1]  # Select first test image
        activations = activation_model.predict(img)

        # Plot activations
        for layer_name, layer_activation in zip(layer_names, activations):
            if len(layer_activation.shape) == 4:
                n_features = layer_activation.shape[-1]
                size = layer_activation.shape[1]
                
                n_cols = n_features // 16
                display_grid = np.zeros((size * n_cols, size * 16))
                
                for col in range(n_cols):
                    for row in range(16):
                        channel_image = layer_activation[0, :, :, col * 16 + row]
                        display_grid[col * size : (col + 1) * size, 
                                    row * size : (row + 1) * size] = channel_image

                scale = 1. / display_grid.max()
                plt.figure(figsize=(20, 10))
                plt.title(layer_name)
                plt.grid(False)
                plt.imshow(display_grid * scale, aspect='auto', cmap='viridis')
                plt.show()


    # def verify_results(self, model, data):
    #     # _train_dataset, eval_dataset = data
    #     # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #     # model.fit(train_dataset, epochs=1)
    #     # model.evaluate(eval_dataset.take(20))
    #     # generator = infinite_mnist_generator(batch_size=64)
    #     if not isinstance(data, tuple):
    #         test_x, test_y = next(data)
    #     else:
    #         # print(f"data: {data}")
    #         test_x, test_y = data
    #         test_x = pd.DataFrame(test_x.take(10))
    #         test_y = pd.DataFrame(test_y.take(10))
    #         print(f"Test data: {test_x}, {test_y}")
    #     print(f"Test data shape: {test_x.shape}, {test_y.shape}")
    #     self.assertEqual(test_x.shape, (10, 28, 28, 1))
    #     self.assertEqual(test_y.shape, (10, 10))
    #     print(f"{tf.argmax(model(test_x))} = {tf.argmax(test_y)}")
        
    #     plot_first_n(10, test_x, test_y, model, "MNIST digits")


if __name__ == "__main__":
    if True:
        tf.test.main()
    else:
    # print(tf.__version__)
        model = load_mnist_cnn()
        print(model.summary())
        
        verify_results(model, mnist_digits(batch_size=10))
        # verify_results(model, infinite_mnist_generator(batch_size=10))
    
