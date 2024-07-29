
import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.keras.models import Model # type: ignore

import numpy as np
import pandas as pd

# tfk = tf.keras
# tfkl = tf.keras.layers
# tfpl = tfp.layers
# tfd = tfp.distributions

def cifar10_images(batch_size=32):
    datasets, datasets_info = tfds.load(name='cifar10', with_info=True, as_supervised=False)
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

def infinite_cifar10_generator(batch_size=32):
    # Load the CIFAR10 dataset
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    print(f"x_train.shape = {x_train.shape}")
    print(f"y_train.shape = {y_train.shape}")

    # Normalize the data
    x_train = x_train.astype('float32') / 255.0

    # Reshape the data
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)

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

class Cifar10DataTest(tf.test.TestCase):
    def test_cifar10_images(self):
        train_dataset, eval_dataset = cifar10_images()
        self.assertIsInstance(train_dataset, tf.data.Dataset)
        self.assertIsInstance(eval_dataset, tf.data.Dataset)
        
    def test_infinite_cifar10_generator(self):
        generator = infinite_cifar10_generator(batch_size=64)
        for _ in range(5):
            batch_x, batch_y = next(generator)
            self.assertEqual(batch_x.shape, (64, 32, 32, 3))
            self.assertEqual(batch_y.shape, (64, 10))

def load_cfar10_densenet(path='cifar10_densenet_model.94.keras'):
    return tf.keras.models.load_model(path)


# # load the CIFAR10 data
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

# # input image dimensions
# input_shape = x_train.shape[1:]

# # mormalize data
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
# print('y_train shape:', y_train.shape)

# # convert class vectors to binary class matrices.
# y_train = to_categorical(y_train, num_classes)
# y_test = to_categorical(y_test, num_classes)

if __name__ == '__main__':
    tf.test.main()
