
import tensorflow as tf

from tensorflow.keras.layers import Conv1D, Conv2D # type: ignore
from tensorflow.keras.models import Model # type: ignore

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_loss_histogram(model, x):
    '''
    Plot histogram of loss values for the given model and input data.
    This is useful for determining a threshold for anomaly detection.
    This is used with autoencoders.
    '''
    # Get train MAE loss.
    x_pred = model.predict(x)
    mae_loss = np.mean(tf.square(x_pred - x), axis=1)
    print("MAE loss shape: ", mae_loss.shape)
    if mae_loss.shape[-1] == 1:
        mae_loss = mae_loss.flatten()

    plt.hist(mae_loss, bins=50)
    plt.xlabel("MAE loss")
    plt.ylabel("No of samples")
    plt.show()

    # Get loss threshold.
    threshold = np.max(mae_loss)
    print("Error threshold: ", threshold)

def plot_history_metrics(history: tf.keras.callbacks.History):
    total_plots = len(history.history)
    cols = total_plots // 2

    rows = total_plots // cols

    if total_plots % cols != 0:
        rows += 1

    pos = range(1, total_plots + 1)
    plt.figure(figsize=(15, 10))
    for i, (key, value) in enumerate(history.history.items()):
        plt.subplot(rows, cols, pos[i])
        plt.plot(range(len(value)), value)
        plt.title(str(key))
    plt.show()

def plot_convolutional_weights(weights, title, two_d=True):
    # if two_d:
    if len(weights.shape) == 4:
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
    else:
        fig = plt.figure(figsize=(20, 5))
        plt.plot(weights, marker='o', markersize=5)
        plt.title(title)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.show()

def print_basic_weight_statistics(model, show_conv=False):
    for layer in model.layers:
        if hasattr(layer, 'weights'):
            weights = layer.weights
            for i, weight in enumerate(weights):
                print(f"Layer: {layer.name}, Weight {i}")
                print(f"  Shape: {weight.shape}")
                print(f"  Mean: {weight.numpy().mean()}")
                print(f"  Std: {weight.numpy().std()}")
                print(f"  Min: {weight.numpy().min()}")
                print(f"  Max: {weight.numpy().max()}")
                if show_conv:
                    if isinstance(layer, Conv2D):
                        plot_convolutional_weights(weight, f"Filters of {layer.name}", two_d=True)
                    elif isinstance(layer, Conv1D):
                        plot_convolutional_weights(weight, f"Filters of {layer.name}", two_d=False)
    print(f"There were {len(model.layers)} layers in total.")

def visualize_activations(model, x_test, n=0):
    # Ensure the model is called with some data
    model.predict(x_test[0:1])
    
    for layer in model.layers:
        print(layer.name, layer.input, layer.output)

    # Get layer names
    layer_names = [layer.name for layer in model.layers]

    # Create a list of layer outputs, excluding the input layer
    layer_outputs = [layer.output for layer in model.layers[1:]]

    # Create a model that will return these outputs, given the model input
    activation_model = tf.keras.Model(inputs=model.layers[0].input, outputs=layer_outputs)

    # Get activations for a single image
    img = x_test[n:n+1]  # Select first test image
    activations = activation_model.predict(img)

    # Plot activations
    for layer_name, layer_activation in zip(layer_names[1:], activations):
        if len(layer_activation.shape) == 4:  # Only for convolutional layers
            n_features = layer_activation.shape[-1]
            size = layer_activation.shape[1]
            
            n_cols = min(n_features, 16)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            display_grid = np.zeros((size * n_rows, size * n_cols))
            
            for i in range(n_features):
                row = i // n_cols
                col = i % n_cols
                channel_image = layer_activation[0, :, :, i]
                display_grid[row * size : (row + 1) * size, 
                             col * size : (col + 1) * size] = channel_image

            scale = 1. / display_grid.max()
            plt.figure(figsize=(20, 10))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid * scale, aspect='auto', cmap='viridis')
            plt.show()


def generate_saliency_map(model, input_image, class_idx):
    input_image = tf.Variable(input_image, dtype=float)
    
    with tf.GradientTape() as tape:
        pred = model(input_image)
        class_channel = pred[:, class_idx]
    
    grads = tape.gradient(class_channel, input_image)
    saliency_map = tf.reduce_max(tf.abs(grads), axis=-1)
    
    return saliency_map.numpy()

def test_generate_saliency_map(model, x_test, y_test, n=0):
    # Generate saliency map for a test image
    test_image = x_test[n:n+1]
    true_label = np.argmax(y_test[n])
    saliency_map = generate_saliency_map(model, test_image, true_label)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(test_image.squeeze(), cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(saliency_map.squeeze(), cmap='hot')
    plt.title('Saliency Map')
    plt.show()

def generate_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.layers[-1].output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def test_gradcam(model, img):
    # Generate Grad-CAM heatmap
    last_conv_layer_name = None
    # Replace with the last conv layer name
    for l in model.layers:
        if isinstance(l, Conv2D):
            last_conv_layer_name = l.name
    
    if not last_conv_layer_name:
        raise ValueError("Could not find a convolutional layer.")

    heatmap = generate_gradcam_heatmap(model, img, last_conv_layer_name)

    # Display heatmap
    plt.matshow(heatmap)
    plt.show()

class VisualizationTests(tf.test.TestCase):
    
    def test_plot_loss_histogram(self):
        model = tf.keras.models.load_model("models/autoencoder-mnist.keras")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
        # x_test = x_test #[:1000]
        plot_loss_histogram(model, x_train)
    
    # def test_plot_history_metrics(self):
    #     model = tf.keras.models.load_model("models/mnist_cnn-80k-99.4.keras")
    #     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #     model.summary()
    #     (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    #     x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    #     x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
    #     y_train = tf.keras.utils.to_categorical(y_train, 10)
    #     y_test = tf.keras.utils.to_categorical(y_test, 10)
    #     history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))
    #     plot_history_metrics(history)
    
    # def test_print_basic_weight_statistics(self):
    #     model = tf.keras.models.load_model("models/mnist_cnn-80k-99.4.keras")
    #     print_basic_weight_statistics(model, show_conv=True)
    
    # def test_visualize_activations(self):
    #     model = tf.keras.models.load_model("models/mnist_cnn-80k-99.4.keras")
    #     (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    #     # x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    #     # y_train = tf.keras.utils.to_categorical(y_train, 10)
    #     x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
    #     y_test = tf.keras.utils.to_categorical(y_test, 10)
    #     for i in range(2):
    #         visualize_activations(model, x_test, n=2+i)

    # def test_generate_saliency_map(self):
    #     model = tf.keras.models.load_model("models/mnist_cnn-80k-99.4.keras")
    #     (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    #     # x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    #     # y_train = tf.keras.utils.to_categorical(y_train, 10)
    #     x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
    #     y_test = tf.keras.utils.to_categorical(y_test, 10)
    #     test_generate_saliency_map(model, x_test, y_test, n=1)
    
    # def test_gradcam(self):
    #     model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
    #     img = tf.keras.preprocessing.image.load_img('vendor/tf-docs/site/en/tutorials/load_data/images/csv/Titanic.jpg')
    #     img_array = tf.keras.preprocessing.image.img_to_array(img)
    #     img_array = np.expand_dims(img_array, axis=0)
    #     img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    #     test_gradcam(model, img_array)
    

if __name__ == "__main__":
    if True:
        tf.test.main()
    else:
        import sys
        # model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
        # get model file name from argument
        if len(sys.argv) > 1:
            path = sys.argv[1]
        model = tf.keras.models.load_model("mnist_cnn-80k-99.4.keras" if not 'path' in locals() else path)
        print_basic_weight_statistics(model)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        x_test = x_test.astype('float32') / 255
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        for n in range(5):
            test_generate_saliency_map(model, x_test, y_test, n=n+5)
        
        visualize_activations(model, x_test)

        # test_gradcam(model, x_test[0:1])

# 
