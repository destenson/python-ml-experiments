import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Conv1D, Conv2D
from tensorflow.keras.models import Model # type: ignore

from scipy.ndimage import zoom

import math
import os
import cv2

from py.mykeras.tools.models import get_all_weights


def make_training_videos(model, training_data, path='videos/', resolution=(1920, 1080), verbose=1):
    '''
    Create videos of the training process by grabbing the weights at each epoch
    and plotting them as frames of the video. Also plot the loss values.
    '''
    from matplotlib.animation import FuncAnimation
    import os
    
    os.makedirs(path, exist_ok=True)
    model_name = model.name
    path_pfx = f"{path}{model_name}"
    print(f"Creating videos for model {model_name} at path {path_pfx}") if verbose > 0 else None
    
    # x_train, y_train, x_test, y_test = training_data

    img_shape = [resolution[1], resolution[0]]

    # start by figuring out the structure and layout of the model
    layer_weights = get_all_weights(model)

    # create a video for each layer
    for layer_name, weights in layer_weights.items():
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        line, = ax.plot([], [], lw=2)
        img = ax.imshow(np.zeros(img_shape), aspect='auto')

        def init():
            line.set_data([], [])
            img.set_data(np.zeros(img_shape))
            return line, img

        def update(frame):
            model.fit(training_data, epochs=1, batch_size=320)
            weights = []
            for layer in model.layers:
                if hasattr(layer, 'get_weights'):
                    layer_weights = layer.get_weights()
                    if layer_weights:
                        wts = [w.reshape(img_shape) for w in layer_weights]
                        weights.append(np.concatenate(wts))

            weights = np.concatenate(weights).reshape(img_shape)
            img.set_data(weights)
            y_pred = model.predict(training_data)
            line.set_data(training_data, y_pred)
            return line, img
        
        ani = FuncAnimation(fig, update, frames=range(100), init_func=init, blit=True)
        ani.save(f"{path_pfx}_{layer_name}.mp4", fps=2)
        

def make_training_video(model, x_train, y_train, x_test, y_test, trainfn=None, filename='training.mp4'):
    '''
    Create a video of the training process by grabbing the weights at each epoch
    and plotting them as an image. Also plot the loss values.
    '''
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    line, = ax.plot([], [], lw=2)
    img_shape = (1920, 1080)
    img = ax.imshow(np.zeros(img_shape), aspect='auto')

    def init():
        line.set_data([], [])
        img.set_data(np.zeros((1280, 1280)))
        return line, img

    def update(frame):
        model.fit(x_train, y_train, epochs=1, batch_size=320, validation_data=(x_test, y_test))
        weights = []
        print("Updating frame: ", frame)
        for layer in model.layers:
            print("Updating layer: ", layer.name)
            if hasattr(layer, 'get_weights'):
                layer_weights = layer.get_weights()
                if layer_weights:
                    print("Layer weights len: ", len(layer_weights))

                    rescaled_weights = []
                    for w in layer_weights:
                        if len(w.shape) == 4:
                            print(f"weights shape: {w.shape}")
                            print("drawing to subplot")
                            weight_slice = w[:, :, 0, :]
                            num_filters = weight_slice.shape[2]
                            grid_size = int(np.ceil(np.sqrt(num_filters)))
                            
                            f, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
                            f.suptitle(layer.name)
                            
                            for i in range(num_filters):
                                axis = axes[i // grid_size, i % grid_size]
                                axis.imshow(weight_slice[:, :, i], cmap='viridis')
                                axis.axis('off')
                        elif len(w.shape) > 1:
                            print(f"weights shape: {w.shape}")
                            newy = max(img_shape[0], min(img_shape[0], w.shape[0]))
                            newx = max(img_shape[1], min(img_shape[1], w.shape[1]))
                            scale_factors = (
                                img_shape[1] / newy, img_shape[0] / newx )
                            print(f"zooming to shape: {scale_factors}")
                            rescaled = zoom(w, scale_factors)
                            print(f"weights shape now: {rescaled.shape}")
                            rescaled_weights.append(rescaled)
                        else:
                            print(f"weights shape: {w.shape}")
                            # newy = max(img_shape[0], min(img_shape[0], w.shape[0])) + 0.5
                            scale_factor = img_shape[0]/w.shape[0]
                            print(f"zoom factor: {scale_factor}")
                            rescaled = zoom([w], scale_factor)
                            print(f"weights shape now: {rescaled.shape}")
                            rescaled_weights.append(rescaled)

                    weights.append(rescaled_weights)
                    # weights.append(np.concatenate([w.flatten() for w in layer_weights]))

            print("Finished updating layer: ", layer.name)

        print("Finished updating frame: ", frame)
        # print the lengths of all the dimensions of weights
        for i, w in enumerate(weights):
            print(f"Layer {i} weights shape: {len(w)}")
            for j, wj in enumerate(w):
                print(f"Layer {i} weights {j} shape: {wj.shape}")
        
        df = pd.DataFrame(weights)
        img.set_data(df)
        # weights = np.concatenate(weights) #.reshape((1280, 1280))
        # img.set_data(weights)
        y_pred = model.predict(x_test)
        line.set_data(y_test, y_pred)
        return line, img
    
    ani = FuncAnimation(fig, update, frames=range(100), init_func=init, blit=True)
    ani.save(filename, fps=2)


def plot_loss_histogram(model, x):
    '''
    Plot histogram of loss values for the given model and input data.
    This is useful for determining a threshold for anomaly detection.
    This is used with autoencoders.
    '''
    # Get train MAE loss.
    x_pred = model.predict(x)
    mae_loss = np.mean(tf.square(x_pred - x), axis=1)

    # print("MAE loss shape: ", mae_loss.shape)
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

    # def test_make_training_video(self):
    #     model = tf.keras.models.load_model("models/rnn-mnist-97.8.keras")
    #     (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    #     x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
    #     x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255
    #     y_train = tf.keras.utils.to_categorical(y_train, 10)
    #     y_test = tf.keras.utils.to_categorical(y_test, 10)
    #     training_data = (x_train[:10], y_train[:10], x_test[:10], y_test[:10])
    #     make_training_videos(model, training_data)
    
    # def test_make_training_video(self):
    #     model = tf.keras.models.load_model("models/mnist_cnn-80k-99.4.keras")
    #     (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    #     x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
    #     x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255
    #     y_train = tf.keras.utils.to_categorical(y_train, 10)
    #     y_test = tf.keras.utils.to_categorical(y_test, 10)
    #     make_training_video(model, x_train[:10], y_train[:10], x_test[:10], y_test[:10])
    
    # def test_plot_loss_histogram(self):
    #     model = tf.keras.models.load_model("models/autoencoder-mnist.keras")
    #     (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    #     x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
    #     x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255
    #     x_train = x_train[:1000]
    #     x_test = x_test[:1000]
    #     plot_loss_histogram(model, x_train)
    #     plot_loss_histogram(model, x_test)
    
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
    
    def test_print_basic_weight_statistics_for_all_known_models(self):
        import os
        for file in os.listdir("models"):
            if file.endswith(".keras") or file.endswith(".h5"):
                print("Loading model: ", file)
                model = tf.keras.models.load_model(f"models/{file}")
                print_basic_weight_statistics(model, show_conv=False)
                print('\n')
    
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
