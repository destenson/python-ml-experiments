import tensorflow as tf
import numpy as np
import math
import keras

# Normalize radar reflection data and pad to
def preprocess_radar_data(reflections, max_length):
    # Normalize the data
    reflections = (reflections - np.mean(reflections, axis=0)) / np.std(reflections, axis=0)

    # Pad reflections to the fixed length
    if len(reflections) < max_length:
        reflections = np.pad(reflections, ((0, max_length - len(reflections)), (0, 0)), mode='constant')
    else:
        reflections = reflections[:max_length]

    return reflections

def create_deepreflecs_model(input_shape, num_classes):
    print(input_shape)
    inputs = keras.Input(shape=input_shape)
    if input_shape[0] is None:
        if len(input_shape) == 3:
            new_input_shape = (None, input_shape[1], input_shape[2])
        elif len(input_shape) == 4:
            new_input_shape = (None, input_shape[1] * input_shape[2], input_shape[3])
    else:
        if len(input_shape) == 2:
            new_input_shape = (input_shape[0], input_shape[1])
        elif len(input_shape) == 3:
            new_input_shape = (input_shape[0] * input_shape[1], input_shape[2])

    print(f"Input shape was: {input_shape}")
    # 1D convolution layer
    if input_shape[0] is None:
        if len(input_shape) == 3:
            x = keras.layers.Conv1D(filters=16, kernel_size=1, activation='relu')(inputs)
        elif len(input_shape) == 4:
            print(f"Using new input shape: {new_input_shape}")
            flat_inputs = keras.layers.Reshape(new_input_shape)(inputs)
            print(f"After reshaping: {flat_inputs.shape}")
            x = keras.layers.Conv1D(filters=16, kernel_size=1, activation='relu')(flat_inputs)
    else:
        if len(input_shape) == 2:
            x = keras.layers.Conv1D(filters=16, kernel_size=1, activation='relu')(inputs)
        elif len(input_shape) == 3:
            print(f"Input shape was: {input_shape}")
            print(f"Using new input shape: {new_input_shape}")
            flat_inputs = keras.layers.Reshape(new_input_shape)(inputs)
            print(f"After reshaping: {flat_inputs.shape}")
            x = keras.layers.Conv1D(filters=16, kernel_size=1, activation='relu')(flat_inputs)

    # Global context layer
    global_features = keras.layers.GlobalMaxPooling1D()(x)
    global_features = keras.layers.RepeatVector(new_input_shape[0])(global_features)
    # global_features = keras.layers.RepeatVector(input_shape[0])(global_features)
    x = keras.layers.Concatenate(axis=-1)([x, global_features])

    # Another 1D convolution layer
    if input_shape[1] > 100:
        x = keras.layers.Conv1D(filters=32, kernel_size=1, activation='relu')(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        x = keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu')(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        x = keras.layers.Conv1D(filters=32, kernel_size=1, activation='relu')(x)
    else:
        x = keras.layers.Conv1D(filters=32, kernel_size=1, activation='relu')(x)

    # Global max pooling
    x = tf.keras.layers.GlobalMaxPooling1D()(x)

    # Dense layer for final classification
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Example usage
input_shape = (64, 5)  # 64 reflections, 5 features per reflection
num_classes = 4  # car, pedestrian, cyclist, non-obstacle

model = create_deepreflecs_model(input_shape, num_classes)
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'],)
model.save('rcnn.h5')
model.save('rcnn.keras')


def mnist_features_for_deepreflecs(X, flatten=False):
    # Normalize the data
    stddev = np.std(X, axis=0)
    stddev[stddev == 0] = 1.0
    X = (X - np.mean(X, axis=0)) / stddev

    # convert grids of pixels to a list of (x,y,alpha) coordinates

    # Assuming you have a tensor with shape (28, 28, 1)
    # Step 1: Reshape the input tensor
    reshaped = tf.reshape(X, [-1, 784, 1])
    print(f"Input shape: {X.shape}")
    print(f"Reshaped shape: {reshaped.shape}")

    # Step 2: Create coordinate matrices
    x_coords = tf.repeat(tf.range(28, dtype=tf.float64), 28)
    y_coords = tf.tile(tf.range(28, dtype=tf.float64), [28])
    coords = tf.stack([x_coords, y_coords], axis=-1)
    coords = tf.expand_dims(coords, 0)
    coords = tf.tile(coords, [tf.shape(X)[0], 1, 1])

    # Step 3: Combine coordinates and pixel values
    result = tf.concat([coords, reshaped], axis=-1)

    # Step 4: Reshape the final tensor
    if flatten:
        result = tf.reshape(result, [-1, 784, 3])
    else:
        result = tf.reshape(result, [-1, 28, 28, 3])

    return result

# # MNIST
# input_shape = (28, 28, 3)  # 768 pixels, 3 features per pixel (x,y,z)
# # input_shape = (784, 3)  # 768 pixels, 3 features per pixel (x,y,z)
# num_classes = 10  # each digit

# model = create_deepreflecs_model(input_shape, num_classes)
# model.summary()
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'],)

# trained_model, history = train_the_model(
#     model,
#     mnist_features_for_deepreflecs(train_X), train_y,
#     # mnist_features_for_deepreflecs(train_X)[:10000], train_y[:10000],
#     mnist_features_for_deepreflecs(test_X), test_y,
#     save_keras_path='models',
#     use_early_stopping=False,
#     learning_rate=0.0025, batch_size=32, epochs=10)


# trained_model.summary()


# # @keras.decorators.register_keras_serializable()
# # class DeepReflecsModel(models.Model):
# #     def __init__(self, **kwargs):
# #         super(KANModel, self).__init__(**kwargs)
