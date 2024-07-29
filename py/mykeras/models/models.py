
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

class QuadModel(tf.Module):
    '''A quadratic model with randomly initialized weights and a bias:
    Quadratic Model : quadratic_weight * x^2 + linear_weight * x + bias
    '''
    def __init__(self):
        # Randomly generate weight and bias terms
        rand_init = tf.random.uniform(shape=[3], minval=0., maxval=5., seed=22)
        # Initialize model parameters
        self.w_q = tf.Variable(rand_init[0])
        self.w_l = tf.Variable(rand_init[1])
        self.b = tf.Variable(rand_init[2])

    @tf.function
    def __call__(self, x):
        # Quadratic Model : quadratic_weight * x^2 + linear_weight * x + bias
        return self.w_q * (x**2) + self.w_l * x + self.b

def simple_split_model():
    inputs = keras.Input(shape=(None, None, 3))
    processed = keras.layers.RandomCrop(width=128, height=128)(inputs)
    conv = keras.layers.Conv2D(filters=32, kernel_size=3)(processed)
    pooling = keras.layers.GlobalAveragePooling2D()(conv)
    feature = keras.layers.Dense(10)(pooling)

    full_model = keras.Model(inputs, feature)
    backbone = keras.Model(processed, conv)
    activations = keras.Model(conv, feature)
    
    return full_model, backbone, activations

if __name__ == "__main__":
    model, backbone, activations = simple_split_model()
    
    print(f"model: {model}")
    print(f"backbone: {backbone}")
    print(f"activations: {activations}")
    (train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
    # print(f"data: {data[:10]}")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_x[:1000], train_y[:1000], epochs=1)
    print(f"{model(train_x[:10])} = {train_y[:10]}")

    model = QuadModel()
    

def plot_preds(x, y, f, model, title):
    plt.figure()
    plt.plot(x, y, '.', label='Data')
    plt.plot(x, f(x), label='Ground truth')
    plt.plot(x, model(x), label='Predictions')
    plt.title(title)
    plt.legend()
