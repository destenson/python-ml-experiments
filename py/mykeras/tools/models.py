
import tensorflow as tf
from tensorflow.keras.layers import Layer


def get_all_weights(model):
    # for layer in model.layers:
    #     print(layer.name, layer.input, layer.output)
    #     weights = np.zeros(img_shape)
    #     if hasattr(layer, 'get_weights'):
    #         layer_weights = layer.get_weights()
    #         if layer_weights:
    #             np.concatenate(weights, (np.concatenate([w.flatten() for w in layer_weights])))
    #     layer_weights[layer.name] = weights
    weights = {}
    for layer in model.layers:
        weights[layer.name] = layer.get_weights()
    return weights



    
if __name__ == '__main__':
    tf.test.main()

#
