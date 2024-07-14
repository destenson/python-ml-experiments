
import tensorflow as tf

import keras


@keras.utils.register_keras_serializable()
def count_errors(y_true, y_pred):
    try:
        # print("Trying to count errors")
        # Ensure y_true and y_pred are of the same type
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Round y_pred to 0 or 1
        y_pred_rounded = tf.round(y_pred)

        # Calculate false positives and false negatives
        false_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred_rounded, 1)), tf.float32))
        false_negatives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred_rounded, 0)), tf.float32))

        # Calculate total errors
        total_errors = false_positives + false_negatives

        # TOO SPAMMY :: Print debug information :: TOO SPAMMY
        # tf.print("False Positives:", false_positives)
        # tf.print("False Negatives:", false_negatives)
        # tf.print("Total Errors:", total_errors)

        return total_errors
    except Exception as e:
        print(f"Exception in count_errors: {e}")
        return None

# count_errors.__name__ = 'count_errors'

# print(f"test_y.shape: {test_y.shape}")
# print(f"count_errors: {count_errors(test_y.iloc[:,:10], test_y.iloc[:,:10])}")
# print(f"count_errors: {count_errors(np.zeros(test_y.iloc[:,:10].shape), test_y.iloc[:,:10])}")
# print(f"count_errors: {count_errors(np.ones(test_y.iloc[:,:10].shape), test_y.iloc[:,:10])}")

# assert(count_errors(test_y.iloc[:,:10], test_y.iloc[:,:10]) == 0)
# assert(count_errors(np.zeros(test_y.iloc[:,:10].shape), test_y.iloc[:,:10]) == 10000)
# assert(count_errors(np.ones(test_y.iloc[:,:10].shape), test_y.iloc[:,:10]) == 90000)

@keras.utils.register_keras_serializable()
def binary_crossentropy_first_n_columns(num_outputs):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, name='bce')
    @keras.utils.register_keras_serializable()
    def actual_loss_from_first_n_columns(y_true, y_pred):
        # Only consider the first 'num_outputs' columns
        return bce(y_true[:, :num_outputs], y_pred[:, :num_outputs])
    return actual_loss_from_first_n_columns
    # return lambda y_true, y_pred: tf.keras.losses.mse(y_true[:, :num_outputs], y_pred[:, :num_outputs])

@keras.utils.register_keras_serializable()
def mnist_loss():
    closs = binary_crossentropy_first_n_columns(10)

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, name='bce')
    # sce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, name='sce')

    tp = tf.keras.metrics.TruePositives(name='tp')
    tn = tf.keras.metrics.TrueNegatives(name='tn')
    fn = tf.keras.metrics.FalseNegatives(name='fn')
    fp = tf.keras.metrics.FalsePositives(name='fp')

    @keras.utils.register_keras_serializable()
    def myloss(y_true, y_pred):
        tp_ = tp(y_true, y_pred)
        tn_ = tn(y_true, y_pred)
        fp_ = fp(y_true, y_pred)
        fn_ = fn(y_true, y_pred)
        k = (fp_+fn_)/(1.+tp_+tn_)
        # k = 0.5
        return k * (closs(y_true, y_pred) + bce(y_true[:, :10], y_pred[:, :10]))

    return myloss

# def test_mnist_loss():
#     loss = mnist_loss()

#     result = loss(train_y.iloc[:5], test_y.iloc[:5])

# test_mnist_loss()

@keras.utils.register_keras_serializable()
def combined_loss(w=[1., 1., 1.]):
    @keras.utils.register_keras_serializable()
    def combined(y_true, y_pred):
        real_loss = binary_crossentropy_first_n_columns(10)
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, name='bce')
        return w[0]*bce(y_true, y_pred) + \
            w[1]*real_loss(y_true, y_pred) + \
            w[2]*count_errors(y_true, y_pred)/(32 if batch_size is None else batch_size)
    return combined


def get_custom_objects():
    return {
        "count_errors": count_errors,
        "binary_crossentropy_first_n_columns": binary_crossentropy_first_n_columns(10),
        "mnist_loss": mnist_loss,
        "combined_loss": combined_loss,
    }
