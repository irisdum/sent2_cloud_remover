import tensorflow as tf
def tanh(x):
    return tf.keras.activations.tanh(x)


def lrelu(alpha, x):
    return tf.nn.leaky_relu(x, alpha=alpha)