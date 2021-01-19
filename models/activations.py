import tensorflow as tf
def tanh(x):
    return tf.keras.activations.tanh(x)


def lrelu(model_yaml, x):
    return tf.nn.leaky_relu(x, alpha=model_yaml["lrelu_alpha"])