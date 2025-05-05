import tensorflow as tf

@tf.custom_gradient
def flip_gradient(x, lambda_value=1.0):
    """
    Custom gradient function for the gradient reversal layer.
    """
    def grad(dy):
        return -lambda_value * dy, None
    
    return x, grad

class GradientReversal(tf.keras.layers.Layer):
    """
    Gradient Reversal Layer for TensorFlow 2.x
    """
    def __init__(self, lambda_value=1.0):
        super(GradientReversal, self).__init__()
        self.lambda_value = lambda_value
    
    def call(self, inputs):
        return flip_gradient(inputs, self.lambda_value)
    
    def get_config(self):
        config = super(GradientReversal, self).get_config()
        config.update({'lambda_value': self.lambda_value})
        return config