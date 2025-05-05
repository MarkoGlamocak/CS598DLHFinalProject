import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class GradientReversal(Layer):
    """
    Gradient Reversal Layer for Domain-Adversarial Neural Networks.
    
    This layer implements the gradient reversal operation described in:
    "Domain-Adversarial Training of Neural Networks" by Ganin et al. (2016)
    
    The forward pass is the identity operation, but the gradient in the backward 
    pass is multiplied by -lambda.
    """
    
    def __init__(self, lambda_=1.0, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.lambda_ = lambda_
        self.supports_masking = True
    
    def call(self, inputs, **kwargs):
        return inputs
    
    def get_config(self):
        config = super(GradientReversal, self).get_config()
        config.update({"lambda_": self.lambda_})
        return config
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    @tf.custom_gradient
    def grad_reverse(self, x):
        y = tf.identity(x)
        def custom_grad(dy):
            return -self.lambda_ * dy
        return y, custom_grad

    def build(self, input_shape):
        self.built = True