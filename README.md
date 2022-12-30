# DGPF-RENet
DGPF-RENet: A new architecture with low data dependence and fast convergence for hyperspectral image classification

Because we have added the activation function swish that is not available in the keras library, please add the function in keras.activation.py. The code segment is as follows:
\ ```
def swish(x):
    """based on Swish activation function.
    # Arguments
        x: Input tensor.
    # Returns
        The Swish activation: `x * sigmoid(x)`.
    # References
        [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
    """
    if K.backend() == 'tensorflow':
        try:
            # The native TF implementation has a more
            # memory-efficient gradient implementation
            return K.tf.nn.swish(x)
        except AttributeError:
            pass
    return x * K.sigmoid(x)
\ ```
