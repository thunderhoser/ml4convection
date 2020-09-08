"""Custom loss functions for Keras models."""

from keras import backend as K


def _log2(input_tensor):
    """Computes logarithm in base 2.

    :param input_tensor: Keras tensor.
    :return: logarithm_tensor: Keras tensor with the same shape as
        `input_tensor`.
    """

    return K.log(input_tensor) / K.log(2.)


def weighted_xentropy(class_weights):
    """Weighted cross-entropy.

    :param class_weights: length-2 numpy with class weights for loss function.
        Elements will be interpreted as
        (negative_class_weight, positive_class_weight).
    :return: loss: Loss function (defined below).
    """

    def loss(target_tensor, prediction_tensor):
        """Computes loss (weighted cross-entropy).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Weighted cross-entropy.
        """

        weight_tensor = (
            target_tensor * class_weights[1] +
            (1. - target_tensor) * class_weights[0]
        )

        xentropy_tensor = (
            target_tensor * _log2(prediction_tensor) +
            (1. - target_tensor) * _log2(1. - prediction_tensor)
        )

        return -K.mean(weight_tensor * xentropy_tensor)

    return loss
