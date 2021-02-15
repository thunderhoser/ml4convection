"""Methods for creating saliency maps."""

from tensorflow.keras import backend as K


def do_saliency_calculations(
        model_object, loss_tensor, list_of_input_matrices):
    """Does saliency calculations.

    T = number of input tensors to the model
    E = number of examples (storm objects)

    :param model_object: Instance of `keras.models.Model`.
    :param loss_tensor: Keras tensor defining the loss function.
    :param list_of_input_matrices: length-T list of numpy arrays, comprising one
        or more examples (storm objects).  list_of_input_matrices[i] must have
        the same dimensions as the [i]th input tensor to the model.
    :return: list_of_saliency_matrices: length-T list of numpy arrays,
        comprising the saliency map for each example.
        list_of_saliency_matrices[i] has the same dimensions as
        list_of_input_matrices[i] and defines the "saliency" of each value x,
        which is the gradient of the loss function with respect to x.
    """

    if isinstance(model_object.input, list):
        list_of_input_tensors = model_object.input
    else:
        list_of_input_tensors = [model_object.input]

    list_of_gradient_tensors = K.gradients(loss_tensor, list_of_input_tensors)
    num_input_tensors = len(list_of_input_tensors)

    for i in range(num_input_tensors):
        list_of_gradient_tensors[i] /= K.maximum(
            K.std(list_of_gradient_tensors[i]), K.epsilon()
        )

    inputs_to_gradients_function = K.function(
        list_of_input_tensors + [K.learning_phase()], list_of_gradient_tensors
    )

    # list_of_saliency_matrices = None
    # num_examples = list_of_input_matrices[0].shape[0]
    #
    # for i in range(num_examples):
    #     these_input_matrices = [a[[i], ...] for a in list_of_input_matrices]
    #     these_saliency_matrices = inputs_to_gradients_function(
    #         these_input_matrices + [0])
    #
    #     if list_of_saliency_matrices is None:
    #         list_of_saliency_matrices = these_saliency_matrices + []
    #     else:
    #         for i in range(num_input_tensors):
    #             list_of_saliency_matrices[i] = numpy.concatenate(
    #                 (list_of_saliency_matrices[i], these_saliency_matrices[i]),
    #                 axis=0)

    list_of_saliency_matrices = inputs_to_gradients_function(
        list_of_input_matrices + [0]
    )

    for i in range(num_input_tensors):
        list_of_saliency_matrices[i] *= -1

    return list_of_saliency_matrices
