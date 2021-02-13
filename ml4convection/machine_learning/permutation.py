"""Permutation-based importance test."""

import numpy
from gewittergefahr.gg_utils import error_checking
from ml4convection.machine_learning import neural_net
from ml4convection.utils import evaluation
from ml4convection.utils import general_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'


def make_fss_cost_function(
        matching_distance_px, square_filter, model_metadata_dict):
    """Creates FSS (fractions skill score) cost function.

    :param matching_distance_px: Matching distance (number of pixels).
    :param square_filter: Boolean flag.  If True, will square FSS filter.  If
        False, will use circular filter.
    :param model_metadata_dict: Dictionary returned by
        `neural_net.read_metafile`.
    :return: cost_function: Function (see below).
    """

    error_checking.assert_is_geq(matching_distance_px, 0.)
    error_checking.assert_is_boolean(square_filter)

    mask_matrix = model_metadata_dict[neural_net.MASK_MATRIX_KEY]
    eroded_mask_matrix = general_utils.erode_binary_matrix(
        binary_matrix=mask_matrix, buffer_distance_px=matching_distance_px
    )

    def cost_function(model_object, generator_object):
        """FSS cost function.

        :param model_object: Model (trained instance of `keras.models.Model` or
            `keras.models.Sequential`).
        :param generator_object: Generator created by
            `neural_net.generator_partial_grids`.
        :return: cost: FSS (a scalar).
        """

        actual_sse = 0.
        reference_sse = 0.

        while True:
            try:
                this_predictor_matrix, this_target_matrix = next(
                    generator_object
                )
            except StopIteration:
                print(SEPARATOR_STRING)
                break

            this_num_examples = this_predictor_matrix.shape[0]
            this_prediction_matrix = neural_net.apply_model_full_grid(
                model_object=model_object,
                predictor_matrix=this_predictor_matrix,
                num_examples_per_batch=this_num_examples, verbose=True
            )

            del this_predictor_matrix
            print(SEPARATOR_STRING)

            for i in range(this_num_examples):
                # TODO(thunderhoser): Make this method public.

                this_actual_sse_matrix, this_reference_sse_matrix = (
                    evaluation._get_fss_components_one_time(
                        actual_target_matrix=this_target_matrix[i, ...],
                        probability_matrix=this_prediction_matrix[i, ...],
                        matching_distance_px=matching_distance_px,
                        eroded_eval_mask_matrix=eroded_mask_matrix,
                        square_filter=square_filter
                    )
                )

                actual_sse += numpy.nansum(this_actual_sse_matrix)
                reference_sse += numpy.nansum(this_reference_sse_matrix)

        return 1. - actual_sse / reference_sse

    return cost_function
