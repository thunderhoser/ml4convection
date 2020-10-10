"""Finds extreme examples (best and worst predictions) for one model."""

import argparse
import numpy
from gewittergefahr.gg_utils import error_checking
from ml4convection.utils import evaluation

TOLERANCE = 1e-6
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MAX_TIME_STEPS_PER_DAY = 144

MOST_ACTUAL_CONVECTION_NAME = 'most_actual_convection'
LEAST_ACTUAL_CONVECTION_NAME = 'least_actual_convection'
MOST_PREDICTED_CONVECTION_NAME = 'most_predicted_convection'
LEAST_PREDICTED_CONVECTION_NAME = 'least_predicted_convection'
HIGHEST_POD_NAME = 'highest_pod'
LOWEST_POD_NAME = 'lowest_pod'
HIGHEST_SUCCESS_RATIO_NAME = 'highest_success_ratio'
LOWEST_SUCCESS_RATIO_NAME = 'lowest_success_ratio'
HIGHEST_CSI_NAME = 'highest_csi'
LOWEST_CSI_NAME = 'lowest_csi'
HIGHEST_FSS_NAME = 'highest_fss'
LOWEST_FSS_NAME = 'lowest_fss'
HIGHEST_BRIER_SCORE_NAME = 'highest_brier_score'
LOWEST_BRIER_SCORE_NAME = 'lowest_brier_score'

VALID_SET_NAMES = [
    MOST_ACTUAL_CONVECTION_NAME, LEAST_ACTUAL_CONVECTION_NAME,
    MOST_PREDICTED_CONVECTION_NAME, LEAST_PREDICTED_CONVECTION_NAME,
    HIGHEST_POD_NAME, LOWEST_POD_NAME,
    HIGHEST_SUCCESS_RATIO_NAME, LOWEST_SUCCESS_RATIO_NAME,
    HIGHEST_CSI_NAME, LOWEST_CSI_NAME, HIGHEST_FSS_NAME, LOWEST_FSS_NAME,
    HIGHEST_BRIER_SCORE_NAME, LOWEST_BRIER_SCORE_NAME
]

VALID_TIMES_KEY = 'valid_times_unix_sec'
ACTUAL_COUNTS_KEY = 'actual_convective_px_counts'
PREDICTED_COUNTS_KEY = 'predicted_convective_px_counts'
POD_VALUES_KEY = 'pod_values'
SUCCESS_RATIOS_KEY = 'success_ratios'
CSI_VALUES_KEY = 'csi_values'
FSS_VALUES_KEY = 'fss_values'
BRIER_SCORES_KEY = 'brier_scores'

INPUT_DIR_ARG_NAME = 'input_basic_score_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
NUM_EXAMPLES_PER_SET_ARG_NAME = 'num_examples_per_set'
SET_NAMES_ARG_NAME = 'set_names'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein (containing basic scores) will be '
    'found by `evaluation.find_basic_score_file` and read by '
    '`evaluation.read_basic_score_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will find extreme examples for the period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

NUM_EXAMPLES_PER_SET_HELP_STRING = 'Number of examples (time steps) per set.'

SET_NAMES_HELP_STRING = (
    'Sets of extreme examples to find.  Each set name must be in the following '
    'list:\n{0:s}'
).format(str(VALID_SET_NAMES))

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  For each set, basic scores will be written here'
    ' by `evaluation.write_basic_score_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_PER_SET_ARG_NAME, type=int, required=False, default=100,
    help=NUM_EXAMPLES_PER_SET_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SET_NAMES_ARG_NAME, type=str, nargs='+', required=False,
    default=VALID_SET_NAMES, help=SET_NAMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _get_contingency_table_scores(basic_score_table_xarray):
    """Computes contingency-table-based scores at each time step.

    Specifically, at each time step this method averages the following scores
    over all probability thresholds:

    - POD
    - success ratio
    - CSI

    T = number of time steps

    :param basic_score_table_xarray: xarray table in format returned by
        `evaluation.read_basic_score_file`.
    :return: pod_values: length-T numpy array of POD values.
    :return: success_ratios: length-T numpy array of success ratios.
    :return: csi_values: length-T numpy array of CSI values.
    """

    t = basic_score_table_xarray

    numerator_matrix = (
        t[evaluation.NUM_ACTUAL_ORIENTED_TP_KEY].values
    ).astype(float)

    denominator_matrix = (
        t[evaluation.NUM_ACTUAL_ORIENTED_TP_KEY].values +
        t[evaluation.NUM_FALSE_NEGATIVES_KEY].values
    ).astype(float)

    denominator_matrix[denominator_matrix == 0] = numpy.nan
    pod_matrix = numerator_matrix / denominator_matrix
    pod_values = numpy.nanmean(pod_matrix, axis=1)

    numerator_matrix = (
        t[evaluation.NUM_PREDICTION_ORIENTED_TP_KEY].values
    ).astype(float)

    denominator_matrix = (
        t[evaluation.NUM_PREDICTION_ORIENTED_TP_KEY].values +
        t[evaluation.NUM_FALSE_POSITIVES_KEY].values
    ).astype(float)

    denominator_matrix[denominator_matrix == 0] = numpy.nan
    success_ratio_matrix = numerator_matrix / denominator_matrix
    success_ratios = numpy.nanmean(success_ratio_matrix, axis=1)

    pod_matrix[pod_matrix == 0] = numpy.nan
    success_ratio_matrix[success_ratio_matrix == 0] = numpy.nan
    csi_matrix = (pod_matrix ** -1 + success_ratio_matrix ** -1 - 1) ** -1
    csi_values = numpy.nanmean(csi_matrix, axis=1)

    return pod_values, success_ratios, csi_values


def _read_scores_one_day(input_file_name):
    """Reads scores for one day.

    T = number of time steps in file

    :param input_file_name: Path to input file (will be read by
        `evaluation.read_basic_score_file`).
    :return: result_dict: Dictionary with the following keys.
    result_dict['valid_times_unix_sec']: length-T numpy array of valid times.
    result_dict['actual_convective_px_counts']: length-T numpy array with
        numbers of actual convective pixels.
    result_dict['predicted_convective_px_counts']: length-T numpy array with
        numbers of predicted convective pixels.
    result_dict['pod_values']: length-T numpy array of POD values.
    result_dict['success_ratios']: length-T numpy array of success ratios.
    result_dict['csi_values']: length-T numpy array of CSI values.
    result_dict['fss_values']: length-T numpy array of FSS values.
    result_dict['brier_scores']: length-T numpy array of Brier scores.
    """

    basic_score_table_xarray = evaluation.read_basic_score_file(input_file_name)
    t = basic_score_table_xarray

    prob_thresholds = t.coords[evaluation.PROBABILITY_THRESHOLD_DIM].values
    zero_threshold_index = numpy.argmin(prob_thresholds)
    assert numpy.isclose(
        prob_thresholds[zero_threshold_index], 0., atol=TOLERANCE
    )

    valid_times_unix_sec = t.coords[evaluation.TIME_DIM].values
    actual_convective_px_counts = (
        t[evaluation.NUM_ACTUAL_ORIENTED_TP_KEY].values[:, zero_threshold_index]
    )

    num_predicted_matrix = (
        t[evaluation.NUM_FALSE_POSITIVES_KEY].values +
        t[evaluation.NUM_PREDICTION_ORIENTED_TP_KEY].values
    )
    predicted_convective_px_counts = numpy.mean(num_predicted_matrix, axis=1)

    pod_values, success_ratios, csi_values = _get_contingency_table_scores(t)

    fss_values = 1. - (
        t[evaluation.ACTUAL_SSE_KEY].values /
        t[evaluation.REFERENCE_SSE_KEY].values
    )

    example_count_matrix = t[evaluation.EXAMPLE_COUNT_KEY].values.astype(float)
    example_count_matrix[example_count_matrix == 0] = numpy.nan

    diff_matrix = (
        t[evaluation.SUMMED_FORECAST_PROB_KEY].values -
        t[evaluation.POSITIVE_EXAMPLE_COUNT_KEY].values
    )
    mse_matrix = (diff_matrix / example_count_matrix) ** 2

    mse_matrix[numpy.isnan(mse_matrix)] = 0.
    example_count_matrix[numpy.isnan(example_count_matrix)] = 0

    brier_scores = numpy.average(
        mse_matrix, weights=example_count_matrix, axis=1
    )

    return {
        VALID_TIMES_KEY: valid_times_unix_sec,
        ACTUAL_COUNTS_KEY: actual_convective_px_counts,
        PREDICTED_COUNTS_KEY: predicted_convective_px_counts,
        POD_VALUES_KEY: pod_values,
        SUCCESS_RATIOS_KEY: success_ratios,
        CSI_VALUES_KEY: csi_values,
        FSS_VALUES_KEY: fss_values,
        BRIER_SCORES_KEY: brier_scores
    }


def _run(top_input_dir_name, first_date_string, last_date_string,
         num_examples_per_set, set_names, output_dir_name):
    """Finds extreme examples (best and worst predictions) for one model.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param num_examples_per_set: Same.
    :param set_names: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if any set names are not in `VALID_SET_NAMES`.
    """

    error_checking.assert_is_geq(num_examples_per_set, 1)
    bad_set_names = [n for n in set_names if n not in VALID_SET_NAMES]

    if len(bad_set_names) > 0:
        error_string = (
            'The following set names are not recognized:\n{0:s}'
        ).format(str(bad_set_names))

        raise ValueError(error_string)

    input_file_names = evaluation.find_many_basic_score_files(
        top_directory_name=top_input_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        gridded=False, raise_error_if_any_missing=False
    )

    num_days = len(input_file_names)
    num_times = MAX_TIME_STEPS_PER_DAY * num_days

    valid_times_unix_sec = numpy.full(num_times, -1, dtype=int)
    actual_convective_px_counts = numpy.full(num_times, -1, dtype=int)
    predicted_convective_px_counts = numpy.full(num_times, -1, dtype=int)
    pod_values = numpy.full(num_times, numpy.nan)
    success_ratios = numpy.full(num_times, numpy.nan)
    csi_values = numpy.full(num_times, numpy.nan)
    fss_values = numpy.full(num_times, numpy.nan)
    brier_scores = numpy.full(num_times, numpy.nan)

    num_times_read = 0

    for i in range(num_days):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        this_result_dict = _read_scores_one_day(input_file_names[i])

        this_num_times = len(this_result_dict[VALID_TIMES_KEY])
        first_time_index = num_times_read + 0
        last_time_index = first_time_index + this_num_times - 1
        num_times_read += this_num_times

        time_indices = numpy.linspace(
            first_time_index, last_time_index, num=this_num_times, dtype=int
        )

        valid_times_unix_sec[time_indices] = this_result_dict[VALID_TIMES_KEY]
        actual_convective_px_counts[time_indices] = (
            this_result_dict[ACTUAL_COUNTS_KEY]
        )
        predicted_convective_px_counts[time_indices] = (
            this_result_dict[PREDICTED_COUNTS_KEY]
        )
        pod_values[time_indices] = this_result_dict[POD_VALUES_KEY]
        success_ratios[time_indices] = this_result_dict[SUCCESS_RATIOS_KEY]
        csi_values[time_indices] = this_result_dict[CSI_VALUES_KEY]
        fss_values[time_indices] = this_result_dict[FSS_VALUES_KEY]
        brier_scores[time_indices] = this_result_dict[BRIER_SCORES_KEY]

    print(SEPARATOR_STRING)

    valid_times_unix_sec = valid_times_unix_sec[:num_times_read]
    actual_convective_px_counts = actual_convective_px_counts[:num_times_read]
    predicted_convective_px_counts = (
        predicted_convective_px_counts[:num_times_read]
    )
    pod_values = pod_values[:num_times_read]
    success_ratios = success_ratios[:num_times_read]
    csi_values = csi_values[:num_times_read]
    fss_values = fss_values[:num_times_read]
    brier_scores = brier_scores[:num_times_read]

    # _find_extreme_examples_one_set(
    #     scores=actual_convective_px_counts,
    #     find_highest=True, num_examples_desired=num_examples_per_set
    # )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        num_examples_per_set=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_SET_ARG_NAME
        ),
        set_names=getattr(INPUT_ARG_OBJECT, SET_NAMES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
