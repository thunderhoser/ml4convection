"""Finds extreme examples (best and worst predictions) for one model."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import number_rounding
import time_conversion
import gg_model_evaluation as gg_model_eval
import error_checking
import evaluation

TOLERANCE = 1e-6
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

DAYS_TO_SECONDS = 86400
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
    :return: score_dict: Dictionary with the following keys.
    score_dict['valid_times_unix_sec']: length-T numpy array of valid times.
    score_dict['actual_convective_px_counts']: length-T numpy array with
        numbers of actual convective pixels.
    score_dict['predicted_convective_px_counts']: length-T numpy array with
        numbers of predicted convective pixels.
    score_dict['pod_values']: length-T numpy array of POD values.
    score_dict['success_ratios']: length-T numpy array of success ratios.
    score_dict['csi_values']: length-T numpy array of CSI values.
    score_dict['fss_values']: length-T numpy array of FSS values.
    score_dict['brier_scores']: length-T numpy array of Brier scores.
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
        t[evaluation.ACTUAL_SSE_FOR_FSS_KEY].values /
        t[evaluation.REFERENCE_SSE_FOR_FSS_KEY].values
    )

    example_count_matrix = (
        t[evaluation.BINNED_NUM_EXAMPLES_KEY].values.astype(float)
    )
    example_count_matrix[example_count_matrix == 0] = numpy.nan

    mean_forecast_prob_matrix = (
        t[evaluation.BINNED_SUM_PROBS_KEY].values / example_count_matrix
    )
    event_frequency_matrix = (
        t[evaluation.BINNED_NUM_POSITIVES_KEY].values.astype(float) /
        example_count_matrix
    )
    example_count_matrix[numpy.isnan(example_count_matrix)] = 0
    example_count_matrix = numpy.round(example_count_matrix).astype(int)

    num_times = len(valid_times_unix_sec)
    brier_scores = numpy.full(num_times, numpy.nan)

    for i in range(num_times):
        brier_scores[i] = gg_model_eval.get_brier_skill_score(
            mean_forecast_prob_by_bin=mean_forecast_prob_matrix[i, :],
            mean_observed_label_by_bin=event_frequency_matrix[i, :],
            num_examples_by_bin=example_count_matrix[i, :],
            climatology=0.01
        )[gg_model_eval.BRIER_SCORE_KEY]

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


def _find_extreme_examples_one_set(
        scores, valid_times_unix_sec, num_examples_desired, find_highest,
        set_name):
    """Finds one set of extreme examples.

    T = number of time steps
    E = number of extreme examples to find

    :param scores: length-T numpy array of scores.
    :param valid_times_unix_sec: length-T numpy array of valid times.
    :param num_examples_desired: E in the above discussion.
    :param find_highest: Boolean flag.  If True (False), will find the E
        examples with the highest (lowest) scores.
    :param set_name: Name of set.
    :return: desired_times_unix_sec: length-E numpy array of desired times.
    """

    scores = scores.astype(float)

    if find_highest:
        scores[numpy.isnan(scores)] = -numpy.inf
        sort_indices = numpy.argsort(-1 * scores)
    else:
        scores[numpy.isnan(scores)] = numpy.inf
        sort_indices = numpy.argsort(scores)

    desired_indices = sort_indices[:num_examples_desired]

    print('Scores in set "{0:s}":\n{1:s}'.format(
        set_name, str(scores[desired_indices])
    ))

    return valid_times_unix_sec[desired_indices]


def _find_extreme_examples_all_sets(
        score_dict, set_names, num_examples_per_set):
    """Finds each set of extreme examples.

    :param score_dict: Dictionary in format returned by `_read_scores_one_day`.
    :param set_names: See documentation at top of file.
    :param num_examples_per_set: Same.
    :return: set_to_valid_times_unix_sec: Dictionary, where each key is a set
        name and the corresponding value is a 1-D numpy array of valid times.
    """

    set_to_valid_times_unix_sec = dict()
    all_times_unix_sec = score_dict[VALID_TIMES_KEY]

    if MOST_ACTUAL_CONVECTION_NAME in set_names:
        set_to_valid_times_unix_sec[MOST_ACTUAL_CONVECTION_NAME] = (
            _find_extreme_examples_one_set(
                scores=score_dict[ACTUAL_COUNTS_KEY] + 0,
                valid_times_unix_sec=all_times_unix_sec,
                find_highest=True, num_examples_desired=num_examples_per_set,
                set_name=MOST_ACTUAL_CONVECTION_NAME
            )
        )

        print(MINOR_SEPARATOR_STRING)

    if LEAST_ACTUAL_CONVECTION_NAME in set_names:
        set_to_valid_times_unix_sec[LEAST_ACTUAL_CONVECTION_NAME] = (
            _find_extreme_examples_one_set(
                scores=score_dict[ACTUAL_COUNTS_KEY] + 0,
                valid_times_unix_sec=all_times_unix_sec,
                find_highest=False, num_examples_desired=num_examples_per_set,
                set_name=LEAST_ACTUAL_CONVECTION_NAME
            )
        )

        print(MINOR_SEPARATOR_STRING)

    if MOST_PREDICTED_CONVECTION_NAME in set_names:
        set_to_valid_times_unix_sec[MOST_PREDICTED_CONVECTION_NAME] = (
            _find_extreme_examples_one_set(
                scores=score_dict[PREDICTED_COUNTS_KEY] + 0,
                valid_times_unix_sec=all_times_unix_sec,
                find_highest=True, num_examples_desired=num_examples_per_set,
                set_name=MOST_PREDICTED_CONVECTION_NAME
            )
        )

        print(MINOR_SEPARATOR_STRING)

    if LEAST_PREDICTED_CONVECTION_NAME in set_names:
        set_to_valid_times_unix_sec[LEAST_PREDICTED_CONVECTION_NAME] = (
            _find_extreme_examples_one_set(
                scores=score_dict[PREDICTED_COUNTS_KEY] + 0,
                valid_times_unix_sec=all_times_unix_sec,
                find_highest=False, num_examples_desired=num_examples_per_set,
                set_name=LEAST_PREDICTED_CONVECTION_NAME
            )
        )

        print(MINOR_SEPARATOR_STRING)

    if HIGHEST_POD_NAME in set_names:
        set_to_valid_times_unix_sec[HIGHEST_POD_NAME] = (
            _find_extreme_examples_one_set(
                scores=score_dict[POD_VALUES_KEY] + 0,
                valid_times_unix_sec=all_times_unix_sec,
                find_highest=True, num_examples_desired=num_examples_per_set,
                set_name=HIGHEST_POD_NAME
            )
        )

        print(MINOR_SEPARATOR_STRING)

    if LOWEST_POD_NAME in set_names:
        set_to_valid_times_unix_sec[LOWEST_POD_NAME] = (
            _find_extreme_examples_one_set(
                scores=score_dict[POD_VALUES_KEY] + 0,
                valid_times_unix_sec=all_times_unix_sec,
                find_highest=False, num_examples_desired=num_examples_per_set,
                set_name=LOWEST_POD_NAME
            )
        )

        print(MINOR_SEPARATOR_STRING)

    if HIGHEST_SUCCESS_RATIO_NAME in set_names:
        set_to_valid_times_unix_sec[HIGHEST_SUCCESS_RATIO_NAME] = (
            _find_extreme_examples_one_set(
                scores=score_dict[SUCCESS_RATIOS_KEY] + 0,
                valid_times_unix_sec=all_times_unix_sec,
                find_highest=True, num_examples_desired=num_examples_per_set,
                set_name=HIGHEST_SUCCESS_RATIO_NAME
            )
        )

        print(MINOR_SEPARATOR_STRING)

    if LOWEST_SUCCESS_RATIO_NAME in set_names:
        set_to_valid_times_unix_sec[LOWEST_SUCCESS_RATIO_NAME] = (
            _find_extreme_examples_one_set(
                scores=score_dict[SUCCESS_RATIOS_KEY] + 0,
                valid_times_unix_sec=all_times_unix_sec,
                find_highest=False, num_examples_desired=num_examples_per_set,
                set_name=LOWEST_SUCCESS_RATIO_NAME
            )
        )

        print(MINOR_SEPARATOR_STRING)

    if HIGHEST_CSI_NAME in set_names:
        set_to_valid_times_unix_sec[HIGHEST_CSI_NAME] = (
            _find_extreme_examples_one_set(
                scores=score_dict[CSI_VALUES_KEY] + 0,
                valid_times_unix_sec=all_times_unix_sec,
                find_highest=True, num_examples_desired=num_examples_per_set,
                set_name=HIGHEST_CSI_NAME
            )
        )

        print(MINOR_SEPARATOR_STRING)

    if LOWEST_CSI_NAME in set_names:
        set_to_valid_times_unix_sec[LOWEST_CSI_NAME] = (
            _find_extreme_examples_one_set(
                scores=score_dict[CSI_VALUES_KEY] + 0,
                valid_times_unix_sec=all_times_unix_sec,
                find_highest=False, num_examples_desired=num_examples_per_set,
                set_name=LOWEST_CSI_NAME
            )
        )

        print(MINOR_SEPARATOR_STRING)

    if HIGHEST_FSS_NAME in set_names:
        set_to_valid_times_unix_sec[HIGHEST_FSS_NAME] = (
            _find_extreme_examples_one_set(
                scores=score_dict[FSS_VALUES_KEY] + 0,
                valid_times_unix_sec=all_times_unix_sec,
                find_highest=True, num_examples_desired=num_examples_per_set,
                set_name=HIGHEST_FSS_NAME
            )
        )

        print(MINOR_SEPARATOR_STRING)

    if LOWEST_FSS_NAME in set_names:
        set_to_valid_times_unix_sec[LOWEST_FSS_NAME] = (
            _find_extreme_examples_one_set(
                scores=score_dict[FSS_VALUES_KEY] + 0,
                valid_times_unix_sec=all_times_unix_sec,
                find_highest=False, num_examples_desired=num_examples_per_set,
                set_name=LOWEST_FSS_NAME
            )
        )

        print(MINOR_SEPARATOR_STRING)

    if HIGHEST_BRIER_SCORE_NAME in set_names:
        set_to_valid_times_unix_sec[HIGHEST_BRIER_SCORE_NAME] = (
            _find_extreme_examples_one_set(
                scores=score_dict[BRIER_SCORES_KEY] + 0,
                valid_times_unix_sec=all_times_unix_sec,
                find_highest=True, num_examples_desired=num_examples_per_set,
                set_name=HIGHEST_BRIER_SCORE_NAME
            )
        )

        print(MINOR_SEPARATOR_STRING)

    if LOWEST_BRIER_SCORE_NAME in set_names:
        set_to_valid_times_unix_sec[LOWEST_BRIER_SCORE_NAME] = (
            _find_extreme_examples_one_set(
                scores=score_dict[BRIER_SCORES_KEY] + 0,
                valid_times_unix_sec=all_times_unix_sec,
                find_highest=False, num_examples_desired=num_examples_per_set,
                set_name=LOWEST_BRIER_SCORE_NAME
            )
        )

        print(MINOR_SEPARATOR_STRING)

    return set_to_valid_times_unix_sec


def _write_output_files(set_to_valid_times_unix_sec, top_input_dir_name,
                        output_dir_name):
    """Writes output files (one basic-score file per set).

    :param set_to_valid_times_unix_sec: Dictionary created by
        `_find_extreme_examples_all_sets`.
    :param top_input_dir_name: See documentation at top of file.
    :param output_dir_name: Same.
    :raises: ValueError: if cannot find all desired examples.
    """

    desired_times_unix_sec = numpy.concatenate([
        t for t in set_to_valid_times_unix_sec.values()
    ])
    desired_dates_unix_sec = numpy.unique(number_rounding.floor_to_nearest(
        desired_times_unix_sec, DAYS_TO_SECONDS
    ))
    desired_dates_unix_sec = desired_dates_unix_sec.astype(int)
    desired_date_strings = [
        time_conversion.unix_sec_to_string(t, evaluation.DATE_FORMAT)
        for t in desired_dates_unix_sec
    ]

    set_to_basic_score_tables = dict()
    set_names = list(set_to_valid_times_unix_sec.keys())
    for this_name in set_names:
        set_to_basic_score_tables[this_name] = []

    for this_date_string in desired_date_strings:
        input_file_name = evaluation.find_basic_score_file(
            top_directory_name=top_input_dir_name,
            valid_date_string=this_date_string,
            gridded=False, raise_error_if_missing=True
        )

        print('Reading data from: "{0:s}"...'.format(input_file_name))
        basic_score_table_xarray = evaluation.read_basic_score_file(
            input_file_name
        )

        for this_name in set_names:
            these_flags = numpy.array([
                t in set_to_valid_times_unix_sec[this_name] for t in
                basic_score_table_xarray.coords[evaluation.TIME_DIM].values
            ], dtype=bool)

            if not numpy.any(these_flags):
                continue

            this_basic_score_table = basic_score_table_xarray.isel(
                indexers={evaluation.TIME_DIM: numpy.where(these_flags)[0]},
                drop=False
            )
            set_to_basic_score_tables[this_name].append(this_basic_score_table)

    print(SEPARATOR_STRING)

    for this_name in set_names:
        basic_score_table_xarray = evaluation.concat_basic_score_tables(
            set_to_basic_score_tables[this_name]
        )

        num_examples_found = len(
            basic_score_table_xarray.coords[evaluation.TIME_DIM].values
        )
        num_examples_desired = len(set_to_valid_times_unix_sec[this_name])

        if num_examples_found != num_examples_desired:
            error_string = (
                'Expected {0:d} examples for set "{1:s}"; found {2:d} examples.'
            ).format(num_examples_desired, this_name, num_examples_found)

            raise ValueError(error_string)

        output_file_name = '{0:s}/basic_scores_{1:s}.nc'.format(
            output_dir_name, this_name.replace('_', '-')
        )

        print('Writing results to: "{0:s}"...'.format(output_file_name))
        evaluation.write_basic_score_file(
            basic_score_table_xarray=basic_score_table_xarray,
            netcdf_file_name=output_file_name
        )


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

    score_dict = {
        VALID_TIMES_KEY: numpy.full(num_times, -1, dtype=int),
        ACTUAL_COUNTS_KEY: numpy.full(num_times, -1, dtype=int),
        PREDICTED_COUNTS_KEY: numpy.full(num_times, -1, dtype=int),
        POD_VALUES_KEY: numpy.full(num_times, numpy.nan),
        SUCCESS_RATIOS_KEY: numpy.full(num_times, numpy.nan),
        CSI_VALUES_KEY: numpy.full(num_times, numpy.nan),
        FSS_VALUES_KEY: numpy.full(num_times, numpy.nan),
        BRIER_SCORES_KEY: numpy.full(num_times, numpy.nan)
    }

    num_times_read = 0

    for i in range(num_days):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        this_score_dict = _read_scores_one_day(input_file_names[i])

        this_num_times = len(this_score_dict[VALID_TIMES_KEY])
        first_time_index = num_times_read + 0
        last_time_index = first_time_index + this_num_times - 1
        num_times_read += this_num_times

        time_indices = numpy.linspace(
            first_time_index, last_time_index, num=this_num_times, dtype=int
        )

        for this_key in score_dict:
            score_dict[this_key][time_indices] = this_score_dict[this_key]

    print(SEPARATOR_STRING)

    for this_key in score_dict:
        score_dict[this_key] = score_dict[this_key][:num_times_read]

    set_to_valid_times_unix_sec = _find_extreme_examples_all_sets(
        score_dict=score_dict, set_names=set_names,
        num_examples_per_set=num_examples_per_set
    )
    print(SEPARATOR_STRING)

    _write_output_files(
        set_to_valid_times_unix_sec=set_to_valid_times_unix_sec,
        top_input_dir_name=top_input_dir_name, output_dir_name=output_dir_name
    )


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
