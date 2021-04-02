"""Computes advanced evaluation scores sans grid (combined over full domain)."""

import os
import sys
import copy
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import error_checking
import climatology_io
import evaluation
import radar_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

NUM_HOURS_PER_DAY = 24
NUM_RADARS = len(radar_utils.RADAR_LATITUDES_DEG_N)

INPUT_DIR_ARG_NAME = 'input_basic_score_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
NUM_BOOTSTRAP_ARG_NAME = 'num_bootstrap_reps'
USE_PARTIAL_GRIDS_ARG_NAME = 'use_partial_grids'
MONTH_ARG_NAME = 'desired_month'
SPLIT_BY_HOUR_ARG_NAME = 'split_by_hour'
CLIMO_FILE_ARG_NAME = 'input_climo_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein (containing basic scores) will be '
    'found by `evaluation.find_basic_score_file` and read by '
    '`evaluation.read_basic_score_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will evaluate predictions for all days in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

NUM_BOOTSTRAP_HELP_STRING = 'Number of bootstrap replicates.'
USE_PARTIAL_GRIDS_HELP_STRING = (
    'Boolean flag.  If 1 (0), will compute scores for partial (full) grids.'
)
MONTH_HELP_STRING = (
    'Will evaluate predictions only for this month (integer in 1...12).  To '
    'evaluate predictions for all months, leave this alone.'
)
SPLIT_BY_HOUR_HELP_STRING = (
    '[used only if `{0:s}` is left alone] Boolean flag.  If 1, will split '
    'evaluation by hour, writing one file for each hour of the day.  If 0, will'
    ' evaluate predictions for all hours.'
).format(MONTH_ARG_NAME)

CLIMO_FILE_HELP_STRING = (
    'Path to file with climatology (event frequencies in training data).'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results (advanced scores) will be written here '
    'by `evaluation.write_advanced_score_file`, to exact locations determined '
    'by `evaluation.find_advanced_score_file`.'
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
    '--' + NUM_BOOTSTRAP_ARG_NAME, type=int, required=True,
    help=NUM_BOOTSTRAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_PARTIAL_GRIDS_ARG_NAME, type=int, required=False, default=0,
    help=USE_PARTIAL_GRIDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MONTH_ARG_NAME, type=int, required=False, default=-1,
    help=MONTH_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SPLIT_BY_HOUR_ARG_NAME, type=int, required=False, default=0,
    help=SPLIT_BY_HOUR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CLIMO_FILE_ARG_NAME, type=str, required=False, default='',
    help=CLIMO_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _compute_scores_partial_grids(
        top_basic_score_dir_name, first_date_string, last_date_string,
        num_bootstrap_reps, climo_file_name, output_dir_name):
    """Computes advanced scores on full grid.

    :param top_basic_score_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param num_bootstrap_reps: Same.
    :param climo_file_name: Same.
    :param output_dir_name: Same.
    """

    print('Reading event frequencies from: "{0:s}"...'.format(
        climo_file_name
    ))
    climo_dict = climatology_io.read_file(climo_file_name)

    date_strings = []
    basic_score_file_names = []

    for k in range(NUM_RADARS):
        if len(date_strings) == 0:
            basic_score_file_names += evaluation.find_many_basic_score_files(
                top_directory_name=top_basic_score_dir_name,
                first_date_string=first_date_string,
                last_date_string=last_date_string,
                gridded=False, radar_number=k,
                raise_error_if_any_missing=False,
                raise_error_if_all_missing=k > 0
            )

            if len(basic_score_file_names) == 0:
                continue

            date_strings = [
                evaluation.basic_file_name_to_date(f)
                for f in basic_score_file_names
            ]
        else:
            basic_score_file_names += [
                evaluation.find_basic_score_file(
                    top_directory_name=top_basic_score_dir_name,
                    valid_date_string=d, gridded=False, radar_number=k,
                    raise_error_if_missing=True
                ) for d in date_strings
            ]

    basic_score_tables = []

    for this_file_name in basic_score_file_names:
        print('Reading basic scores from: "{0:s}"...'.format(this_file_name))
        basic_score_tables.append(
            evaluation.read_basic_score_file(this_file_name)
        )

    print(SEPARATOR_STRING)

    basic_score_table_xarray = (
        evaluation.concat_basic_score_tables(basic_score_tables)
    )
    del basic_score_tables

    advanced_score_table_xarray = (
        evaluation.get_advanced_scores_ungridded(
            basic_score_table_xarray=basic_score_table_xarray,
            training_event_frequency=
            climo_dict[climatology_io.EVENT_FREQ_OVERALL_KEY],
            num_bootstrap_reps=num_bootstrap_reps
        )
    )
    print(advanced_score_table_xarray)

    output_file_name = evaluation.find_advanced_score_file(
        directory_name=output_dir_name, month=None, hour=None, gridded=False,
        raise_error_if_missing=False
    )

    print('\nWriting advanced scores to: "{0:s}"...'.format(output_file_name))
    evaluation.write_advanced_score_file(
        advanced_score_table_xarray=advanced_score_table_xarray,
        pickle_file_name=output_file_name
    )


def _compute_scores_full_grid(
        top_basic_score_dir_name, first_date_string, last_date_string,
        num_bootstrap_reps, desired_month, split_by_hour, climo_file_name,
        output_dir_name):
    """Computes advanced scores on full grid.

    :param top_basic_score_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param num_bootstrap_reps: Same.
    :param desired_month: Same.
    :param split_by_hour: Same.
    :param climo_file_name: Same.
    :param output_dir_name: Same.
    """

    print('Reading event frequencies from: "{0:s}"...'.format(
        climo_file_name
    ))
    climo_dict = climatology_io.read_file(climo_file_name)

    basic_score_file_names = evaluation.find_many_basic_score_files(
        top_directory_name=top_basic_score_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        gridded=False, radar_number=None, raise_error_if_any_missing=False
    )
    date_strings = [
        evaluation.basic_file_name_to_date(f) for f in basic_score_file_names
    ]

    if desired_month is not None:
        dates_unix_sec = numpy.array([
            time_conversion.string_to_unix_sec(t, evaluation.DATE_FORMAT)
            for t in date_strings
        ], dtype=int)

        months = numpy.array([
            int(time_conversion.unix_sec_to_string(t, '%m'))
            for t in dates_unix_sec
        ], dtype=int)

        good_indices = numpy.where(months == desired_month)[0]
        basic_score_file_names = [
            basic_score_file_names[k] for k in good_indices
        ]
        date_strings = [date_strings[k] for k in good_indices]

        del dates_unix_sec, months

    print(SEPARATOR_STRING)

    num_dates = len(date_strings)
    num_splits = NUM_HOURS_PER_DAY if split_by_hour else 1

    basic_score_table_matrix = numpy.full(
        (num_dates, num_splits), '', dtype=object
    )

    for i in range(num_dates):
        print('Reading basic scores from: "{0:s}"...'.format(
            basic_score_file_names[i]
        ))
        this_score_table = evaluation.read_basic_score_file(
            basic_score_file_names[i]
        )

        if not split_by_hour:
            basic_score_table_matrix[i, 0] = copy.deepcopy(this_score_table)
            continue

        for j in range(NUM_HOURS_PER_DAY):
            basic_score_table_matrix[i, j] = (
                evaluation.subset_basic_scores_by_hour(
                    basic_score_table_xarray=this_score_table, desired_hour=j
                )
            )

    print(SEPARATOR_STRING)

    for j in range(num_splits):
        basic_score_table_xarray = evaluation.concat_basic_score_tables(
            basic_score_table_matrix[:, j]
        )

        print((
            'Computing advanced scores for {0:d}th of {1:d} splits...'
        ).format(
            j + 1, num_splits
        ))

        if desired_month is not None:
            this_event_freq = (
                climo_dict[climatology_io.EVENT_FREQ_BY_MONTH_KEY][
                    desired_month - 1
                    ]
            )
        elif split_by_hour:
            this_event_freq = (
                climo_dict[climatology_io.EVENT_FREQ_BY_HOUR_KEY][j]
            )
        else:
            this_event_freq = (
                climo_dict[climatology_io.EVENT_FREQ_OVERALL_KEY]
            )

        advanced_score_table_xarray = (
            evaluation.get_advanced_scores_ungridded(
                basic_score_table_xarray=basic_score_table_xarray,
                training_event_frequency=this_event_freq,
                num_bootstrap_reps=num_bootstrap_reps
            )
        )
        print(advanced_score_table_xarray)

        output_file_name = evaluation.find_advanced_score_file(
            directory_name=output_dir_name,
            month=desired_month, hour=j if split_by_hour else None,
            gridded=False, raise_error_if_missing=False
        )

        print('\nWriting advanced scores to: "{0:s}"...'.format(
            output_file_name
        ))
        evaluation.write_advanced_score_file(
            advanced_score_table_xarray=advanced_score_table_xarray,
            pickle_file_name=output_file_name
        )

        if j != num_splits - 1:
            print(MINOR_SEPARATOR_STRING)


def _run(top_basic_score_dir_name, first_date_string, last_date_string,
         num_bootstrap_reps, use_partial_grids, desired_month, split_by_hour,
         climo_file_name, output_dir_name):
    """Computes advanced eval scores sans grid (combined over full domain).

    This is effectively the main method.

    :param top_basic_score_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param num_bootstrap_reps: Same.
    :param desired_month: Same.
    :param split_by_hour: Same.
    :param climo_file_name: Same.
    :param output_dir_name: Same.
    """

    if desired_month <= 0:
        desired_month = None

    if desired_month is not None:
        split_by_hour = False
        error_checking.assert_is_geq(desired_month, 1)
        error_checking.assert_is_leq(desired_month, 12)

    if not use_partial_grids:
        _compute_scores_full_grid(
            top_basic_score_dir_name=top_basic_score_dir_name,
            first_date_string=first_date_string,
            last_date_string=last_date_string,
            num_bootstrap_reps=num_bootstrap_reps,
            desired_month=desired_month, split_by_hour=split_by_hour,
            climo_file_name=climo_file_name, output_dir_name=output_dir_name
        )

        return

    _compute_scores_partial_grids(
        top_basic_score_dir_name=top_basic_score_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        num_bootstrap_reps=num_bootstrap_reps,
        climo_file_name=climo_file_name, output_dir_name=output_dir_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_basic_score_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        num_bootstrap_reps=getattr(INPUT_ARG_OBJECT, NUM_BOOTSTRAP_ARG_NAME),
        use_partial_grids=bool(
            getattr(INPUT_ARG_OBJECT, USE_PARTIAL_GRIDS_ARG_NAME)
        ),
        desired_month=getattr(INPUT_ARG_OBJECT, MONTH_ARG_NAME),
        split_by_hour=bool(getattr(INPUT_ARG_OBJECT, SPLIT_BY_HOUR_ARG_NAME)),
        climo_file_name=getattr(INPUT_ARG_OBJECT, CLIMO_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
