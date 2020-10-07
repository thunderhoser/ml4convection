"""Computes advanced evaluation scores."""

import os
import sys
import copy
import argparse
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import error_checking
import climatology_io
import evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

NUM_HOURS_PER_DAY = 24

INPUT_DIR_ARG_NAME = 'input_basic_score_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
MONTH_ARG_NAME = 'desired_month'
SPLIT_BY_HOUR_ARG_NAME = 'split_by_hour'
GRIDDED_ARG_NAME = 'gridded'
CLIMO_FILE_ARG_NAME = 'input_climo_file_name'
NUM_SUBGRIDS_ARG_NAME = 'num_subgrids_per_dim'
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

MONTH_HELP_STRING = (
    'Will evaluate predictions only for this month (integer in 1...12).  To '
    'evaluate predictions for all months, leave this alone.'
)
SPLIT_BY_HOUR_HELP_STRING = (
    '[used only if `{0:s}` is left alone] Boolean flag.  If 1, will split '
    'evaluation by hour, writing one file for each hour of the day.  If 0, will'
    ' evaluate predictions for all hours.'
).format(MONTH_ARG_NAME)

GRIDDED_HELP_STRING = (
    'Boolean flag.  If 1, scores will be gridded (one set for each pixel).  If '
    '0, scores will be aggregated (one set for the full domain).'
)
CLIMO_FILE_HELP_STRING = (
    '[used only if `{0:s}` = 0] Path to file with climatology (event '
    'frequencies in training data).'
).format(GRIDDED_ARG_NAME)

NUM_SUBGRIDS_HELP_STRING = (
    '[used only if `{0:s}` = 0] Number of subgrids per dimension.  Basic scores'
    ' will be read in K^2 pieces, where K = `{1:s}`.'
).format(GRIDDED_ARG_NAME, NUM_SUBGRIDS_ARG_NAME)

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
    '--' + MONTH_ARG_NAME, type=int, required=False, default=-1,
    help=MONTH_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SPLIT_BY_HOUR_ARG_NAME, type=int, required=False, default=0,
    help=SPLIT_BY_HOUR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + GRIDDED_ARG_NAME, type=int, required=True, help=GRIDDED_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CLIMO_FILE_ARG_NAME, type=str, required=False, default='',
    help=CLIMO_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_SUBGRIDS_ARG_NAME, type=int, required=False, default=3,
    help=NUM_SUBGRIDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _get_subgrid_indices(num_grid_rows, num_grid_columns, num_subgrids_per_dim):
    """Returns indices (first/last row and column) for each subgrid.

    K = number of subgrids per dimension

    :param num_grid_rows: Number of rows in full grid.
    :param num_grid_columns: Number of columns in full grid.
    :param num_subgrids_per_dim: Number of subgrids per dimension.
    :return: start_row_indices: length-K numpy array of indices.
    :return: end_row_indices: Same.
    :return: start_column_indices: Same.
    :return: end_column_indices: Same.
    """

    row_indices = numpy.linspace(
        0, num_grid_rows - 1, num=num_subgrids_per_dim + 1, dtype=float
    )
    row_indices = numpy.round(row_indices).astype(int)
    end_row_indices = row_indices[1:]
    start_row_indices = row_indices[:-1] + 1
    start_row_indices[0] = 0

    column_indices = numpy.linspace(
        0, num_grid_columns - 1, num=num_subgrids_per_dim + 1, dtype=float
    )
    column_indices = numpy.round(column_indices).astype(int)
    end_column_indices = column_indices[1:]
    start_column_indices = column_indices[:-1] + 1
    start_column_indices[0] = 0

    return (
        start_row_indices, end_row_indices,
        start_column_indices, end_column_indices
    )


def _read_basic_scores_gridded(basic_score_file_names, num_subgrids_per_dim):
    """Reads basic scores on grid.

    :param basic_score_file_names: 1-D list of paths to input files (will be
        read by `evaluation.read_basic_score_file`).
    :param num_subgrids_per_dim: See documentation at top of file.
    :return: basic_score_table_xarray: xarray table in format returned by
        `evaluation.read_basic_score_file`.
    """

    first_score_table = evaluation.read_basic_score_file(
        basic_score_file_names[0]
    )

    num_grid_rows = len(
        first_score_table.coords[evaluation.LATITUDE_DIM].values
    )
    num_grid_columns = len(
        first_score_table.coords[evaluation.LONGITUDE_DIM].values
    )

    error_checking.assert_is_geq(num_subgrids_per_dim, 0)
    error_checking.assert_is_leq(
        num_subgrids_per_dim, min([num_grid_rows, num_grid_columns])
    )

    (
        start_row_indices, end_row_indices,
        start_column_indices, end_column_indices
    ) = _get_subgrid_indices(
        num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns,
        num_subgrids_per_dim=num_subgrids_per_dim
    )

    basic_score_table_matrix = numpy.full(
        (num_subgrids_per_dim, num_subgrids_per_dim), '', dtype=object
    )
    num_dates = len(basic_score_file_names)

    for j in range(num_subgrids_per_dim):
        for k in range(num_subgrids_per_dim):
            these_score_tables = [None] * num_dates

            for i in range(num_dates):
                print('Reading basic scores from: "{0:s}"...'.format(
                    basic_score_file_names[i]
                ))
                these_score_tables[i] = evaluation.read_basic_score_file(
                    basic_score_file_names[i]
                )

                print((
                    'Subsetting rows {0:d}-{1:d} and columns {2:d}-{3:d}...'
                ).format(
                    start_row_indices[j], end_row_indices[j],
                    start_column_indices[k], end_column_indices[k]
                ))

                these_score_tables[i] = evaluation.subset_basic_scores_by_space(
                    basic_score_table_xarray=these_score_tables[i],
                    first_grid_row=start_row_indices[j],
                    last_grid_row=end_row_indices[j],
                    first_grid_column=start_column_indices[k],
                    last_grid_column=end_column_indices[k]
                )

            basic_score_table_matrix[j, k] = (
                evaluation.concat_basic_score_tables(these_score_tables)
            )
            del these_score_tables

            if j == k == num_subgrids_per_dim - 1:
                pass

            print(MINOR_SEPARATOR_STRING)

    these_tables = [
        xarray.concat(
            objs=basic_score_table_matrix[j, ...].tolist(),
            dim=evaluation.LONGITUDE_DIM
        )
        for j in range(num_subgrids_per_dim)
    ]

    return xarray.concat(objs=these_tables, dim=evaluation.LATITUDE_DIM)


def _run(top_basic_score_dir_name, first_date_string, last_date_string,
         desired_month, split_by_hour, gridded, climo_file_name,
         num_subgrids_per_dim, output_dir_name):
    """Computes advanced evaluation scores.

    This is effectively the main method.

    :param top_basic_score_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param desired_month: Same.
    :param split_by_hour: Same.
    :param gridded: Same.
    :param climo_file_name: Same.
    :param num_subgrids_per_dim: Same.
    :param output_dir_name: Same.
    """

    if desired_month <= 0:
        desired_month = None

    if desired_month is not None:
        split_by_hour = False
        error_checking.assert_is_geq(desired_month, 1)
        error_checking.assert_is_leq(desired_month, 12)

    if desired_month is not None or split_by_hour:
        gridded = False

    if gridded:
        climo_dict = dict()
        error_checking.assert_is_geq(num_subgrids_per_dim, 1)
    else:
        print('Reading event frequencies from: "{0:s}"...'.format(
            climo_file_name
        ))
        climo_dict = climatology_io.read_file(climo_file_name)

    basic_score_file_names = evaluation.find_many_basic_score_files(
        top_directory_name=top_basic_score_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        gridded=gridded, raise_error_if_any_missing=False
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

    # TODO(thunderhoser): Split this into two scripts, one for gridded scores
    # and one for ungridded.
    if gridded:
        basic_score_table_xarray = _read_basic_scores_gridded(
            basic_score_file_names=basic_score_file_names,
            num_subgrids_per_dim=num_subgrids_per_dim
        )
        print(SEPARATOR_STRING)
        print(basic_score_table_xarray)

        advanced_score_table_xarray = evaluation.get_advanced_scores_gridded(
            basic_score_table_xarray=basic_score_table_xarray
        )
        print(advanced_score_table_xarray)

        output_file_name = evaluation.find_advanced_score_file(
            directory_name=output_dir_name,
            month=None, hour=None, gridded=True, raise_error_if_missing=False
        )

        print('\nWriting advanced scores to: "{0:s}"...'.format(
            output_file_name
        ))
        evaluation.write_advanced_score_file(
            advanced_score_table_xarray=advanced_score_table_xarray,
            pickle_file_name=output_file_name
        )

        return

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
                training_event_frequency=this_event_freq
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


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_basic_score_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        desired_month=getattr(INPUT_ARG_OBJECT, MONTH_ARG_NAME),
        split_by_hour=bool(getattr(INPUT_ARG_OBJECT, SPLIT_BY_HOUR_ARG_NAME)),
        gridded=bool(getattr(INPUT_ARG_OBJECT, GRIDDED_ARG_NAME)),
        climo_file_name=getattr(INPUT_ARG_OBJECT, CLIMO_FILE_ARG_NAME),
        num_subgrids_per_dim=getattr(INPUT_ARG_OBJECT, NUM_SUBGRIDS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
