"""Computes advanced evaluation scores on grid (one set of scores per px)."""

import os
import sys
import argparse
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_basic_score_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
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

NUM_SUBGRIDS_HELP_STRING = (
    'Number of subgrids per dimension.  Basic scores will be read in K^2 '
    'pieces, where K = `{0:s}`.'
).format(NUM_SUBGRIDS_ARG_NAME)

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


def _run(top_basic_score_dir_name, first_date_string, last_date_string,
         num_subgrids_per_dim, output_dir_name):
    """Computes advanced evaluation scores on grid (one set of scores per px).

    This is effectively the main method.

    :param top_basic_score_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param num_subgrids_per_dim: Same.
    :param output_dir_name: Same.
    """

    # Find input files.
    basic_score_file_names = evaluation.find_many_basic_score_files(
        top_directory_name=top_basic_score_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        gridded=True, raise_error_if_any_missing=False
    )

    # Check input args.
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

    advanced_score_table_matrix = numpy.full(
        (num_subgrids_per_dim, num_subgrids_per_dim), '', dtype=object
    )
    num_dates = len(basic_score_file_names)

    for j in range(num_subgrids_per_dim):
        for k in range(num_subgrids_per_dim):
            these_basic_score_tables = [None] * num_dates

            for i in range(num_dates):
                print('Reading basic scores from: "{0:s}"...'.format(
                    basic_score_file_names[i]
                ))
                these_basic_score_tables[i] = evaluation.read_basic_score_file(
                    basic_score_file_names[i]
                )

                print((
                    'Subsetting rows {0:d}-{1:d} and columns {2:d}-{3:d}...'
                ).format(
                    start_row_indices[j], end_row_indices[j],
                    start_column_indices[k], end_column_indices[k]
                ))

                these_basic_score_tables[i] = (
                    evaluation.subset_basic_scores_by_space(
                        basic_score_table_xarray=these_basic_score_tables[i],
                        first_grid_row=start_row_indices[j],
                        last_grid_row=end_row_indices[j],
                        first_grid_column=start_column_indices[k],
                        last_grid_column=end_column_indices[k]
                    )
                )

            print(MINOR_SEPARATOR_STRING)

            this_basic_score_table = (
                evaluation.concat_basic_score_tables(these_basic_score_tables)
            )
            del these_basic_score_tables

            advanced_score_table_matrix[j, k] = (
                evaluation.get_advanced_scores_gridded(
                    basic_score_table_xarray=this_basic_score_table
                )
            )
            del this_basic_score_table
            print(advanced_score_table_matrix[j, k])

            if j == k == num_subgrids_per_dim - 1:
                print(SEPARATOR_STRING)
            else:
                print(MINOR_SEPARATOR_STRING)

    print('Concatenating advanced scores over longitude...')
    advanced_score_table_by_lng = [
        xarray.concat(
            objs=advanced_score_table_matrix[j, ...].tolist(),
            dim=evaluation.LONGITUDE_DIM
        )
        for j in range(num_subgrids_per_dim)
    ]
    del advanced_score_table_matrix

    print('Concatenating advanced scores over latitude...\n')
    advanced_score_table_xarray = xarray.concat(
        objs=advanced_score_table_by_lng, dim=evaluation.LATITUDE_DIM
    )
    del advanced_score_table_by_lng
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


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_basic_score_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        num_subgrids_per_dim=getattr(INPUT_ARG_OBJECT, NUM_SUBGRIDS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
