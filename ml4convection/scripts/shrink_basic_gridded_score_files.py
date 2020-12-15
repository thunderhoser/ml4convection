"""Shrinks files with basic gridded scores.

USE ONCE AND DESTROY.
"""

import argparse
import numpy
from ml4convection.utils import evaluation

INPUT_DIR_ARG_NAME = 'input_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein (containing basic scores) will be '
    'found by `evaluation.find_basic_score_file` and read by '
    '`evaluation.read_basic_score_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will shrink files for all days in the period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results (basic scores on smaller grids) will be'
    ' written here by `evaluation.write_basic_score_file`, to exact locations '
    'determined by `evaluation.find_basic_score_file`.'
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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(top_input_dir_name, first_date_string, last_date_string,
         top_output_dir_name):
    """Shrinks files with basic gridded scores.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param top_output_dir_name: Same.
    """

    # Find input files.
    input_file_names = evaluation.find_many_basic_score_files(
        top_directory_name=top_input_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        gridded=True, radar_number=None, raise_error_if_any_missing=False
    )

    date_strings = [
        evaluation.basic_file_name_to_date(f) for f in input_file_names
    ]

    output_file_names = [
        evaluation.find_basic_score_file(
            top_directory_name=top_output_dir_name, valid_date_string=d,
            gridded=True, radar_number=None, raise_error_if_missing=False
        ) for d in date_strings
    ]

    unmasked_row_flags = None
    first_unmasked_row = None
    last_unmasked_row = None
    unmasked_column_flags = None
    first_unmasked_column = None
    last_unmasked_column = None

    for i in range(len(input_file_names)):
        print('\nReading data from: "{0:s}"...'.format(input_file_names[i]))
        basic_score_table_xarray = evaluation.read_basic_score_file(
            input_file_names[i]
        )

        if first_unmasked_row is None:
            unmasked_row_flags = numpy.any(numpy.invert(numpy.isnan(
                basic_score_table_xarray[evaluation.ACTUAL_SSE_FOR_FSS_KEY]
            )),
                axis=(0, 2)
            )

            first_unmasked_row = numpy.where(unmasked_row_flags)[0][0]
            last_unmasked_row = numpy.where(unmasked_row_flags)[0][-1]

            unmasked_column_flags = numpy.any(numpy.invert(numpy.isnan(
                basic_score_table_xarray[evaluation.ACTUAL_SSE_FOR_FSS_KEY]
            )),
                axis=(0, 1)
            )

            first_unmasked_column = numpy.where(unmasked_column_flags)[0][0]
            last_unmasked_column = numpy.where(unmasked_column_flags)[0][-1]

        print((
            'Subsetting to rows {0:d}-{1:d} of {2:d}, columns {3:d}-{4:d} of '
            '{5:d}...'
        ).format(
            first_unmasked_row + 1, last_unmasked_row + 1,
            len(unmasked_row_flags),
            first_unmasked_column + 1, last_unmasked_column + 1,
            len(unmasked_column_flags)
        ))

        basic_score_table_xarray = evaluation.subset_basic_scores_by_space(
            basic_score_table_xarray=basic_score_table_xarray,
            first_grid_row=first_unmasked_row,
            last_grid_row=last_unmasked_row,
            first_grid_column=first_unmasked_column,
            last_grid_column=last_unmasked_column
        )

        print('Writing smaller grids to: "{0:s}"...'.format(
            output_file_names[i]
        ))
        evaluation.write_basic_score_file(
            basic_score_table_xarray=basic_score_table_xarray,
            netcdf_file_name=output_file_names[i]
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
