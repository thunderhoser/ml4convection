"""Computes basic evaluation scores on grid (one set of scores per px)."""

import argparse
import numpy
from ml4convection.io import prediction_io
from ml4convection.io import climatology_io
from ml4convection.utils import evaluation
from ml4convection.utils import radar_utils

# TODO(thunderhoser): Need better baselines than climo.

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_RADARS = len(radar_utils.RADAR_LATITUDES_DEG_N)

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
MATCHING_DISTANCES_ARG_NAME = 'matching_distances_px'
CLIMO_FILES_ARG_NAME = 'climo_file_names'
PROB_THRESHOLDS_ARG_NAME = 'prob_thresholds'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`prediction_io.find_file` and read by `prediction_io.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will evaluate predictions for all days in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

MATCHING_DISTANCES_HELP_STRING = (
    'List of matching distances (pixels).  Neighbourhood evaluation will be '
    'done for each matching distance.'
)
CLIMO_FILES_HELP_STRING = (
    'List of climatology files (one per matching distance).  Will be used to '
    'compute Brier skill score.'
)
PROB_THRESHOLDS_HELP_STRING = (
    'List of probability thresholds.  One contingency table will be created for'
    ' each.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results will be written here by '
    '`evaluation.write_basic_score_file`, to exact locations determined by '
    '`evaluation.find_basic_score_file`.'
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
    '--' + MATCHING_DISTANCES_ARG_NAME, type=float, nargs='+', required=True,
    help=MATCHING_DISTANCES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CLIMO_FILES_ARG_NAME, type=str, required=True, nargs='+',
    help=CLIMO_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PROB_THRESHOLDS_ARG_NAME, type=float, nargs='+', required=True,
    help=PROB_THRESHOLDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(top_prediction_dir_name, first_date_string, last_date_string,
         matching_distances_px, climo_file_names, prob_thresholds,
         top_output_dir_name):
    """Computes basic evaluation scores on grid (one set of scores per px).

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param matching_distances_px: Same.
    :param climo_file_names: Same.
    :param prob_thresholds: Same.
    :param top_output_dir_name: Same.
    """

    prediction_file_names = prediction_io.find_many_files(
        top_directory_name=top_prediction_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        radar_number=None, prefer_zipped=False, allow_other_format=True,
        raise_error_if_any_missing=False
    )
    date_strings = [
        prediction_io.file_name_to_date(f) for f in prediction_file_names
    ]

    num_dates = len(date_strings)
    num_matching_distances = len(matching_distances_px)
    event_freq_matrices = [None] * num_matching_distances

    for j in range(num_matching_distances):
        print('Reading event frequencies from: "{0:s}"...'.format(
            climo_file_names[j]
        ))
        this_climo_dict = climatology_io.read_file(climo_file_names[j])
        event_freq_matrices[j] = (
            this_climo_dict[climatology_io.EVENT_FREQ_BY_PIXEL_KEY]
        )

    first_unmasked_row = None
    last_unmasked_row = None
    first_unmasked_column = None
    last_unmasked_column = None

    for i in range(num_dates):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[i]))
        prediction_dict = prediction_io.read_file(prediction_file_names[i])

        for j in range(num_matching_distances):
            print('\n')

            basic_score_table_xarray = evaluation.get_basic_scores_gridded(
                prediction_dict=prediction_dict,
                matching_distance_px=matching_distances_px[j],
                probability_thresholds=prob_thresholds,
                training_event_freq_matrix=event_freq_matrices[j]
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

            basic_score_table_xarray = evaluation.subset_basic_scores_by_space(
                basic_score_table_xarray=basic_score_table_xarray,
                first_grid_row=first_unmasked_row,
                last_grid_row=last_unmasked_row,
                first_grid_column=first_unmasked_column,
                last_grid_column=last_unmasked_column
            )

            output_dir_name = '{0:s}/matching_distance_px={1:.6f}'.format(
                top_output_dir_name, matching_distances_px[j]
            )
            output_file_name = evaluation.find_basic_score_file(
                top_directory_name=output_dir_name,
                valid_date_string=date_strings[i],
                gridded=True, radar_number=None, raise_error_if_missing=False
            )

            print('\nWriting results to: "{0:s}"...'.format(output_file_name))
            evaluation.write_basic_score_file(
                basic_score_table_xarray=basic_score_table_xarray,
                netcdf_file_name=output_file_name
            )

        if i == num_dates - 1:
            continue

        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        matching_distances_px=numpy.array(
            getattr(INPUT_ARG_OBJECT, MATCHING_DISTANCES_ARG_NAME), dtype=float
        ),
        climo_file_names=getattr(INPUT_ARG_OBJECT, CLIMO_FILES_ARG_NAME),
        prob_thresholds=numpy.array(getattr(
            INPUT_ARG_OBJECT, PROB_THRESHOLDS_ARG_NAME
        )),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
