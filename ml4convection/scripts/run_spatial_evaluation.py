"""Runs spatially aware (not pixelwise) evaluation."""

import argparse
import numpy
from ml4convection.io import prediction_io
from ml4convection.utils import spatial_evaluation as spatial_eval

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
HALF_WINDOW_SIZES_ARG_NAME = 'fss_half_window_sizes_px'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`prediction_io.find_file` and read by `prediction_io.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will evaluate predictions for all days in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

HALF_WINDOW_SIZES_HELP_STRING = (
    'List of half-window sizes for fractions skill score (units are pixels).'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Results will be written here by '
    '`spatial_evaluation.write_file`.'
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
    '--' + HALF_WINDOW_SIZES_ARG_NAME, type=int, nargs='+', required=True,
    help=HALF_WINDOW_SIZES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(top_prediction_dir_name, first_date_string, last_date_string,
         fss_half_window_sizes_px, output_file_name):
    """Runs spatially aware (not pixelwise) evaluation.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param fss_half_window_sizes_px: Same.
    :param output_file_name: Same.
    """

    prediction_file_names = prediction_io.find_many_files(
        top_directory_name=top_prediction_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        raise_error_if_any_missing=False
    )

    fractions_skill_scores = spatial_eval.get_fractions_skill_scores(
        prediction_file_names=prediction_file_names,
        half_window_sizes_px=fss_half_window_sizes_px
    )
    print(SEPARATOR_STRING)

    num_scales = len(fss_half_window_sizes_px)
    for k in range(num_scales):
        print((
            'Fractions skill score for {0:d}-by-{0:d} window = {1:.4f}'
        ).format(
            int(numpy.round(fss_half_window_sizes_px[k] * 2)) + 1,
            fractions_skill_scores[k]
        ))

    print(SEPARATOR_STRING)
    print('Writing results to: "{0:s}"...'.format(output_file_name))
    spatial_eval.write_file(
        fractions_skill_scores=fractions_skill_scores,
        half_window_sizes_px=fss_half_window_sizes_px,
        pickle_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        fss_half_window_sizes_px=numpy.array(
            getattr(INPUT_ARG_OBJECT, HALF_WINDOW_SIZES_ARG_NAME), dtype=int
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
