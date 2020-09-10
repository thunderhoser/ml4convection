"""Runs pixelwise (grid-point-by-grid-point) evaluation."""

import argparse
import numpy
from ml4convection.io import prediction_io
from ml4convection.utils import pixelwise_evaluation as pixelwise_eval

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
NUM_THRESHOLDS_ARG_NAME = 'num_prob_thresholds'
NUM_BINS_ARG_NAME = 'num_bins_for_reliability'
EVENT_FREQUENCY_ARG_NAME = 'event_frequency_in_training'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`prediction_io.find_file` and read by `prediction_io.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will evaluate predictions for all days in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

NUM_THRESHOLDS_HELP_STRING = (
    'Number of probability thresholds, used to convert forecasts from '
    'probabilistic to deterministic.'
)
NUM_BINS_HELP_STRING = (
    'Number of bins for reliability info.  Data will be grouped into `{0:s}` '
    'bins, based on forecast probability.'
).format(NUM_BINS_ARG_NAME)

EVENT_FREQUENCY_HELP_STRING = (
    'Event frequency in training data, used to compute climatology component of'
    ' Brier skill score.  If you do not know this value, leave the argument '
    'alone.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Result files will be saved here.'
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
    '--' + NUM_THRESHOLDS_ARG_NAME, type=int, required=False,
    default=pixelwise_eval.DEFAULT_NUM_PROB_THRESHOLDS,
    help=NUM_THRESHOLDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BINS_ARG_NAME, type=int, required=False,
    default=pixelwise_eval.DEFAULT_NUM_RELIA_BINS, help=NUM_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EVENT_FREQUENCY_ARG_NAME, type=float, required=False, default=-1,
    help=EVENT_FREQUENCY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(top_prediction_dir_name, first_date_string, last_date_string,
         num_prob_thresholds, num_bins_for_reliability,
         event_frequency_in_training, output_dir_name):
    """Runs pixelwise (grid-point-by-grid-point) evaluation.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param num_prob_thresholds: Same.
    :param num_bins_for_reliability: Same.
    :param event_frequency_in_training: Same.
    :param output_dir_name: Same.
    """

    if event_frequency_in_training <= 0:
        event_frequency_in_training = None

    prediction_file_names = prediction_io.find_many_files(
        top_directory_name=top_prediction_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        raise_error_if_any_missing=False
    )

    basic_score_table_xarray = pixelwise_eval.get_basic_scores(
        prediction_file_names=prediction_file_names,
        event_frequency_in_training=event_frequency_in_training,
        num_prob_thresholds=num_prob_thresholds,
        num_bins_for_reliability=num_bins_for_reliability
    )
    print(SEPARATOR_STRING)

    advanced_score_table_xarray = pixelwise_eval.get_advanced_scores(
        basic_score_table_xarray
    )
    print(advanced_score_table_xarray)
    print(SEPARATOR_STRING)

    best_index = numpy.argmax(
        advanced_score_table_xarray[pixelwise_eval.CSI_KEY].values
    )
    a = advanced_score_table_xarray

    print((
        'Best CSI = {0:.4f} ... probability threshold = {1:.4f} ... '
        'POD = {2:.4f} ... POFD = {3:.4f} ... success ratio = {4:.4f} ... '
        'frequency bias = {5:.4f} ... accuracy = {6:.4f} ... '
        'Heidke score = {7:.4f}\n'
    ).format(
        a[pixelwise_eval.CSI_KEY].values[best_index],
        a.coords[pixelwise_eval.PROBABILITY_THRESHOLD_DIM].values[best_index],
        a[pixelwise_eval.POD_KEY].values[best_index],
        a[pixelwise_eval.POFD_KEY].values[best_index],
        a[pixelwise_eval.SUCCESS_RATIO_KEY].values[best_index],
        a[pixelwise_eval.FREQUENCY_BIAS_KEY].values[best_index],
        a[pixelwise_eval.ACCURACY_KEY].values[best_index],
        a[pixelwise_eval.HEIDKE_SCORE_KEY].values[best_index]
    ))

    basic_file_name = '{0:s}/basic_scores.nc'.format(output_dir_name)
    print('Writing results to: "{0:s}"...'.format(basic_file_name))
    pixelwise_eval.write_file(
        score_table_xarray=basic_score_table_xarray,
        netcdf_file_name=basic_file_name
    )

    advanced_file_name = '{0:s}/advanced_scores.nc'.format(output_dir_name)
    print('Writing results to: "{0:s}"...'.format(advanced_file_name))
    pixelwise_eval.write_file(
        score_table_xarray=advanced_score_table_xarray,
        netcdf_file_name=advanced_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        num_prob_thresholds=getattr(INPUT_ARG_OBJECT, NUM_THRESHOLDS_ARG_NAME),
        num_bins_for_reliability=getattr(INPUT_ARG_OBJECT, NUM_BINS_ARG_NAME),
        event_frequency_in_training=getattr(
            INPUT_ARG_OBJECT, EVENT_FREQUENCY_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
