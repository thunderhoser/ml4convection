"""Plots results of pixelwise evaluation."""

import os
import sys
import argparse
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import pixelwise_evaluation as pixelwise_eval
import evaluation_plotting as eval_plotting

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

BASIC_SCORE_FILE_ARG_NAME = 'input_basic_score_file_name'
ADVANCED_SCORE_FILE_ARG_NAME = 'input_advanced_score_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

BASIC_SCORE_FILE_HELP_STRING = (
    'Path to file with basic scores for pixelwise evaluation.  Will be read by '
    '`pixelwise_evaluation.read_file`.'
)
ADVANCED_SCORE_FILE_HELP_STRING = (
    'Path to file with advanced scores for pixelwise evaluation.  Will be read '
    'by `pixelwise_evaluation.read_file`.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + BASIC_SCORE_FILE_ARG_NAME, type=str, required=True,
    help=BASIC_SCORE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ADVANCED_SCORE_FILE_ARG_NAME, type=str, required=True,
    help=ADVANCED_SCORE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(basic_score_file_name, advanced_score_file_name, output_dir_name):
    """Plots results of pixelwise evaluation.

    This is effectively the main method.

    :param basic_score_file_name: See documentation at top of file.
    :param advanced_score_file_name: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(basic_score_file_name))
    basic_score_table_xarray = pixelwise_eval.read_file(basic_score_file_name)

    print('Reading data from: "{0:s}"...'.format(advanced_score_file_name))
    advanced_score_table_xarray = pixelwise_eval.read_file(
        advanced_score_file_name
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    eval_plotting.plot_roc_curve(
        axes_object=axes_object,
        pod_by_threshold=
        advanced_score_table_xarray[pixelwise_eval.POD_KEY].values,
        pofd_by_threshold=
        advanced_score_table_xarray[pixelwise_eval.POFD_KEY].values
    )

    figure_file_name = '{0:s}/roc_curve.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    eval_plotting.plot_performance_diagram(
        axes_object=axes_object,
        pod_by_threshold=
        advanced_score_table_xarray[pixelwise_eval.POD_KEY].values,
        success_ratio_by_threshold=
        advanced_score_table_xarray[pixelwise_eval.SUCCESS_RATIO_KEY].values
    )

    figure_file_name = '{0:s}/performance_diagram.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    eval_plotting.plot_attributes_diagram(
        figure_object=figure_object, axes_object=axes_object,
        mean_predictions=
        basic_score_table_xarray[pixelwise_eval.MEAN_FORECAST_PROB_KEY].values,
        mean_observations=
        basic_score_table_xarray[pixelwise_eval.EVENT_FREQUENCY_KEY].values,
        example_counts=
        basic_score_table_xarray[pixelwise_eval.NUM_EXAMPLES_KEY].values,
        mean_value_in_training=
        basic_score_table_xarray.attrs[pixelwise_eval.CLIMO_EVENT_FREQ_KEY],
        min_value_to_plot=0., max_value_to_plot=1.
    )

    figure_file_name = '{0:s}/attributes_diagram.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        basic_score_file_name=getattr(
            INPUT_ARG_OBJECT, BASIC_SCORE_FILE_ARG_NAME
        ),
        advanced_score_file_name=getattr(
            INPUT_ARG_OBJECT, ADVANCED_SCORE_FILE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
