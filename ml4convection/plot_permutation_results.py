"""Plots results of permutation-based importance test."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import gg_permutation
import gg_plotting_utils
import permutation_plotting
import permutation

BAR_FACE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
FIGURE_RESOLUTION_DPI = 300

PREDICTOR_NAME_TO_VERBOSE = {
    'Band 8': r'Band 8 (6.25 $\mu$m)',
    'Band 9': r'Band 9 (6.95 $\mu$m)',
    'Band 10': r'Band 10 (7.35 $\mu$m)',
    'Band 11': r'Band 11 (8.60 $\mu$m)',
    'Band 13': r'Band 13 (10.45 $\mu$m)',
    'Band 14': r'Band 14 (11.20 $\mu$m)',
    'Band 16': r'Band 16 (13.30 $\mu$m)'
}

INPUT_FILE_ARG_NAME = 'input_file_name'
NUM_PREDICTORS_ARG_NAME = 'num_predictors_to_plot'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (will be read by `permutation.read_file` in the '
    'ml4convection library).'
)
NUM_PREDICTORS_HELP_STRING = (
    'Will plot only the `{0:s}` most important predictors in each figure.  To '
    'plot all predictors, leave this argument alone.'
).format(NUM_PREDICTORS_ARG_NAME)

CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level for error bars (in range 0...1).'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PREDICTORS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_PREDICTORS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _results_to_gg_format(permutation_dict):
    """Converts permutation results from ml4rt format to GewitterGefahr format.

    :param permutation_dict: Dictionary created by `run_forward_test` or
        `run_backwards_test` in `ml4rt.machine_learning.permutation`.
    :return: permutation_dict: Same but in format created by `run_forward_test`
        or `run_backwards_test` in `gewittergefahr.deep_learning.permutation`.
    """

    permutation_dict[gg_permutation.ORIGINAL_COST_ARRAY_KEY] = (
        permutation_dict[permutation.ORIGINAL_COST_KEY]
    )

    permutation_dict[gg_permutation.BACKWARDS_FLAG] = (
        permutation_dict[permutation.BACKWARDS_FLAG_KEY]
    )

    permutation_dict[gg_permutation.BEST_PREDICTORS_KEY] = [
        PREDICTOR_NAME_TO_VERBOSE[s] for s in
        permutation_dict[permutation.BEST_PREDICTORS_KEY]
    ]

    permutation_dict[gg_permutation.STEP1_PREDICTORS_KEY] = [
        PREDICTOR_NAME_TO_VERBOSE[s] for s in
        permutation_dict[permutation.STEP1_PREDICTORS_KEY]
    ]

    return permutation_dict


def _run(input_file_name, num_predictors_to_plot, confidence_level,
         output_dir_name):
    """Plots results of permutation-based importance test.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param num_predictors_to_plot: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    if num_predictors_to_plot <= 0:
        num_predictors_to_plot = None

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    permutation_dict = permutation.read_file(input_file_name)
    permutation_dict = _results_to_gg_format(permutation_dict)

    figure_object, axes_object_matrix = gg_plotting_utils.create_paneled_figure(
        num_rows=1, num_columns=2, shared_x_axis=False, shared_y_axis=True,
        keep_aspect_ratio=False, horizontal_spacing=0.1, vertical_spacing=0.05
    )
    permutation_plotting.plot_single_pass_test(
        permutation_dict=permutation_dict, axes_object=axes_object_matrix[0, 0],
        num_predictors_to_plot=num_predictors_to_plot,
        plot_percent_increase=False, confidence_level=confidence_level,
        bar_face_colour=BAR_FACE_COLOUR
    )
    axes_object_matrix[0, 0].set_title('Single-pass test')
    axes_object_matrix[0, 0].set_xlabel('Negative FSS')

    permutation_plotting.plot_multipass_test(
        permutation_dict=permutation_dict, axes_object=axes_object_matrix[0, 1],
        num_predictors_to_plot=num_predictors_to_plot,
        plot_percent_increase=False, confidence_level=confidence_level,
        bar_face_colour=BAR_FACE_COLOUR
    )
    axes_object_matrix[0, 1].set_title('Multi-pass test')
    axes_object_matrix[0, 1].set_xlabel('Negative FSS')
    axes_object_matrix[0, 1].set_ylabel('')

    figure_file_name = '{0:s}/permutation_test_abs-values.jpg'.format(
        output_dir_name
    )

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object_matrix = gg_plotting_utils.create_paneled_figure(
        num_rows=1, num_columns=2, shared_x_axis=False, shared_y_axis=True,
        keep_aspect_ratio=False, horizontal_spacing=0.1, vertical_spacing=0.05
    )
    permutation_plotting.plot_single_pass_test(
        permutation_dict=permutation_dict, axes_object=axes_object_matrix[0, 0],
        num_predictors_to_plot=num_predictors_to_plot,
        plot_percent_increase=True, confidence_level=confidence_level,
        bar_face_colour=BAR_FACE_COLOUR
    )
    axes_object_matrix[0, 0].set_title('Single-pass test')
    axes_object_matrix[0, 0].set_xlabel('Negative FSS (fraction of original)')

    permutation_plotting.plot_multipass_test(
        permutation_dict=permutation_dict, axes_object=axes_object_matrix[0, 1],
        num_predictors_to_plot=num_predictors_to_plot,
        plot_percent_increase=True, confidence_level=confidence_level,
        bar_face_colour=BAR_FACE_COLOUR
    )
    axes_object_matrix[0, 1].set_title('Multi-pass test')
    axes_object_matrix[0, 1].set_xlabel('Negative FSS (fraction of original)')
    axes_object_matrix[0, 1].set_ylabel('')

    figure_file_name = '{0:s}/permutation_test_percentage.jpg'.format(
        output_dir_name
    )

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        num_predictors_to_plot=getattr(
            INPUT_ARG_OBJECT, NUM_PREDICTORS_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
