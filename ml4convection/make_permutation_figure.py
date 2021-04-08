"""Makes 4-panel figure to show results of permutation test."""

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
import imagemagick_utils
import permutation

BAR_FACE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255

FIGURE_WIDTH_INCHES = 15.
FIGURE_HEIGHT_INCHES = 15.
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

PREDICTOR_NAME_TO_VERBOSE = {
    'Band 8': r'Band 8 (6.25 $\mu$m)',
    'Band 9': r'Band 9 (6.95 $\mu$m)',
    'Band 10': r'Band 10 (7.35 $\mu$m)',
    'Band 11': r'Band 11 (8.60 $\mu$m)',
    'Band 13': r'Band 13 (10.45 $\mu$m)',
    'Band 14': r'Band 14 (11.20 $\mu$m)',
    'Band 16': r'Band 16 (13.30 $\mu$m)'
}

FORWARD_FILE_ARG_NAME = 'input_forward_file_name'
BACKWARDS_FILE_ARG_NAME = 'input_backwards_file_name'
NUM_PREDICTORS_ARG_NAME = 'num_predictors_to_plot'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

FORWARD_FILE_HELP_STRING = (
    'Path to file with results of forward test (will be read by '
    '`permutation.read_file` in the ml4convection library).'
)
BACKWARDS_FILE_HELP_STRING = (
    'Path to file with results of backwards test (will be read by '
    '`permutation.read_file` in the ml4convection library).'
)
NUM_PREDICTORS_HELP_STRING = (
    'Will plot only the `{0:s}` most important predictors in each panel.  To '
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
    '--' + FORWARD_FILE_ARG_NAME, type=str, required=True,
    help=FORWARD_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BACKWARDS_FILE_ARG_NAME, type=str, required=True,
    help=BACKWARDS_FILE_HELP_STRING
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


def _run(forward_file_name, backwards_file_name, num_predictors_to_plot,
         confidence_level, output_dir_name):
    """Makes 4-panel figure to show results of permutation test.

    This is effectively the main method.

    :param forward_file_name: See documentation at top of file.
    :param backwards_file_name: Same.
    :param num_predictors_to_plot: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    if num_predictors_to_plot <= 0:
        num_predictors_to_plot = None

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(forward_file_name))
    forward_permutation_dict = permutation.read_file(forward_file_name)
    forward_permutation_dict = _results_to_gg_format(forward_permutation_dict)

    print('Reading data from: "{0:s}"...'.format(backwards_file_name))
    backwards_permutation_dict = permutation.read_file(backwards_file_name)
    backwards_permutation_dict = _results_to_gg_format(
        backwards_permutation_dict
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    permutation_plotting.plot_single_pass_test(
        permutation_dict=forward_permutation_dict, axes_object=axes_object,
        num_predictors_to_plot=num_predictors_to_plot,
        plot_percent_increase=False, confidence_level=confidence_level,
        bar_face_colour=BAR_FACE_COLOUR
    )
    axes_object.set_title('Single-pass forward')
    axes_object.set_xlabel('')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(a)')

    this_file_name = '{0:s}/single_pass_forward.jpg'.format(output_dir_name)
    panel_file_names = [this_file_name]

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    permutation_plotting.plot_multipass_test(
        permutation_dict=forward_permutation_dict, axes_object=axes_object,
        num_predictors_to_plot=num_predictors_to_plot,
        plot_percent_increase=False, confidence_level=confidence_level,
        bar_face_colour=BAR_FACE_COLOUR
    )
    axes_object.set_title('Multi-pass forward')
    axes_object.set_xlabel('')
    axes_object.set_ylabel('')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(b)')

    this_file_name = '{0:s}/multi_pass_forward.jpg'.format(output_dir_name)
    panel_file_names.append(this_file_name)

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    permutation_plotting.plot_single_pass_test(
        permutation_dict=backwards_permutation_dict, axes_object=axes_object,
        num_predictors_to_plot=num_predictors_to_plot,
        plot_percent_increase=False, confidence_level=confidence_level,
        bar_face_colour=BAR_FACE_COLOUR
    )
    axes_object.set_title('Single-pass backward')
    axes_object.set_xlabel('1 - FSS')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(c)')

    this_file_name = '{0:s}/single_pass_backward.jpg'.format(output_dir_name)
    panel_file_names.append(this_file_name)

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    permutation_plotting.plot_multipass_test(
        permutation_dict=backwards_permutation_dict, axes_object=axes_object,
        num_predictors_to_plot=num_predictors_to_plot,
        plot_percent_increase=False, confidence_level=confidence_level,
        bar_face_colour=BAR_FACE_COLOUR
    )
    axes_object.set_title('Multi-pass backward')
    axes_object.set_xlabel('1 - FSS')
    axes_object.set_ylabel('')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(d)')

    this_file_name = '{0:s}/multi_pass_backward.jpg'.format(output_dir_name)
    panel_file_names.append(this_file_name)

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    concat_figure_file_name = '{0:s}/permutation_test.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=2, num_panel_columns=2
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )

    for this_file_name in panel_file_names:
        os.remove(this_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        forward_file_name=getattr(INPUT_ARG_OBJECT, FORWARD_FILE_ARG_NAME),
        backwards_file_name=getattr(INPUT_ARG_OBJECT, BACKWARDS_FILE_ARG_NAME),
        num_predictors_to_plot=getattr(
            INPUT_ARG_OBJECT, NUM_PREDICTORS_ARG_NAME
        ),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
