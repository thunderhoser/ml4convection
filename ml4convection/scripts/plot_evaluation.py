"""Plots results of model evaluation."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import model_evaluation as gg_model_eval
from gewittergefahr.gg_utils import file_system_utils
from ml4convection.utils import evaluation
from ml4convection.plotting import evaluation_plotting as eval_plotting

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

ADVANCED_SCORE_FILE_ARG_NAME = 'input_advanced_score_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

ADVANCED_SCORE_FILE_HELP_STRING = (
    'Path to file with advanced evaluation scores.  Will be read by '
    '`evaluation.read_file`.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ADVANCED_SCORE_FILE_ARG_NAME, type=str, required=True,
    help=ADVANCED_SCORE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(advanced_score_file_name, output_dir_name):
    """Plots results of model evaluation.

    This is effectively the main method.

    :param advanced_score_file_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(advanced_score_file_name))
    advanced_score_table_xarray = evaluation.read_file(advanced_score_file_name)

    # Plot performance diagram.
    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    eval_plotting.plot_performance_diagram(
        axes_object=axes_object,
        pod_by_threshold=
        advanced_score_table_xarray[evaluation.POD_KEY].values,
        success_ratio_by_threshold=
        advanced_score_table_xarray[evaluation.SUCCESS_RATIO_KEY].values
    )

    area_under_curve = gg_model_eval.get_area_under_perf_diagram(
        pod_by_threshold=
        advanced_score_table_xarray[evaluation.POD_KEY].values,
        success_ratio_by_threshold=
        advanced_score_table_xarray[evaluation.SUCCESS_RATIO_KEY].values
    )
    max_csi = numpy.nanmax(
        advanced_score_table_xarray[evaluation.CSI_KEY].values
    )
    title_string = 'Area under curve = {0:.3f} ... max CSI = {1:.3f}'.format(
        area_under_curve, max_csi
    )
    axes_object.set_title(title_string)

    figure_file_name = '{0:s}/performance_diagram.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot attributes diagram.
    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    eval_plotting.plot_attributes_diagram(
        figure_object=figure_object, axes_object=axes_object,
        mean_predictions=
        advanced_score_table_xarray[evaluation.MEAN_FORECAST_PROB_KEY].values,
        mean_observations=
        advanced_score_table_xarray[evaluation.EVENT_FREQUENCY_KEY].values,
        example_counts=
        advanced_score_table_xarray[evaluation.NUM_EXAMPLES_KEY].values,
        mean_value_in_training=
        advanced_score_table_xarray.attrs[evaluation.TRAINING_EVENT_FREQ_KEY],
        min_value_to_plot=0., max_value_to_plot=1.
    )

    title_string = (
        'BS = {0:.3f} ... BSS = {1:.3f} ... REL = {2:.3f} ... RES = {3:.3f}'
    ).format(
        advanced_score_table_xarray.attrs[evaluation.BRIER_SCORE_KEY],
        advanced_score_table_xarray.attrs[evaluation.BRIER_SKILL_SCORE_KEY],
        advanced_score_table_xarray.attrs[evaluation.RELIABILITY_KEY],
        advanced_score_table_xarray.attrs[evaluation.RESOLUTION_KEY]
    )
    axes_object.set_title(title_string)

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
        advanced_score_file_name=getattr(
            INPUT_ARG_OBJECT, ADVANCED_SCORE_FILE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
