"""Plots figure comparing neigh-based evaluation scores for different models.

Specifically, each model gets 2 panels in the figure: one with the attributes
diagram, one with the performance diagram.
"""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from scipy.stats import percentileofscore
from gewittergefahr.gg_utils import model_evaluation as gg_model_eval
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4convection.utils import evaluation
from ml4convection.plotting import evaluation_plotting as eval_plotting

TOLERANCE = 1e-6

BOUNDING_BOX_DICT = {
    'facecolor': 'white',
    'alpha': 0.5,
    'edgecolor': 'black',
    'linewidth': 2,
    'boxstyle': 'round'
}

MARKER_TYPE = '*'
MARKER_SIZE = 50
MARKER_EDGE_WIDTH = 0

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
CONCAT_FIGURE_SIZE_PX = int(1e7)

DEFAULT_FONT_SIZE = 50
ANNOTATION_FONT_SIZE = 42.5

pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

INPUT_FILES_ARG_NAME = 'input_advanced_score_file_names'
MODEL_DESCRIPTIONS_ARG_NAME = 'model_description_strings'
NUM_PANEL_ROWS_ARG_NAME = 'num_panel_rows'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILES_HELP_STRING = (
    'List of file paths to advanced evaluation scores, one per model.  Each '
    'file will be read by `evaluation.read_advanced_score_file`.'
)
MODEL_DESCRIPTIONS_HELP_STRING = (
    'Space-separated list of model descriptions (one per model), to be used as '
    'panel titles.  Underscores will be replaced by spaces.'
)
NUM_PANEL_ROWS_HELP_STRING = (
    'Number of panel rows.  If you want number of rows to be determined '
    'automatically, leave this argument alone.'
)
CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence intervals (if number of bootstrap replicates > 1) will be '
    'plotted at this level.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, nargs='+', type=str, required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_DESCRIPTIONS_ARG_NAME, nargs='+', type=str, required=True,
    help=MODEL_DESCRIPTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PANEL_ROWS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_PANEL_ROWS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(advanced_score_file_names, model_descriptions_abbrev, num_panel_rows,
         confidence_level, output_dir_name):
    """Plots figure comparing neigh-based evaluation scores for diff models.

    This is effectively the main method.

    :param advanced_score_file_names: See documentation at top of file.
    :param model_descriptions_abbrev: Same.
    :param num_panel_rows: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    # Housekeeping.
    num_models = len(advanced_score_file_names)
    expected_dim = numpy.array([num_models], dtype=int)
    error_checking.assert_is_numpy_array(
        numpy.array(model_descriptions_abbrev), exact_dimensions=expected_dim
    )

    model_descriptions_verbose = [
        s.replace('_', ' ') for s in model_descriptions_abbrev
    ]
    model_descriptions_verbose = [
        s.replace('inf', r'$\infty$') for s in model_descriptions_verbose
    ]
    model_descriptions_verbose = [
        s.replace('deg', r'$^{\circ}$') for s in model_descriptions_verbose
    ]
    model_descriptions_abbrev = [
        s.replace('_', '-').lower() for s in model_descriptions_abbrev
    ]

    num_panels = 1 * num_models
    if num_panel_rows <= 0:
        num_panel_rows = int(numpy.floor(
            numpy.sqrt(num_panels)
        ))

    num_panel_columns = int(numpy.ceil(
        float(num_panels) / num_panel_rows
    ))

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Do actual stuff.
    panel_file_names = [''] * num_panels
    letter_label = None

    bss_values_by_model = [numpy.array([])] * num_models
    reliability_values_by_model = [numpy.array([])] * num_models

    for i in range(num_models):
        print('Reading data from: "{0:s}"...'.format(
            advanced_score_file_names[i]
        ))
        advanced_score_table_xarray = evaluation.read_advanced_score_file(
            advanced_score_file_names[i]
        )
        a = advanced_score_table_xarray

        # Plot attributes diagram for [i]th model.
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )
        eval_plotting.plot_attributes_diagram(
            figure_object=figure_object, axes_object=axes_object,
            mean_prediction_matrix=a[evaluation.BINNED_MEAN_PROBS_KEY].values,
            mean_observation_matrix=a[evaluation.BINNED_EVENT_FREQS_KEY].values,
            example_counts=a[evaluation.BINNED_NUM_EXAMPLES_KEY].values,
            mean_value_in_training=
            a[evaluation.TRAINING_EVENT_FREQ_KEY].values[0],
            confidence_level=confidence_level,
            min_value_to_plot=0., max_value_to_plot=1.
        )

        this_row = int(numpy.floor(
            float(i) / num_panel_columns
        ))
        this_column = int(numpy.round(
            numpy.mod(i, num_panel_columns)
        ))

        if this_column != 0:
            num_y_ticks = len(axes_object.get_yticks())
            axes_object.set_yticklabels([''] * num_y_ticks)
            axes_object.set_ylabel('')

        if this_row != num_panel_rows - 1:
            num_x_ticks = len(axes_object.get_xticks())
            axes_object.set_xticklabels([''] * num_x_ticks)
            axes_object.set_xlabel('')

        axes_object.set_title('Attrib diagram for {0:s}'.format(
            model_descriptions_verbose[i]
        ))

        bss_values_by_model[i] = a[evaluation.BRIER_SKILL_SCORE_KEY].values
        reliability_values_by_model[i] = a[evaluation.RELIABILITY_KEY].values

        for j in range(i):
            percentile_level = 0.01 * percentileofscore(
                a=bss_values_by_model[i] - bss_values_by_model[j],
                score=0., kind='mean'
            )
            if percentile_level > 0.5:
                percentile_level = 1 - percentile_level

            p_value = 2 * percentile_level

            print((
                'p-value for BSS difference between models {0:s} and {1:s} = '
                '{2:.4f}'
            ).format(
                model_descriptions_abbrev[i], model_descriptions_abbrev[j],
                p_value
            ))

            percentile_level = 0.01 * percentileofscore(
                a=(
                        reliability_values_by_model[i] -
                        reliability_values_by_model[j]
                ),
                score=0., kind='mean'
            )
            if percentile_level > 0.5:
                percentile_level = 1 - percentile_level

            p_value = 2 * percentile_level

            print((
                'p-value for REL difference between models {0:s} and {1:s} = '
                '{2:.4f}'
            ).format(
                model_descriptions_abbrev[i], model_descriptions_abbrev[j],
                p_value
            ))

        annotation_string = (
            'REL = {0:.5f} ({1:.5f} to {2:.5f})\n'
            'BSS = {3:.3f} ({4:.3f} to {5:.3f})'
        ).format(
            numpy.nanmean(reliability_values_by_model[i]),
            numpy.nanpercentile(
                reliability_values_by_model[i], 50 * (1 - confidence_level)
            ),
            numpy.nanpercentile(
                reliability_values_by_model[i], 50 * (1 + confidence_level)
            ),
            numpy.nanmean(bss_values_by_model[i]),
            numpy.nanpercentile(
                bss_values_by_model[i], 50 * (1 - confidence_level)
            ),
            numpy.nanpercentile(
                bss_values_by_model[i], 50 * (1 + confidence_level)
            )
        )
        axes_object.text(
            0.98, 0.02, annotation_string, bbox=BOUNDING_BOX_DICT, color='k',
            horizontalalignment='right', verticalalignment='bottom',
            transform=axes_object.transAxes, fontsize=ANNOTATION_FONT_SIZE
        )

        if letter_label is None:
            letter_label = 'a'
        else:
            letter_label = chr(ord(letter_label) + 1)

        gg_plotting_utils.label_axes(
            axes_object=axes_object,
            label_string='({0:s})'.format(letter_label),
            x_coord_normalized=-0.04, y_coord_normalized=1.025,
            font_size=DEFAULT_FONT_SIZE
        )

        panel_file_names[i] = '{0:s}/{1:s}_attributes_diagram.jpg'.format(
            output_dir_name, model_descriptions_abbrev[i]
        )
        print('Saving figure to: "{0:s}"...'.format(panel_file_names[i]))
        figure_object.savefig(
            panel_file_names[i], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    attrib_figure_file_name = '{0:s}/neigh_eval_attrib_diagrams.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(attrib_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=attrib_figure_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=attrib_figure_file_name,
        output_file_name=attrib_figure_file_name,
        border_width_pixels=100
    )
    imagemagick_utils.resize_image(
        input_file_name=attrib_figure_file_name,
        output_file_name=attrib_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )

    for k in range(num_panels):
        os.remove(panel_file_names[k])

    panel_file_names = [''] * num_panels
    auc_values_by_model = [numpy.array([])] * num_models

    for i in range(num_models):
        print('Reading data from: "{0:s}"...'.format(
            advanced_score_file_names[i]
        ))
        advanced_score_table_xarray = evaluation.read_advanced_score_file(
            advanced_score_file_names[i]
        )
        a = advanced_score_table_xarray

        # Plot performance diagram for [i]th model.
        this_row = int(numpy.floor(
            float(i) / num_panel_columns
        ))
        this_column = int(numpy.round(
            numpy.mod(i, num_panel_columns)
        ))

        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )
        eval_plotting.plot_performance_diagram(
            axes_object=axes_object,
            pod_matrix=a[evaluation.POD_KEY].values,
            success_ratio_matrix=a[evaluation.SUCCESS_RATIO_KEY].values,
            confidence_level=confidence_level,
            plot_colour_bar=this_column == num_panel_columns - 1
        )

        if this_column != 0:
            num_y_ticks = len(axes_object.get_yticks())
            axes_object.set_yticklabels([''] * num_y_ticks)
            axes_object.set_ylabel('')

        if this_row != num_panel_rows - 1:
            num_x_ticks = len(axes_object.get_xticks())
            axes_object.set_xticklabels([''] * num_x_ticks)
            axes_object.set_xlabel('')

        num_bootstrap_reps = a[evaluation.POD_KEY].values.shape[0]

        auc_values_by_model[i] = numpy.array([
            gg_model_eval.get_area_under_perf_diagram(
                pod_by_threshold=a[evaluation.POD_KEY].values[j, :],
                success_ratio_by_threshold=
                a[evaluation.SUCCESS_RATIO_KEY].values[j, :]
            )
            for j in range(num_bootstrap_reps)
        ])

        for j in range(i):
            percentile_level = 0.01 * percentileofscore(
                a=auc_values_by_model[i] - auc_values_by_model[j],
                score=0., kind='mean'
            )
            if percentile_level > 0.5:
                percentile_level = 1 - percentile_level

            p_value = 2 * percentile_level

            print((
                'p-value for AUC difference between models {0:s} and {1:s} = '
                '{2:.4f}'
            ).format(
                model_descriptions_abbrev[i], model_descriptions_abbrev[j],
                p_value
            ))

        mean_frequency_biases = numpy.nanmean(
            a[evaluation.FREQUENCY_BIAS_KEY].values, axis=0
        )
        mean_csi_values = numpy.nanmean(a[evaluation.CSI_KEY].values, axis=0)

        best_threshold_index = numpy.nanargmin(
            numpy.absolute(mean_frequency_biases - 1.)
        )
        max_csi = numpy.nanmax(mean_csi_values)
        csi_at_best_threshold = mean_csi_values[best_threshold_index]
        if csi_at_best_threshold < 0.9 * max_csi:
            best_threshold_index = numpy.nanargmax(mean_csi_values)

        best_prob_threshold = a.coords[
            evaluation.PROBABILITY_THRESHOLD_DIM
        ].values[best_threshold_index]
        csi_at_best_threshold = mean_csi_values[best_threshold_index]
        bias_at_best_threshold = mean_frequency_biases[best_threshold_index]

        annotation_string = 'AUC = {0:.3f} ({1:.3f} to {2:.3f})'.format(
            numpy.mean(auc_values_by_model[i]),
            numpy.percentile(
                auc_values_by_model[i], 50 * (1 - confidence_level)
            ),
            numpy.percentile(
                auc_values_by_model[i], 50 * (1 + confidence_level)
            )
        )

        axes_object.text(
            0.98, 0.02, annotation_string, bbox=BOUNDING_BOX_DICT, color='k',
            horizontalalignment='right', verticalalignment='bottom',
            transform=axes_object.transAxes, fontsize=ANNOTATION_FONT_SIZE
        )

        mean_success_ratios = numpy.nanmean(
            a[evaluation.SUCCESS_RATIO_KEY].values, axis=0
        )
        mean_pod_values = numpy.nanmean(a[evaluation.POD_KEY].values, axis=0)
        # axes_object.plot(
        #     mean_success_ratios[best_threshold_index],
        #     mean_pod_values[best_threshold_index],
        #     linestyle='None', marker=MARKER_TYPE, markersize=MARKER_SIZE,
        #     markeredgewidth=MARKER_EDGE_WIDTH,
        #     markerfacecolor=eval_plotting.PERF_DIAGRAM_COLOUR,
        #     markeredgecolor=eval_plotting.PERF_DIAGRAM_COLOUR
        # )

        axes_object.set_title('Perf diagram for {0:s}'.format(
            model_descriptions_verbose[i]
        ))

        letter_label = chr(ord(letter_label) + 1)
        gg_plotting_utils.label_axes(
            axes_object=axes_object,
            label_string='({0:s})'.format(letter_label),
            x_coord_normalized=-0.04, y_coord_normalized=1.025,
            font_size=DEFAULT_FONT_SIZE
        )

        panel_file_names[i] = (
            '{0:s}/{1:s}_performance_diagram.jpg'
        ).format(output_dir_name, model_descriptions_abbrev[i])

        print('Saving figure to: "{0:s}"...'.format(
            panel_file_names[i]
        ))
        figure_object.savefig(
            panel_file_names[i], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    perf_figure_file_name = '{0:s}/neigh_eval_perf_diagrams.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(perf_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=perf_figure_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=perf_figure_file_name,
        output_file_name=perf_figure_file_name,
        border_width_pixels=100
    )
    imagemagick_utils.resize_image(
        input_file_name=perf_figure_file_name,
        output_file_name=perf_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )

    for k in range(num_panels):
        os.remove(panel_file_names[k])

    concat_figure_file_name = '{0:s}/neigh_eval_comparison.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=[attrib_figure_file_name, perf_figure_file_name],
        output_file_name=concat_figure_file_name,
        num_panel_rows=2, num_panel_columns=1
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        advanced_score_file_names=getattr(
            INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME
        ),
        model_descriptions_abbrev=getattr(
            INPUT_ARG_OBJECT, MODEL_DESCRIPTIONS_ARG_NAME
        ),
        num_panel_rows=getattr(INPUT_ARG_OBJECT, NUM_PANEL_ROWS_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
