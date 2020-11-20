"""Computes basic evaluation scores."""

import argparse
import numpy
from gewittergefahr.gg_utils import model_evaluation as gg_model_eval
from ml4convection.io import prediction_io
from ml4convection.io import climatology_io
from ml4convection.utils import evaluation
from ml4convection.utils import radar_utils
from ml4convection.machine_learning import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_RADARS = len(radar_utils.RADAR_LATITUDES_DEG_N)

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
USE_PARTIAL_GRIDS_ARG_NAME = 'use_partial_grids'
GRIDDED_ARG_NAME = 'gridded'
MATCHING_DISTANCE_ARG_NAME = 'matching_distance_px'
NUM_PROB_THRESHOLDS_ARG_NAME = 'num_prob_thresholds'
PROB_THRESHOLDS_ARG_NAME = 'prob_thresholds'
CLIMO_FILE_ARG_NAME = 'input_climo_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`prediction_io.find_file` and read by `prediction_io.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will evaluate predictions for all days in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

USE_PARTIAL_GRIDS_HELP_STRING = (
    'Boolean flag.  If 1 (0), will compute scores for partial (full) grids.'
)
GRIDDED_HELP_STRING = (
    '[used only if `{0:s} == 0`] Boolean flag.  If 1, scores will be gridded '
    '(one set for each pixel).  If 0, scores will be aggregated (one set for '
    'the full domain).'
).format(USE_PARTIAL_GRIDS_ARG_NAME)

MATCHING_DISTANCE_HELP_STRING = (
    'Matching distance (pixels) for neighbourhood evaluation.'
)
NUM_PROB_THRESHOLDS_HELP_STRING = (
    'Number of probability thresholds.  One contingency table will be created '
    'for each.  If you want to use specific thresholds, leave this argument '
    'alone and specify `{0:s}`.'
).format(PROB_THRESHOLDS_ARG_NAME)

PROB_THRESHOLDS_HELP_STRING = (
    'List of exact probability thresholds.  One contingency table will be '
    'created for each.  If you do not want to use specific thresholds, leave '
    'this argument alone and specify `{0:s}`.'
).format(NUM_PROB_THRESHOLDS_ARG_NAME)

CLIMO_FILE_HELP_STRING = (
    '[used only if `{0:s} = 1`] Path to file with climatology (event '
    'frequencies in training data).'
).format(GRIDDED_ARG_NAME)

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
    '--' + USE_PARTIAL_GRIDS_ARG_NAME, type=int, required=False, default=0,
    help=USE_PARTIAL_GRIDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + GRIDDED_ARG_NAME, type=int, required=False, default=0,
    help=GRIDDED_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MATCHING_DISTANCE_ARG_NAME, type=float, required=True,
    help=MATCHING_DISTANCE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PROB_THRESHOLDS_ARG_NAME, type=int, required=False,
    default=-1, help=NUM_PROB_THRESHOLDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PROB_THRESHOLDS_ARG_NAME, type=float, nargs='+', required=False,
    default=[-1], help=PROB_THRESHOLDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CLIMO_FILE_ARG_NAME, type=str, required=False, default='',
    help=CLIMO_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _compute_scores_partial_grids(
        top_prediction_dir_name, first_date_string, last_date_string,
        matching_distance_px, prob_thresholds, top_output_dir_name):
    """Computes scores on partial grids.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param matching_distance_px: Same.
    :param prob_thresholds: Same.
    :param top_output_dir_name: Same.
    """

    date_strings = []

    for k in range(NUM_RADARS):
        if k == 0:
            prediction_file_names = prediction_io.find_many_files(
                top_directory_name=top_prediction_dir_name,
                first_date_string=first_date_string,
                last_date_string=last_date_string,
                radar_number=k, prefer_zipped=True, allow_other_format=True,
                raise_error_if_any_missing=False
            )

            date_strings = [
                prediction_io.file_name_to_date(f)
                for f in prediction_file_names
            ]
        else:
            prediction_file_names = [
                prediction_io.find_file(
                    top_directory_name=top_prediction_dir_name,
                    valid_date_string=d, radar_number=k,
                    prefer_zipped=True, allow_other_format=True,
                    raise_error_if_missing=True
                ) for d in date_strings
            ]

        num_dates = len(date_strings)

        for i in range(num_dates):
            this_score_table_xarray = evaluation.get_basic_scores_ungridded(
                prediction_file_name=prediction_file_names[i],
                matching_distance_px=matching_distance_px,
                probability_thresholds=prob_thresholds
            )

            this_output_file_name = evaluation.find_basic_score_file(
                top_directory_name=top_output_dir_name,
                valid_date_string=date_strings[i],
                gridded=False, radar_number=k, raise_error_if_missing=False
            )

            print('\nWriting results to file: "{0:s}"...'.format(
                this_output_file_name
            ))
            evaluation.write_basic_score_file(
                basic_score_table_xarray=this_score_table_xarray,
                netcdf_file_name=this_output_file_name
            )

            if i == num_dates - 1:
                continue

            print(SEPARATOR_STRING)


def _compute_scores_full_grid(
        top_prediction_dir_name, first_date_string, last_date_string, gridded,
        matching_distance_px, prob_thresholds, climo_file_name,
        top_output_dir_name):
    """Computes scores on full grid.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param gridded: Same.
    :param matching_distance_px: Same.
    :param prob_thresholds: Same.
    :param climo_file_name: Same.
    :param top_output_dir_name: Same.
    """

    prediction_file_names = prediction_io.find_many_files(
        top_directory_name=top_prediction_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        radar_number=None, prefer_zipped=True, allow_other_format=True,
        raise_error_if_any_missing=False
    )

    date_strings = [
        prediction_io.file_name_to_date(f) for f in prediction_file_names
    ]
    num_dates = len(date_strings)

    if gridded:
        print('Reading event frequencies from: "{0:s}"...'.format(
            climo_file_name
        ))
        climo_dict = climatology_io.read_file(climo_file_name)
        training_event_freq_matrix = (
            climo_dict[climatology_io.EVENT_FREQ_BY_PIXEL_KEY]
        )
    else:
        training_event_freq_matrix = None

    for i in range(num_dates):
        if gridded:
            this_score_table_xarray = evaluation.get_basic_scores_gridded(
                prediction_file_name=prediction_file_names[i],
                matching_distance_px=matching_distance_px,
                probability_thresholds=prob_thresholds,
                training_event_freq_matrix=training_event_freq_matrix
            )

            model_file_name = (
                this_score_table_xarray.attrs[evaluation.MODEL_FILE_KEY]
            )
            model_metafile_name = neural_net.find_metafile(
                model_file_name=model_file_name, raise_error_if_missing=True
            )
            model_metadata_dict = neural_net.read_metafile(model_metafile_name)
            eval_mask_matrix = (
                model_metadata_dict[neural_net.FULL_MASK_MATRIX_KEY]
            )

            unmasked_row_indices, unmasked_column_indices = (
                numpy.where(eval_mask_matrix)
            )

            this_score_table_xarray = evaluation.subset_basic_scores_by_space(
                basic_score_table_xarray=this_score_table_xarray,
                first_grid_row=numpy.min(unmasked_row_indices),
                last_grid_row=numpy.max(unmasked_row_indices),
                first_grid_column=numpy.min(unmasked_column_indices),
                last_grid_column=numpy.max(unmasked_column_indices)
            )
        else:
            this_score_table_xarray = evaluation.get_basic_scores_ungridded(
                prediction_file_name=prediction_file_names[i],
                matching_distance_px=matching_distance_px,
                probability_thresholds=prob_thresholds
            )

        this_output_file_name = evaluation.find_basic_score_file(
            top_directory_name=top_output_dir_name,
            valid_date_string=date_strings[i],
            gridded=gridded, radar_number=None, raise_error_if_missing=False
        )

        print('\nWriting results to file: "{0:s}"...'.format(
            this_output_file_name
        ))
        evaluation.write_basic_score_file(
            basic_score_table_xarray=this_score_table_xarray,
            netcdf_file_name=this_output_file_name
        )

        if i == num_dates - 1:
            continue

        print(SEPARATOR_STRING)


def _run(top_prediction_dir_name, first_date_string, last_date_string,
         use_partial_grids, gridded, matching_distance_px, num_prob_thresholds,
         prob_thresholds, climo_file_name, top_output_dir_name):
    """Computes basic evaluation scores.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param use_partial_grids: Same.
    :param gridded: Same.
    :param matching_distance_px: Same.
    :param num_prob_thresholds: Same.
    :param prob_thresholds: Same.
    :param climo_file_name: Same.
    :param top_output_dir_name: Same.
    """

    if num_prob_thresholds > 0:
        prob_thresholds = gg_model_eval.get_binarization_thresholds(
            threshold_arg=num_prob_thresholds
        )

    if not use_partial_grids:
        _compute_scores_full_grid(
            top_prediction_dir_name=top_prediction_dir_name,
            first_date_string=first_date_string,
            last_date_string=last_date_string, gridded=gridded,
            matching_distance_px=matching_distance_px,
            prob_thresholds=prob_thresholds,
            climo_file_name=climo_file_name,
            top_output_dir_name=top_output_dir_name
        )

        return

    _compute_scores_partial_grids(
        top_prediction_dir_name=top_prediction_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        matching_distance_px=matching_distance_px,
        prob_thresholds=prob_thresholds,
        top_output_dir_name=top_output_dir_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        use_partial_grids=bool(
            getattr(INPUT_ARG_OBJECT, USE_PARTIAL_GRIDS_ARG_NAME)
        ),
        gridded=bool(getattr(INPUT_ARG_OBJECT, GRIDDED_ARG_NAME)),
        matching_distance_px=getattr(
            INPUT_ARG_OBJECT, MATCHING_DISTANCE_ARG_NAME
        ),
        num_prob_thresholds=getattr(
            INPUT_ARG_OBJECT, NUM_PROB_THRESHOLDS_ARG_NAME
        ),
        prob_thresholds=numpy.array(getattr(
            INPUT_ARG_OBJECT, PROB_THRESHOLDS_ARG_NAME
        )),
        climo_file_name=getattr(INPUT_ARG_OBJECT, CLIMO_FILE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
