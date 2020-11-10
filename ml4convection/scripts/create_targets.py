"""Creates target values from echo-classification data."""

import argparse
from ml4convection.io import example_io

INPUT_DIR_ARG_NAME = 'input_echo_classifn_dir_name'
MASK_FILE_ARG_NAME = 'input_mask_file_name'
USE_PARTIAL_GRIDS_ARG_NAME = 'use_partial_grids'
HALF_GRID_SIZE_ARG_NAME = 'half_grid_size_px'
SPATIAL_DS_FACTOR_ARG_NAME = 'spatial_downsampling_factor'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
OUTPUT_DIR_ARG_NAME = 'output_target_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with echo-classification data.  Files therein '
    'will be found by `radar_io.find_file` and read by '
    '`radar_io.read_echo_classifn_file`.'
)
MASK_FILE_HELP_STRING = (
    'Path to file with mask (used to censor locations with bad radar coverage).'
    '  Will be read by `radar_io.read_mask_file`.'
)
USE_PARTIAL_GRIDS_HELP_STRING = (
    'Boolean flag.  If 1 (0), will create targets on partial (full) grids.'
)
HALF_GRID_SIZE_HELP_STRING = (
    '[used only if {0:s} == 1] Size of half-grid (pixels).  If this number is '
    'K, the grid will have 2 * K + 1 rows and 2 * K + 1 columns.'
).format(USE_PARTIAL_GRIDS_ARG_NAME)

SPATIAL_DS_FACTOR_HELP_STRING = (
    '[used only if {0:s} == 0] Downsampling factor, used to coarsen spatial '
    'resolution.  If you do not want to coarsen spatial resolution, make this '
    '1.'
).format(USE_PARTIAL_GRIDS_ARG_NAME)

DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will process data for all days in the period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Target files will be written by '
    '`example_io._write_target_file`, to exact locations therein determined '
    'by `example_io.find_target_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MASK_FILE_ARG_NAME, type=str, required=True,
    help=MASK_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_PARTIAL_GRIDS_ARG_NAME, type=int, required=True,
    help=USE_PARTIAL_GRIDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + HALF_GRID_SIZE_ARG_NAME, type=int, required=False, default=-1,
    help=HALF_GRID_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SPATIAL_DS_FACTOR_ARG_NAME, type=int, required=False, default=1,
    help=SPATIAL_DS_FACTOR_HELP_STRING
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


def _run(top_echo_classifn_dir_name, mask_file_name, use_partial_grids,
         half_grid_size_px, spatial_downsampling_factor, first_date_string,
         last_date_string, top_output_dir_name):
    """Creates target values from echo-classification data.

    This is effectively the main method.

    :param top_echo_classifn_dir_name: See documentation at top of file.
    :param mask_file_name: Same.
    :param use_partial_grids: Same.
    :param half_grid_size_px: Same.
    :param spatial_downsampling_factor: Same.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param top_output_dir_name: Same.
    """

    if use_partial_grids:
        example_io.create_targets_partial_grids(
            top_echo_classifn_dir_name=top_echo_classifn_dir_name,
            mask_file_name=mask_file_name,
            half_grid_size_px=half_grid_size_px,
            first_date_string=first_date_string,
            last_date_string=last_date_string,
            top_output_dir_name=top_output_dir_name
        )

        return

    example_io.create_targets(
        top_echo_classifn_dir_name=top_echo_classifn_dir_name,
        mask_file_name=mask_file_name,
        spatial_downsampling_factor=spatial_downsampling_factor,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        top_output_dir_name=top_output_dir_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_echo_classifn_dir_name=getattr(
            INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME
        ),
        mask_file_name=getattr(INPUT_ARG_OBJECT, MASK_FILE_ARG_NAME),
        use_partial_grids=bool(getattr(
            INPUT_ARG_OBJECT, USE_PARTIAL_GRIDS_ARG_NAME
        )),
        half_grid_size_px=getattr(INPUT_ARG_OBJECT, HALF_GRID_SIZE_ARG_NAME),
        spatial_downsampling_factor=getattr(
            INPUT_ARG_OBJECT, SPATIAL_DS_FACTOR_ARG_NAME
        ),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
