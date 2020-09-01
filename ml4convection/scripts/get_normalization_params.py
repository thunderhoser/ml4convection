"""Finds normalization parameters for satellite variables (predictors)."""

import argparse
from ml4convection.utils import normalization

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SATELLITE_DIR_ARG_NAME = 'input_satellite_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
NUM_VALUES_PER_BAND_ARG_NAME = 'num_values_per_band'
DO_TEMPERATURES_ARG_NAME = 'do_temperatures'
DO_COUNTS_ARG_NAME = 'do_counts'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

SATELLITE_DIR_HELP_STRING = (
    'Name of top-level directory with satellite data.  Files therein will be '
    'found by `satellite_io.find_file` and read by `satellite_io.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Normalization parameters will be based on data '
    'from the period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

NUM_VALUES_PER_BAND_HELP_STRING = (
    'Number of values to save from each spectral band.  These will be randomly '
    'sampled.'
)
DO_TEMPERATURES_HELP_STRING = (
    'Boolean flag.  If 1, will compute normalization parameters for brightness '
    'temperatures.'
)
DO_COUNTS_HELP_STRING = (
    'Boolean flag.  If 1, will compute normalization parameters for brightness '
    'counts.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Normalization parameters will be written here by '
    '`normalization.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SATELLITE_DIR_ARG_NAME, type=str, required=True,
    help=SATELLITE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_VALUES_PER_BAND_ARG_NAME, type=int, required=False, default=2e5,
    help=NUM_VALUES_PER_BAND_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DO_TEMPERATURES_ARG_NAME, type=int, required=False, default=1,
    help=DO_TEMPERATURES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DO_COUNTS_ARG_NAME, type=int, required=False, default=1,
    help=DO_COUNTS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(top_satellite_dir_name, first_date_string, last_date_string,
         num_values_per_band, do_temperatures, do_counts, output_file_name):
    """Finds normalization parameters for satellite variables (predictors).

    This is effectively the main method.

    :param top_satellite_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param num_values_per_band: Same.
    :param do_temperatures: Same.
    :param do_counts: Same.
    :param output_file_name: Same.
    """

    norm_dict_for_temperature, norm_dict_for_count = (
        normalization.get_normalization_params(
            top_satellite_dir_name=top_satellite_dir_name,
            first_date_string=first_date_string,
            last_date_string=last_date_string,
            num_values_per_band=num_values_per_band,
            do_temperatures=do_temperatures, do_counts=do_counts
        )
    )

    print(SEPARATOR_STRING)
    print('Writing normalization parameters to: "{0:s}"...'.format(
        output_file_name
    ))

    normalization.write_file(
        pickle_file_name=output_file_name,
        norm_dict_for_temperature=norm_dict_for_temperature,
        norm_dict_for_count=norm_dict_for_count
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_satellite_dir_name=getattr(
            INPUT_ARG_OBJECT, SATELLITE_DIR_ARG_NAME
        ),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        num_values_per_band=getattr(
            INPUT_ARG_OBJECT, NUM_VALUES_PER_BAND_ARG_NAME
        ),
        do_temperatures=bool(getattr(
            INPUT_ARG_OBJECT, DO_TEMPERATURES_ARG_NAME
        )),
        do_counts=bool(getattr(INPUT_ARG_OBJECT, DO_COUNTS_ARG_NAME)),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
