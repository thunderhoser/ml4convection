"""Skeleton version of radar_utils.py from GewitterGefahr."""

import os
import sys

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking

ECHO_TOP_15DBZ_NAME = 'echo_top_15dbz_km'
ECHO_TOP_18DBZ_NAME = 'echo_top_18dbz_km'
ECHO_TOP_20DBZ_NAME = 'echo_top_20dbz_km'
ECHO_TOP_25DBZ_NAME = 'echo_top_25dbz_km'
ECHO_TOP_40DBZ_NAME = 'echo_top_40dbz_km'
ECHO_TOP_50DBZ_NAME = 'echo_top_50dbz_km'
LOW_LEVEL_SHEAR_NAME = 'low_level_shear_s01'
MID_LEVEL_SHEAR_NAME = 'mid_level_shear_s01'
MESH_NAME = 'mesh_mm'
REFL_NAME = 'reflectivity_dbz'
REFL_COLUMN_MAX_NAME = 'reflectivity_column_max_dbz'
REFL_0CELSIUS_NAME = 'reflectivity_0celsius_dbz'
REFL_M10CELSIUS_NAME = 'reflectivity_m10celsius_dbz'
REFL_M20CELSIUS_NAME = 'reflectivity_m20celsius_dbz'
REFL_LOWEST_ALTITUDE_NAME = 'reflectivity_lowest_altitude_dbz'
SHI_NAME = 'shi'
VIL_NAME = 'vil_mm'
DIFFERENTIAL_REFL_NAME = 'differential_reflectivity_db'
SPEC_DIFF_PHASE_NAME = 'specific_differential_phase_deg_km01'
CORRELATION_COEFF_NAME = 'correlation_coefficient'
SPECTRUM_WIDTH_NAME = 'spectrum_width_m_s01'
VORTICITY_NAME = 'vorticity_s01'
DIVERGENCE_NAME = 'divergence_s01'
STORM_ID_NAME = 'storm_id_string'

RADAR_FIELD_NAMES = [
    ECHO_TOP_15DBZ_NAME, ECHO_TOP_18DBZ_NAME,
    ECHO_TOP_20DBZ_NAME, ECHO_TOP_25DBZ_NAME,
    ECHO_TOP_40DBZ_NAME, ECHO_TOP_50DBZ_NAME,
    LOW_LEVEL_SHEAR_NAME, MID_LEVEL_SHEAR_NAME,
    MESH_NAME, REFL_NAME,
    REFL_COLUMN_MAX_NAME, REFL_0CELSIUS_NAME,
    REFL_M10CELSIUS_NAME, REFL_M20CELSIUS_NAME,
    REFL_LOWEST_ALTITUDE_NAME, SHI_NAME, VIL_NAME,
    DIFFERENTIAL_REFL_NAME, SPEC_DIFF_PHASE_NAME,
    CORRELATION_COEFF_NAME, SPECTRUM_WIDTH_NAME,
    VORTICITY_NAME, DIVERGENCE_NAME,
    STORM_ID_NAME
]

SHEAR_NAMES = [LOW_LEVEL_SHEAR_NAME, MID_LEVEL_SHEAR_NAME]
ECHO_TOP_NAMES = [
    ECHO_TOP_15DBZ_NAME, ECHO_TOP_18DBZ_NAME, ECHO_TOP_20DBZ_NAME,
    ECHO_TOP_25DBZ_NAME, ECHO_TOP_40DBZ_NAME, ECHO_TOP_50DBZ_NAME
]
REFLECTIVITY_NAMES = [
    REFL_NAME, REFL_COLUMN_MAX_NAME, REFL_0CELSIUS_NAME, REFL_M10CELSIUS_NAME,
    REFL_M20CELSIUS_NAME, REFL_LOWEST_ALTITUDE_NAME
]


def check_field_name(field_name):
    """Ensures that name of radar field is recognized.

    :param field_name: Name of radar field in GewitterGefahr format.
    :raises: ValueError: if name of radar field is not recognized.
    """

    error_checking.assert_is_string(field_name)

    if field_name not in RADAR_FIELD_NAMES:
        error_string = (
            '\n{0:s}\nValid radar fields (listed above) do not include "{1:s}".'
        ).format(str(RADAR_FIELD_NAMES), field_name)

        raise ValueError(error_string)
