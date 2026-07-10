"""
Utility for converting between concentration/flux units that are all
anchored to a common base unit: ppm (parts per million).

Supported units
---------------
- 'ppm'        : parts per million (the reference/base unit)
- 'ppm m'      : ppm integrated over a path length in meters
- 'ppb'        : parts per billion
- 'umol m-2'   : micromoles per square meter

How it works
------------
Every unit is defined by a single number: the factor you multiply a
ppm value by to express it in that unit (SCALE_FROM_PPM below). To
convert between *any* two supported units, we simply:

    1. Convert the input value back to ppm (divide by its own factor).
    2. Convert that ppm value into the target unit (multiply by the
       target's factor).

This means adding a new unit only requires adding one new entry to
the SCALE_FROM_PPM dictionary — the conversion logic itself does not
need to change.

Usage
-----
    from unit_conversion import convert_units, get_scale_factor

    # Convert a value directly
    value_ppb = convert_units(2.5, input_unit='ppm', output_unit='ppb')

    # Or get the multiplicative factor and apply it yourself
    factor = get_scale_factor(input_unit='ppm', output_unit='ppb')
    value_ppb = 2.5 * factor
"""

# ---------------------------------------------------------------------------
# Base conversion table: factor to go FROM ppm TO the given unit.
#   units_value = ppm_value * SCALE_FROM_PPM[unit]
#
# NOTE on 'umol m-2': this factor (1000/2900*1e6) encodes a specific
# assumption about the air column / boundary-layer depth used to turn
# a ppm concentration into an areal molar amount. If that assumption
# ever changes (e.g. a different column height), update this value
# and add a comment explaining the new basis.
# ---------------------------------------------------------------------------
SCALE_FROM_PPM = {
    'ppm': 1,                      # reference unit, no scaling needed
    'ppm m': 1 / 1.25e-4,           # ppm integrated over a path length (m)
    'ppb': 1e3,                     # ppm -> ppb is just x1000
    'umol m-2': 1000 / 2900 * 1e6,  # ppm -> umol m-2 (column-depth assumption, see note above)
}


def get_scale_factor(input_unit, output_unit):
    """
    Compute the multiplicative factor to convert a value from
    `input_unit` to `output_unit`.

    Both units must be keys in SCALE_FROM_PPM. Conversion always
    routes through ppm as the common base:

        factor = SCALE_FROM_PPM[output_unit] / SCALE_FROM_PPM[input_unit]

    Parameters
    ----------
    input_unit : str
        Unit of the value you currently have (e.g. 'ppm', 'ppb').
    output_unit : str
        Unit you want to convert to.

    Returns
    -------
    float
        Multiply your input value by this to get the output value.

    Raises
    ------
    ValueError
        If either unit is not in SCALE_FROM_PPM.
    """
    if input_unit not in SCALE_FROM_PPM:
        raise ValueError(
            f"Unsupported input_unit '{input_unit}'. "
            f"Supported units: {list(SCALE_FROM_PPM.keys())}"
        )
    if output_unit not in SCALE_FROM_PPM:
        raise ValueError(
            f"Unsupported output_unit '{output_unit}'. "
            f"Supported units: {list(SCALE_FROM_PPM.keys())}"
        )

    # Route through ppm: undo the input unit's scaling, then apply
    # the output unit's scaling.
    return SCALE_FROM_PPM[output_unit] / SCALE_FROM_PPM[input_unit]


def convert_units(value, input_unit, output_unit):
    """
    Convert `value` from `input_unit` to `output_unit`.

    Parameters
    ----------
    value : float or array-like
        The value(s) to convert. Works with plain numbers as well as
        numpy arrays / pandas Series, since it's just a multiplication.
    input_unit : str
        Unit that `value` is currently expressed in.
    output_unit : str
        Unit to convert `value` into.

    Returns
    -------
    float or array-like
        `value` converted into `output_unit`.

    Examples
    --------
    >>> convert_units(2.5, 'ppm', 'ppb')
    2500.0
    >>> convert_units(2500, 'ppb', 'ppm')
    2.5
    """
    factor = get_scale_factor(input_unit, output_unit)
    return value * factor


# ---------------------------------------------------------------------------
# Backwards-compatible helper, matching the original `_scale_units`
# behavior: returns the factor to go FROM ppm TO the given unit.
# Kept so existing code that imported/used the old function name can
# be migrated with minimal changes.
# ---------------------------------------------------------------------------
def scale_from_ppm(units):
    """
    Return the factor to convert a value FROM ppm TO `units`.

    Equivalent to the original `_scale_units` method, but as a
    standalone function. Prefer `convert_units`/`get_scale_factor`
    for new code, since they support arbitrary input/output unit
    pairs rather than assuming ppm as the input.
    """
    return get_scale_factor(input_unit='ppm', output_unit=units)


if __name__ == '__main__':
    # Quick sanity checks / usage demo
    print(convert_units(1, 'ppm', 'ppb'))       # -> 1000.0
    print(convert_units(1000, 'ppb', 'ppm'))    # -> 1.0
    print(convert_units(1, 'ppm', 'ppm m'))     # -> 8000.0
    print(convert_units(1, 'ppm', 'umol m-2'))  # -> ~344827.6
