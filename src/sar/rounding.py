"""SCRUM-189: one rounding rule for every number the app displays, so the screen and the download agree.

Python's round() and format specifications round to the nearest value with ties to even, so 2.5 shows as 2
and a module median of 10.5 as 10. Displayed values go through these helpers instead: the value, as its
shortest decimal form (repr), is rounded once with halves away from zero (2.5 -> 3, -2.5 -> -3,
10.25 -> 10.3, 0.0065 -> 0.007).
"""

from decimal import ROUND_HALF_UP, Decimal


def _quantized(value, digits: int) -> Decimal:
    return Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)


def round_half_up(value, digits: int = 0):
    """`value` rounded half up to `digits` decimals: an int for 0 digits, otherwise a float."""
    q = _quantized(value, digits)
    return int(q) if digits == 0 else float(q)


def fixed(value, digits: int = 0, sign: bool = False, grouping: bool = False) -> str:
    """`value` rounded half up and written with exactly `digits` decimals; `sign` adds + to positive values and
    `grouping` adds thousands separators."""
    return format(_quantized(value, digits), f"{'+' if sign else ''}{',' if grouping else ''}.{digits}f")
