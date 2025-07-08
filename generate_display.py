from __future__ import annotations

import logging
from fractions import Fraction
from typing import List, Tuple

import numpy as np
import pandas as pd

from ce_utils import format_numeric_value
import ce_constants as cnst


def _decimal_to_simple_fraction(
    val: float | int | str, *, max_denominator: int = 64
) -> str:
    """Convert *val* (float/int/str) to a *reduced* numerator/denominator string.

    Examples
    --------
    >>> _decimal_to_simple_fraction(1.25)
    '5/4'
    >>> _decimal_to_simple_fraction('0.3333', max_denominator=8)
    '1/3'
    """
    if pd.isna(val):  # includes None / np.nan
        return ""

    try:
        num = float(val)
    except (ValueError, TypeError):
        return str(val)

    # Check if the number is an integer
    if num.is_integer():
        return str(int(num))

    frac = Fraction(num).limit_denominator(max_denominator)
    return f"{frac.numerator}/{frac.denominator}"


def _apply_spacing(left: str, right: str, flag: str | float | None) -> str:
    """Return ``left + space? + right`` where the space is added **only** when
    *flag* (from the reference unit table) is the literal string ``"space"``.
    """
    if not left and not right:
        return ""

    space = " " if str(flag).strip().lower() == "space" else ""
    return f"{left}{space}{right}"


def _compose_numeric(value: float | int | str | np.nan, *, style: str) -> str:
    """Format a numeric *value* into either decimal or simple‑fraction string.

    Parameters
    ----------
    style : {'decimal', 'fraction'}
        • ``'decimal'``   – use :pyfunc:`format_numeric_value` (round & strip).
        • ``'fraction'``  – convert to *numerator/denominator* with
                             :pyfunc:`_decimal_to_simple_fraction`.
    """
    if pd.isna(value):
        return ""

    style = str(style).lower()
    if style == "fraction":
        return _decimal_to_simple_fraction(value)
    return str(format_numeric_value(value))


def _compose_value_with_unit(
    value: float | int | str | np.nan,
    polarity: str | None,
    unit_disp: str | None,
    space_flag: str | None,
    num_style: str,
) -> str:
    """Build the *value + unit* segment respecting polarity, spacing & numeric style."""

    polarity = str(polarity).strip()
    if polarity == "-":
        pol = "-"
    elif polarity == "+":
        pol = "+"
    numeric_part = _compose_numeric(value, style=num_style)

    if not numeric_part and not unit_disp:
        return ""  # nothing to show

    numeric_with_pol = f"{pol}{numeric_part}" if numeric_part else pol
    return _apply_spacing(numeric_with_pol, str(unit_disp or ""), space_flag)


def build_display_values(
    df: pd.DataFrame,
    unit_df: pd.DataFrame,
    logger: logging.Logger = logging.getLogger(__name__),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Populate/overwrite the ``display_value`` column for *df*.

    The algorithm augments *df* with metadata from *unit_df* (matched on
    ``unit1`` ↔ ``unit_df.display``) and applies the following rules **per row**:

    1. **Spacing** – governed by ``unit_df.space``.  Only the literal value
       ``"space"`` inserts a space; everything else results in *no* space.
    2. **Numeric style** – from ``unit_df.type``: «decimal» (keep as is) or
       «fraction» (convert to *num/den*).
    3. **Range joiner** – for ``data_type == 'range1'`` the middle joiner comes
       from ``unit_df.range``; fallback is ``"to"``.

    Any row whose ``unit1`` **does not exist** in *unit_df.display* (case‑
    sensitive) is shunted to **mod_df** with ``mod_reason='unit1 not found'``.

    Notes
    -----
    • Only a *single* pass merge is performed, so *unit1* and *unit2* are
      assumed to have already been synchronised by the upstream cleaners.
    """

    if df.empty:
        logger.info("build_display_values: received empty DataFrame → nothing to do.")
        return df.copy()

    unit_meta = unit_df.drop_duplicates()

    # ── 2. Merge once on unit1 ⇢ display ───────────────────────────────────
    work = df.copy()
    work = work.merge(
        unit_meta,
        how="left",
        left_on="unit1",
        right_on="display",
        suffixes=("", "_unit"),
    )

    # ── 3. Split into (missing‑unit → mod) and the rest ────────────────────
    # missing_unit_mask = work["display"].isna()

    # mod_df = work[missing_unit_mask].copy()
    # if not mod_df.empty:
    #     mod_df["mod_reason"] = mod_df.get("mod_reason", pd.NA)
    #     mod_df["mod_reason"] = mod_df["mod_reason"].fillna("unit1 not found in unit_df")

    # pass_df = work[~missing_unit_mask].copy()

    # ── 4. Build display strings row‑by‑row ────────────────────────────────
    def _build_row(row: pd.Series) -> str:
        dt = str(row.get("data_type", "")).lower()
        num_style = str(row.get("type", "decimal")).lower()
        space_flag = row.get("space", "space")  # default is to *insert* space

        if dt == "numerical_with_unit":
            return _compose_value_with_unit(
                row.get("value1"),
                row.get("polarity1"),
                row.get("unit1"),
                space_flag,
                num_style,
            )

        if dt == "varchar":
            return str(row.get("value1", ""))

        if dt == "categorical":
            return str(row.get("value1", "")).title()

        if dt == "categorical2":
            return ", ".join(
                seg.strip().title() for seg in str(row.get("value1", "")).split(",")
            )

        if dt == "range1":
            joiner = str(row.get("range", "to")) or "to"
            # Special case: dash joiner should omit unit1 on left side
            if joiner == "-":
                # left: numeric + polarity
                left_num = _compose_numeric(row.get("value1"), style=num_style)
                left_pol = "-" if str(row.get("polarity1")).strip() == "-" else ""
                left = f"{left_pol}{left_num}" if left_num else left_pol
                # right: full value + unit2
                right = _compose_value_with_unit(
                    row.get("value2"),
                    row.get("polarity2"),
                    row.get("unit2"),
                    space_flag,
                    num_style,
                )
                return f"{left}{joiner}{right}"

            # default behavior
            left = _compose_value_with_unit(
                row.get("value1"),
                row.get("polarity1"),
                row.get("unit1"),
                space_flag,
                num_style,
            )
            right = _compose_value_with_unit(
                row.get("value2"),
                row.get("polarity2"),
                row.get("unit2"),
                space_flag,
                num_style,
            )
            # Respect spacing around joiner as stored in unit_df.range
            if " " in joiner:
                return f"{left}{joiner}{right}"
            return f"{left} {joiner} {right}"

        return str(row.get("attribute_value", ""))

    if not work.empty:
        work["display_value"] = work.apply(_build_row, axis=1)

    # Remove the merged helper columns so the output column set == input
    drop_cols = cnst.REQUIRED_COLUMNS.get("unit_df")
    logger.debug(f"drop columns: {drop_cols}")
    work.drop(columns=drop_cols, inplace=True, errors="ignore")
    work.reset_index(drop=True, inplace=True)

    return work
