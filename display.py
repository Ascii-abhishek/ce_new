from __future__ import annotations

import logging
from fractions import Fraction
from typing import Tuple

import numpy as np
import pandas as pd

from .ce_utils import format_numeric_value
from .constants import MAX_DENOMINATOR, DEFAULT_SPACE_FLAG, DEFAULT_NUM_STYLE


def _decimal_to_simple_fraction(
    val: float | int | str, *, max_denominator: int = MAX_DENOMINATOR
) -> str:
    if pd.isna(val):
        return ""
    try:
        num = float(val)
    except (ValueError, TypeError):
        return str(val)
    if num.is_integer():
        return str(int(num))
    frac = Fraction(num).limit_denominator(max_denominator)
    return f"{frac.numerator}/{frac.denominator}"


def _apply_spacing(left: str, right: str, flag: str | float | None) -> str:
    is_left_empty = pd.isna(left) or left == ""
    is_right_empty = pd.isna(right) or right == ""
    if is_left_empty and is_right_empty:
        return ""
    left_str = "" if is_left_empty else left
    right_str = "" if is_right_empty else right
    space = " " if str(flag).strip().lower() == "space" else ""
    return f"{left_str}{space}{right_str}"


def _compose_numeric(
    value: float | int | str | np.generic, *, style: str = DEFAULT_NUM_STYLE
) -> str:
    if pd.isna(value):
        return ""
    style = style.lower()
    if style == "fraction":
        return _decimal_to_simple_fraction(value)
    return str(format_numeric_value(value))


def _compose_value_with_unit(
    value: float | int | str | np.generic,
    polarity: str | None,
    unit: str | None,
    space_flag: str | None,
    num_style: str,
) -> str:
    polarity = str(polarity).strip() if polarity is not None else ""
    pol = "-" if polarity == "-" else "+" if polarity == "+" else ""
    numeric_part = _compose_numeric(value, style=num_style)
    unit_str = "" if unit is None or pd.isna(unit) else str(unit)
    space_str = "" if space_flag is None or pd.isna(space_flag) else str(space_flag)

    if not numeric_part and not unit_str:
        return ""

    numeric_with_pol = f"{pol}{numeric_part}" if numeric_part else pol

    return _apply_spacing(numeric_with_pol, unit_str, space_str)


def build_display_values(
    df: pd.DataFrame,
    unit_df: pd.DataFrame,
    logger: logging.Logger = logging.getLogger(__name__),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute `display_value` for each row in `df` using `unit_df` metadata,
    then split out rows where display_value could not be generated.

    Returns:
        pass_df: rows where `display_value` was successfully built.
        mod_df: rows without a `display_value`, annotated with `mod_reason`.
    """
    if df.empty:
        logger.info("build_display_values: empty input DataFrame.")
        empty_mod = pd.DataFrame(columns=df.columns.tolist() + ["mod_reason"])
        return df.copy(), empty_mod

    # Merge unit metadata once on unit1
    unit_meta = unit_df.drop_duplicates(subset=["display"])
    work = df.copy()
    work = work.merge(
        unit_meta.add_suffix("_unit"),
        how="left",
        left_on="unit1",
        right_on="display_unit",
    )

    # Build display_value for all rows
    def _build_row(row: pd.Series) -> str:
        dt = str(row.get("data_type", "")).lower()
        num_style = str(row.get("type_unit", DEFAULT_NUM_STYLE)).lower()
        space_flag = row.get("space_unit", DEFAULT_SPACE_FLAG)

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
            rng = row.get("range")
            joiner = str(rng) if pd.notna(rng) else "to"
            # Dash joiner: omit left unit
            if joiner == "-":
                left_num = _compose_numeric(row.get("value1"), style=num_style)
                left_pol = "-" if str(row.get("polarity1")).strip() == "-" else ""
                left = f"{left_pol}{left_num}" if left_num or left_pol else ""
                right = _compose_value_with_unit(
                    row.get("value2"),
                    row.get("polarity2"),
                    row.get("unit2"),
                    space_flag,
                    num_style,
                )
                return f"{left}{joiner}{right}"
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
            if " " in joiner:
                return f"{left}{joiner}{right}"
            return f"{left} {joiner} {right}"
        return str(row.get("attribute_value", ""))

    work["display_value"] = work.apply(_build_row, axis=1)

    mod_df = work[work["display_value"].eq("")].copy()
    if not mod_df.empty:
        mod_df["mod_reason"] = "could not build display_value"

    pass_df = work[work["display_value"].ne("")].copy()

    drop_meta = [col for col in pass_df.columns if col.endswith("_unit")]
    pass_df.drop(columns=drop_meta, inplace=True)
    mod_df.drop(columns=drop_meta, inplace=True)

    return pass_df.reset_index(drop=True), mod_df.reset_index(drop=True)
