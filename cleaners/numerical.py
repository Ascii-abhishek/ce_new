import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..ce_helpers import get_compiled_regex, standardize_unit
from ..ce_utils import (check_required_columns, format_numeric_value, log_step,
                     safe_str_to_float)


def _calculate_mixed_fraction_value(
    whole_str: str, num_str: str, den_str: str, logger: logging.Logger, context: str
) -> Tuple[float, str]:
    """Calculates the absolute value of a mixed fraction, returning value and error."""
    whole = safe_str_to_float(whole_str, logger, f"whole part in {context}")
    num = safe_str_to_float(num_str, logger, f"numerator in {context}")
    den = safe_str_to_float(den_str, logger, f"denominator in {context}")

    if any(np.isnan([whole, num, den])):
        return np.nan, "Invalid number in mixed fraction"
    if den == 0.0:
        return np.nan, "Division by zero in mixed fraction"
    return abs(whole) + abs(num) / abs(den), ""


def _calculate_simple_fraction_value(
    num_str: str, den_str: str, logger: logging.Logger, context: str
) -> Tuple[float, str]:
    """Calculates the absolute value of a simple fraction, returning value and error."""
    num = safe_str_to_float(num_str, logger, f"numerator in {context}")
    den = safe_str_to_float(den_str, logger, f"denominator in {context}")

    if any(np.isnan([num, den])):
        return np.nan, "Invalid number in simple fraction"
    if den == 0.0:
        return np.nan, "Division by zero in simple fraction"
    return abs(num) / abs(den), ""


def _convert_decimal_to_float(
    val_str: str, logger: logging.Logger, context: str
) -> Tuple[float, str]:
    """Converts a decimal string to an absolute float, returning value and error."""
    if pd.isna(val_str) or not isinstance(val_str, str) or not val_str.strip():
        return np.nan, "Empty or invalid decimal string"

    cleaned = val_str.replace(",", "").strip()
    if not cleaned or cleaned.count(".") > 1:
        return np.nan, "Invalid decimal format"

    value = safe_str_to_float(cleaned.lstrip("+-"), logger, context)
    if np.isnan(value):
        return np.nan, "Invalid characters in decimal string"
    return abs(value), ""


def _process_row_for_numerical_unit(
    row: pd.Series, regex_cols_map: Dict[str, str], logger: logging.Logger
) -> Tuple[float | int | np.floating, str]:
    """
    Convert the regex captures in *row* to a single numeric value
    (value1 / value2, etc.), then format it so that:
        • it is rounded to 4 dp,
        • trailing zeros are stripped,
        • whole numbers are returned as plain ints.

    Returns
    -------
    value : int | float | np.nan
        The formatted numeric value.
    err_msg : str
        Empty string on success; otherwise a short description of what went wrong.
    """
    context = f"attribute_value '{row.get('attribute_value', '')}'"

    # ── Mixed fraction ──────────────────────────────────────────────────
    if pd.notna(row.get(regex_cols_map["mixed_whole"])):
        raw_val, err = _calculate_mixed_fraction_value(
            row[regex_cols_map["mixed_whole"]],
            row[regex_cols_map["mixed_num"]],
            row[regex_cols_map["mixed_den"]],
            logger,
            context,
        )
        return format_numeric_value(raw_val), err

    # ── Simple fraction ─────────────────────────────────────────────────
    if pd.notna(row.get(regex_cols_map["simple_num"])):
        raw_val, err = _calculate_simple_fraction_value(
            row[regex_cols_map["simple_num"]],
            row[regex_cols_map["simple_den"]],
            logger,
            context,
        )
        return format_numeric_value(raw_val), err

    # ── Decimal / integer ──────────────────────────────────────────────
    if pd.notna(row.get(regex_cols_map["decimal"])):
        raw_val, err = _convert_decimal_to_float(
            row[regex_cols_map["decimal"]], logger, context
        )
        return format_numeric_value(raw_val), err

    # ── Nothing matched ────────────────────────────────────────────────
    return np.nan, "No recognizable numeric format"


@log_step
def clean_numerical_unit(
    df: pd.DataFrame,
    unit_df: pd.DataFrame,
    value_column: str = "attribute_value",
    logger: logging.Logger = logging.getLogger(__name__),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cleans numeric values (with or without units) from a DataFrame.
    Returns a tuple of DataFrames: (passed_df, modified_for_error_df, remaining_df).

    - passed_df: rows where numeric extraction succeeded without errors
    - modified_for_error_df: rows where numeric extraction matched but had conversion errors
    - remaining_df: rows where regex did not match at all
    """
    if value_column not in df.columns:
        raise KeyError(f"Input DataFrame must contain '{value_column}' column.")

    df_working = df.copy()
    pattern = get_compiled_regex("numeric_with_optional_unit")
    extracted = df_working[value_column].astype(str).str.extract(pattern)
    extracted.columns = [
        "sign",
        "mixed_whole",
        "mixed_num",
        "mixed_den",
        "_unused_simple_whole",
        "simple_num",
        "simple_den",
        "decimal",
        "_unused_dec_int_num",
        "_unused_dec_int_den",
        "unit",
    ]
    extracted.index = df_working.index

    combined = pd.concat([df_working, extracted], axis=1)

    # Rows where none of the numeric groups matched
    no_match_mask = (
        combined["mixed_whole"].isna()
        & combined["simple_num"].isna()
        & combined["decimal"].isna()
    )
    remaining_df = combined.loc[no_match_mask, [*df_working.columns]].copy()
    passed_intermediate = combined.loc[~no_match_mask].copy()

    logger.info(
        f"{len(passed_intermediate)} rows matched clean_numeric_unit regex; {len(remaining_df)} did not."
    )

    if not passed_intermediate.empty:
        # Determine polarity from sign
        passed_intermediate["polarity1"] = passed_intermediate["sign"].replace(
            {"": pd.NA, np.nan: pd.NA}
        )
        # Assign unit directly from regex group
        passed_intermediate["unit1"] = passed_intermediate["unit"].where(
            passed_intermediate["unit"].notna(), pd.NA
        )

        # Map for extraction
        regex_cols_map = {
            "mixed_whole": "mixed_whole",
            "mixed_num": "mixed_num",
            "mixed_den": "mixed_den",
            "simple_num": "simple_num",
            "simple_den": "simple_den",
            "decimal": "decimal",
        }

        # Compute numeric value and error reason
        values_reasons = passed_intermediate.apply(
            lambda row: _process_row_for_numerical_unit(row, regex_cols_map, logger),
            axis=1,
        )
        passed_intermediate[["value1", "mod_reason"]] = pd.DataFrame(
            values_reasons.tolist(), index=passed_intermediate.index
        )

        # Coerce value1 to numeric
        passed_intermediate["value1"] = pd.to_numeric(
            passed_intermediate["value1"], errors="coerce"
        )

        # Determine data_type based on presence of unit
        has_unit = passed_intermediate["unit1"].notna() & (
            passed_intermediate["unit1"] != ""
        )
        passed_intermediate["data_type"] = np.where(
            has_unit, "numerical_with_unit", "numerical_without_unit"
        )

        # Split into passed (no errors) and modified_for_error (errors present)
        error_mask = passed_intermediate["mod_reason"].notna() & (
            passed_intermediate["mod_reason"] != ""
        )
        initial_mod_df = passed_intermediate[error_mask].copy()
        initial_passed_df = passed_intermediate[~error_mask].copy()

        # Standardize units on the successfully parsed data
        if not initial_passed_df.empty:
            passed_after_units, unit_mod_df = standardize_unit(
                initial_passed_df, unit_df, "unit1", logger
            )
            if not unit_mod_df.empty:
                unit_mod_df["mod_reason"] = "Invalid unit"
                all_mods = [df for df in [initial_mod_df, unit_mod_df] if not df.empty]
                final_mod_df = (
                    pd.concat(all_mods, ignore_index=True)
                    if all_mods
                    else pd.DataFrame()
                )
            else:
                final_mod_df = initial_mod_df
            final_passed_df = passed_after_units
        else:  # If all initial rows had errors
            final_passed_df = initial_passed_df
            final_mod_df = initial_mod_df
    else:
        final_passed_df, final_mod_df = pd.DataFrame(), pd.DataFrame()

    # Reorder and drop helper columns
    passed_df = check_required_columns(df=final_passed_df, logger=logger)
    mod_df = check_required_columns(df=final_mod_df, logger=logger)
    remaining_df = check_required_columns(df=remaining_df, logger=logger)

    return passed_df, mod_df, remaining_df
