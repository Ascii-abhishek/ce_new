import logging
import os
import re
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def safe_str_to_float(val: Any, logger: logging.Logger, context_msg: str) -> float:
    """
    Safely converts a value (usually string) to float.
    Returns np.nan on error and logs a warning.
    """
    if pd.isna(val) or val == "":
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        logger.warning(f"Conversion Error: Could not convert '{val}' to float for {context_msg}.")
        return np.nan


def check_required_columns(
    df: pd.DataFrame,
    logger: logging.Logger,
    required_context_cols: Tuple[str, ...] = ("l3_name", "attribute_name", "attribute_value"),
) -> pd.DataFrame:
    """
    Ensures that the DataFrame has all required context columns, initializes missing columns,
    and orders them in a standard way.
    """
    columns_to_initialize = [
        "data_type",
        "polarity1",
        "value1",
        "unit1",
        "polarity2",
        "value2",
        "unit2",
        "key_value",
        "display_value",
        "mod_reason",
    ]
    final_order = list(required_context_cols) + columns_to_initialize

    if df.empty:
        return df

    missing_context = [col for col in required_context_cols if col not in df.columns]
    if missing_context:
        raise KeyError(f"Missing required context columns: {missing_context}")

    df_copy = df.copy()
    for col in columns_to_initialize:
        if col not in df_copy.columns:
            df_copy[col] = pd.NA

    try:
        df_final = df_copy[final_order]
    except KeyError as e:
        raise ValueError(f"Error ordering columns in check_required_columns: {e}")

    return df_final


def lowercase_columns_and_values(
    df: pd.DataFrame,
    value_columns: List[str] | None = None,
) -> pd.DataFrame:
    if value_columns is None:
        value_columns = ["l3_name", "attribute_name"]

    df_processed = df.copy()

    def to_snake_case(column_name: str) -> str:
        column_name = re.sub(r"\s+", "_", column_name)
        return column_name.lower()

    df_processed.columns = [to_snake_case(col) for col in df_processed.columns]

    for col in value_columns:
        if col in df_processed.columns and pd.api.types.is_object_dtype(df_processed[col]):
            df_processed.loc[:, col] = df_processed[col].str.lower().str.strip()

    return df_processed


def process_rp(df: pd.DataFrame, remove: bool = False) -> pd.DataFrame:
    columns_to_process = ["attribute_value", "value1", "display_value", "value2"]
    present_columns = [c for c in columns_to_process if c in df.columns]

    if not present_columns:
        return df

    if not remove:
        for col in present_columns:
            df[col] = df[col].apply(lambda x: f"rp_{x}" if pd.notna(x) else x)
    else:
        pattern = r"(?i)^rp_"  # case‑insensitive
        for col in present_columns:
            df[col] = df[col].astype(str).str.replace(pattern, "", regex=True)
    return df


def ce_start_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df = check_required_columns(df, LOGGER)
    df = lowercase_columns_and_values(df)
    df = process_rp(df, remove=True)
    return df


def get_compiled_regex(pattern_name: str) -> re.Pattern:
    """
    Returns a compiled regex object based on the requested pattern name.
    Add new patterns to the `patterns` dict as needed.
    """
    patterns = {
        "numeric_with_optional_unit": r"^([+-]?)\s*(?:(?:(\d+)\s+(\d+)/(\d+))|(?:()(\d+)/(\d+))|(?:(\d*(?:\d+,\d+)?\.?\d+(?!\.))()()))(?:\s*([A-Za-z\"']+(?:\s+[A-Za-z\"']+)?))?$",
    }

    raw_pattern = patterns.get(pattern_name)
    if raw_pattern is None:
        raise KeyError(f"No regex pattern found for '{pattern_name}'")

    return re.compile(raw_pattern, re.VERBOSE)


def standardize_unit(
    data_df: pd.DataFrame,
    unit_df: pd.DataFrame,
    unit_column: str = "unit1",
    logger: logging.Logger = logging.getLogger(__name__),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardizes unit values in `unit_column` of data_df based on mappings in unit_df.
    - unit_df must have columns: 'units' (raw unit values) and 'display' (standardized values).
    Returns a tuple: (standardized_df, mod_df).
      * standardized_df: rows where unit_column was valid and replaced with display value
      * mod_df: rows where unit_column contained an unrecognized unit
    """
    logger.info("Starting unit standardization.")

    if unit_column not in data_df.columns:
        raise KeyError(f"Input DataFrame must contain '{unit_column}' column.")

    # Build mapping dictionary: raw unit -> standardized display
    mapping = dict(zip(unit_df["units"].astype(str), unit_df["display"].astype(str)))
    # Valid values include both raw 'units' and 'display'
    valid_values = set(unit_df["units"].astype(str)) | set(unit_df["display"].astype(str))

    df_copy = data_df.copy()
    df_copy[unit_column] = df_copy[unit_column].fillna("").astype(str).str.strip()

    # Identify rows with invalid units
    invalid_mask = ~df_copy[unit_column].isin(valid_values) & (df_copy[unit_column] != "")
    mod_df = df_copy.loc[invalid_mask].copy()
    valid_df = df_copy.loc[~invalid_mask].copy()

    # Replace raw units with their display equivalents
    valid_df[unit_column] = valid_df[unit_column].replace(mapping)

    # For rows where unit_column is empty string, set to pd.NA
    valid_df[unit_column] = valid_df[unit_column].replace({"": pd.NA})

    # Reorder columns if needed (optional)
    mod_df = mod_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    logger.info(f"Unit standardization complete: {len(valid_df)} valid, {len(mod_df)} invalid.")
    return valid_df, mod_df


def cleanup_range(
    data_df: pd.DataFrame,
    unit_df: pd.DataFrame,
    logger: logging.Logger = logging.getLogger(__name__),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For rows with data_type == "range1", enforce:
      1) if (polarity1 == "+" and polarity2 == "-") → swap polarities & values.
      2) if polarity conditions (both null, or polarity1 null and polarity2 '+', or polarity1 == polarity2), ensure value1 <= value2 → swap values if needed.
      3) fill missing unit from the other side.
      4) call standardize_unit on unit1, then on unit2; collect any failures into mod.
      5) finally, any rows where unit1 and unit2 still mismatch → move to mod.
    Returns: (passed_df, mod_df) where mod_df has a "reason" column.
    """
    if not (data_df["data_type"] == "range1").all():
        raise ValueError("All rows must have data_type == 'range1'")

    df_work = data_df.copy().reset_index(drop=True)
    for col in ("polarity1", "polarity2", "unit1", "unit2"):
        df_work[col] = df_work[col].fillna("").astype(str).str.strip().replace({"": pd.NA})

    mod_rows = []
    mask_swap_polarity = (df_work["polarity1"] == "+") & (df_work["polarity2"] == "-")
    if mask_swap_polarity.any():
        rows = df_work.loc[mask_swap_polarity].index
        df_work.loc[rows, ["polarity1", "polarity2"]] = df_work.loc[rows, ["polarity2", "polarity1"]].values
        df_work.loc[rows, ["value1", "value2"]] = df_work.loc[rows, ["value2", "value1"]].values

    mask_order = (
        (df_work["polarity1"].isna() & df_work["polarity2"].isna())
        | (df_work["polarity1"].isna() & (df_work["polarity2"] == "+"))
        | (df_work["polarity1"] == df_work["polarity2"])
    )
    if mask_order.any():
        sub = df_work.loc[mask_order]
        vals1 = sub["value1"].apply(safe_str_to_float, logger, f"Custom ")
        vals2 = sub["value2"].apply(safe_str_to_float, logger, f"Custom ")
        swap_idx = sub.index[vals1 > vals2]
        if len(swap_idx):
            df_work.loc[swap_idx, ["value1", "value2"]] = df_work.loc[swap_idx, ["value2", "value1"]].values

    mask_unit2_missing = df_work["unit1"].notna() & df_work["unit2"].isna()
    if mask_unit2_missing.any():
        df_work.loc[mask_unit2_missing, "unit2"] = df_work.loc[mask_unit2_missing, "unit1"]
    mask_unit1_missing = df_work["unit2"].notna() & df_work["unit1"].isna()
    if mask_unit1_missing.any():
        df_work.loc[mask_unit1_missing, "unit1"] = df_work.loc[mask_unit1_missing, "unit2"]

    valid1, mod1 = standardize_unit(df_work, unit_df, unit_column="unit1", logger=logger)
    if not mod1.empty:
        temp = mod1.copy()
        temp["reason"] = "invalid unit1"
        mod_rows.append(temp)

    valid2, mod2 = standardize_unit(valid1, unit_df, unit_column="unit2", logger=logger)
    if not mod2.empty:
        temp = mod2.copy()
        temp["reason"] = "invalid unit2"
        mod_rows.append(temp)

    mask_mismatch = (
        valid2["unit1"].notna()
        & valid2["unit2"].notna()
        & (valid2["unit1"].str.lower() != valid2["unit2"].str.lower())
    )
    if mask_mismatch.any():
        temp = valid2.loc[mask_mismatch].copy()
        temp["reason"] = "unit1 and unit2 are different after standardization"
        mod_rows.append(temp)
        valid2 = valid2.loc[~mask_mismatch].reset_index(drop=True)

    passed_df = valid2.reset_index(drop=True)
    if mod_rows:
        mod_df = pd.concat(mod_rows, ignore_index=True)
    else:
        mod_df = pd.DataFrame(columns=list(data_df.columns) + ["reason"])

    return passed_df, mod_df


def save_dfs(func_name: str, passed_df: pd.DataFrame, mod_df: pd.DataFrame, base_dir: str = ".") -> None:
    os.makedirs(os.path.join(base_dir, "passed"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "mod"), exist_ok=True)

    if passed_df is not None and not passed_df.empty:
        passed_path = os.path.join(base_dir, "passed", f"{func_name}_passed.csv")
        passed_df.to_csv(passed_path, index=False)

    if mod_df is not None and not mod_df.empty:
        mod_path = os.path.join(base_dir, "mod", f"{func_name}_mod.csv")
        mod_df.to_csv(mod_path, index=False)
