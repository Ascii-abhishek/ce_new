from datetime import datetime
import logging
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from .ce_utils import check_required_columns, format_numeric_value


def build_sep_pattern(separators: List[str]) -> str:
    """Builds a regex pattern for a list of separators."""
    escaped = sorted([re.escape(s.strip()) for s in separators], key=len, reverse=True)
    return rf"\s*(?:{'|'.join(escaped)})\s*"


def recreate_base_dir(base_dir: str):
    """Deletes and recreates the base directory."""
    base_dir_path = Path(base_dir)
    if base_dir_path.exists():
        shutil.rmtree(base_dir_path)
    base_dir_path.mkdir(parents=True, exist_ok=True)


def lowercase_columns_and_values(
    df: pd.DataFrame,
    value_columns: List[str] | None = None,
    logger=logging.getLogger(__name__),
) -> pd.DataFrame:
    """Converts column names to snake_case and lowercases specified string columns."""
    if value_columns is None:
        value_columns = ["l3_name", "attribute_name"]

    df_processed = df.copy()

    def to_snake_case(column_name: str) -> str:
        column_name = re.sub(r"\s+", "_", column_name)
        return column_name.lower()

    df_processed.columns = [to_snake_case(col) for col in df_processed.columns]
    logger.info("Standardized column names to snake_case.")

    for col in value_columns:
        if col in df_processed.columns and pd.api.types.is_object_dtype(
            df_processed[col]
        ):
            df_processed.loc[:, col] = df_processed[col].str.lower().str.strip()

    logger.info(f"Lowercased values in columns: {value_columns}.")
    return df_processed


def process_rp(
    df: pd.DataFrame, remove: bool = False, logger=logging.getLogger(__name__)
) -> pd.DataFrame:
    """Adds or removes the 'rp_' prefix from specified value columns."""
    action = "Removing" if remove else "Prepending"
    logger.info(f"{action} 'rp_' prefix in value columns...")

    columns_to_process = ["attribute_value", "value1", "display_value", "value2"]
    present_columns = [c for c in columns_to_process if c in df.columns]

    if not present_columns:
        return df

    if not remove:
        for col in present_columns:
            df[col] = df[col].apply(lambda x: f"rp_{x}" if pd.notna(x) else x)
    else:
        pattern = r"(?i)^rp_"  # case-insensitive
        for col in present_columns:
            df[col] = df[col].astype(str).str.replace(pattern, "", regex=True)
    return df


def ce_start_cleanup(
    df: pd.DataFrame,
    logger: logging.Logger = logging.getLogger(__name__),
) -> pd.DataFrame:
    """Runs the initial cleanup sequence on the raw DataFrame."""
    df = lowercase_columns_and_values(df=df, logger=logger)
    df = process_rp(df=df, remove=True, logger=logger)
    df = check_required_columns(df=df, overwrite_init=True, logger=logger)
    return df


def get_compiled_regex(pattern_name: str) -> re.Pattern:
    num_opt_unit = r"""([+-]?)\s*(?:(?:(\d+)\s+(\d+)/(\d+))|
                       (?:()(\d+)/(\d+))|
                       (?:(\d*(?:,\d+)*\.?\d+)))()()(?:\s*
                       ([A-Za-z"°'\.]+(?:\s+[A-Za-z"'\.]+)?))?"""

    patterns = {
        "numeric_with_optional_unit": rf"^{num_opt_unit}$",
        "thread_metric": rf"""(?ix) ^\s*m {num_opt_unit} \s*
                              [x×] \s*
                              ([0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?)\s*
                              (?:"|mm)?\s*$""",
    }

    raw = patterns.get(pattern_name)
    if raw is None:
        raise KeyError(f"No regex pattern for “{pattern_name}”")
    return re.compile(raw, re.VERBOSE | re.IGNORECASE)


def standardize_unit(
    data_df: pd.DataFrame,
    unit_df: pd.DataFrame,
    unit_column: str = "unit1",
    logger: logging.Logger = logging.getLogger(__name__),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardizes units in a given column against a reference mapping.

    Returns a tuple of two DataFrames:
    - standardized_df: Rows with valid or correctable units.
    - mod_df: Rows with unrecognized units (to be moved to the 'mod' pile).
    """
    logger.info(f"Starting unit standardization on column '{unit_column}'.")

    if unit_column not in data_df.columns:
        raise KeyError(f"Input DataFrame must contain '{unit_column}' column.")

    mapping = dict(zip(unit_df["units"].astype(str), unit_df["display"].astype(str)))
    valid_values = set(unit_df["units"].astype(str)) | set(
        unit_df["display"].astype(str)
    )

    df_copy = data_df.copy()
    df_copy[unit_column] = df_copy[unit_column].fillna("").astype(str).str.strip()

    is_invalid_mask = ~df_copy[unit_column].isin(valid_values) & (
        df_copy[unit_column] != ""
    )
    mod_df = df_copy.loc[is_invalid_mask].copy()
    valid_df = df_copy.loc[~is_invalid_mask].copy()

    valid_df[unit_column] = valid_df[unit_column].replace(mapping)
    valid_df[unit_column] = valid_df[unit_column].replace({"": pd.NA})

    mod_df = mod_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    logger.info(
        f"Unit standardization complete: {len(valid_df)} valid, {len(mod_df)} invalid."
    )
    return valid_df, mod_df


def save_dfs(
    func_name: str,
    passed_df: pd.DataFrame,
    mod_df: pd.DataFrame,
    base_dir: str = ".",
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    """
    Saves DataFrames, applying final numeric formatting and prefixing in a
    single, atomic operation just before saving to prevent data type issues.
    """
    os.makedirs(os.path.join(base_dir, "passed"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "mod"), exist_ok=True)

    def _format_and_save(df: pd.DataFrame, path: str):
        if df is None or df.empty:
            return

        df_copy = df.copy()
        cols_to_prefix = ["attribute_value", "value1", "display_value", "value2"]

        for col in cols_to_prefix:
            if col in df_copy.columns:
                if col in ["value1", "value2"]:
                    df_copy[col] = df_copy[col].apply(
                        lambda x: (
                            f"rp_{format_numeric_value(x)}" if pd.notna(x) else pd.NA
                        )
                    )
                else:
                    df_copy[col] = df_copy[col].apply(
                        lambda x: f"rp_{x}" if pd.notna(x) else pd.NA
                    )

        if "passed" in path and "mod_reason" in df_copy.columns:
            df_copy.drop(columns=["mod_reason"], inplace=True)

        df_copy.to_csv(path, index=False)

    passed_path = os.path.join(base_dir, "passed", f"{func_name}_passed.csv")
    _format_and_save(passed_df, passed_path)

    mod_path = os.path.join(base_dir, "mod", f"{func_name}_mod.csv")
    _format_and_save(mod_df, mod_path)
