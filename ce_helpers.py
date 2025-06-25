from datetime import datetime
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd


def setup_logger(base_dir: str | Path):
    """Initializes a file-based logger with a timestamped log file."""
    log_directory = Path(base_dir)
    log_directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"ce_{timestamp}.log"
    log_filepath = log_directory / log_filename

    logger = logging.getLogger("ce_pipeline")
    logger.setLevel(logging.INFO)  # Set base level to INFO

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


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
        logger.warning(
            f"Conversion Error: Could not convert '{val}' to float for {context_msg}."
        )
        return np.nan


def recreate_base_dir(base_dir: str):
    """Deletes and recreates the base directory."""
    base_dir_path = Path(base_dir)
    if base_dir_path.exists():
        shutil.rmtree(base_dir_path)
    base_dir_path.mkdir(parents=True, exist_ok=True)


def check_required_columns(
    df: pd.DataFrame,
    required_context_cols: Tuple[str, ...] = (
        "l3_name",
        "attribute_name",
        "attribute_value",
    ),
    logger: logging.Logger = logging.getLogger(__name__),
) -> pd.DataFrame:
    """
    Ensures the DataFrame has all required columns, initializing missing ones.

    This function enforces a standard column structure. If the input DataFrame
    is empty, it logs this information and returns the empty frame with the
    correct columns, which is an expected condition between pipeline stages.
    """
    columns_to_initialize = [
        "data_type",
        "key_value",
        "polarity1",
        "value1",
        "unit1",
        "polarity2",
        "value2",
        "unit2",
        "display_value",
        "mod_reason",
    ]
    final_order = list(required_context_cols) + columns_to_initialize

    if df.empty:
        logger.info(
            "Input DataFrame is empty, returning an empty DataFrame with standard columns."
        )
        return pd.DataFrame(columns=final_order)

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
    df = check_required_columns(df=df, logger=logger)
    return df


def get_compiled_regex(pattern_name: str) -> re.Pattern:
    """Returns a compiled regex object for a given pattern name."""
    patterns = {
        "numeric_with_optional_unit": r"^([+-]?)\s*(?:(?:(\d+)\s+(\d+)/(\d+))|(?:()(\d+)/(\d+))|(?:(\d*(?:,\d+)*\.?\d+)))()()(?:\s*([A-Za-z\"\°\'\.]+(?:\s+[A-Za-z\"\'\.]+)?))?$",
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
    func_name: str, passed_df: pd.DataFrame, mod_df: pd.DataFrame, base_dir: str = "."
) -> None:
    """Saves passed and modified DataFrames to their respective directories."""
    os.makedirs(os.path.join(base_dir, "passed"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "mod"), exist_ok=True)

    if passed_df is not None and not passed_df.empty:
        passed_path = os.path.join(base_dir, "passed", f"{func_name}_passed.csv")
        passed_df.to_csv(passed_path, index=False)

    if mod_df is not None and not mod_df.empty:
        mod_path = os.path.join(base_dir, "mod", f"{func_name}_mod.csv")
        mod_df.to_csv(mod_path, index=False)
