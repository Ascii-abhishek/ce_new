import json
import logging
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
import pandas as pd

from .constants import *


class ValidationError(Exception):
    """Raised when validation of inputs fails."""

    pass


StepResult = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


def log_step(fn: Callable[[pd.DataFrame], StepResult]) -> Callable[..., StepResult]:
    """
    Decorator to log entry, exit, row counts, and execution time of each cleaning step.
    Uses a passed-in logger if provided via the 'logger' keyword argument to the wrapped function, otherwise falls back to the module-level logger.
    """

    @wraps(fn)
    def wrapper(df: pd.DataFrame, *args, **kwargs) -> StepResult:
        # Extract logger from kwargs or use default
        log = kwargs.get("logger", logging.getLogger(__name__))
        name = fn.__name__
        in_count = len(df)
        start = time.time()
        # Call the actual cleaning function; remove logger from kwargs if present
        func_kwargs = {k: v for k, v in kwargs.items() if k != "logger"}
        passed, mod, remain = fn(df, *args, **func_kwargs)
        elapsed = time.time() - start
        log.info(
            json.dumps(
                {
                    "step": name,
                    "in": in_count,
                    "passed": len(passed),
                    "mod": len(mod),
                    "remain": len(remain),
                    "time_s": round(elapsed, 3),
                }
            )
        )
        return passed, mod, remain

    return wrapper


def ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValidationError(f"Missing columns: {missing}")


def check_required_columns(
    df: pd.DataFrame,
    df_name: str = "base_df",
    overwrite_init: bool = False,
    logger: logging.Logger = logging.getLogger(__name__),
) -> pd.DataFrame:
    """Ensure the dataframe has the required and initialized columns."""
    required_cols = REQUIRED_COLUMNS.get(df_name)
    if not required_cols:
        raise KeyError(ERROR_MESSAGES["invalid_df_name"].format(df_name=df_name))

    init_cols = INIT_COLUMNS.get(df_name, [])
    final_order = list(required_cols) + init_cols

    if df.empty:
        return pd.DataFrame(columns=final_order)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(
            ERROR_MESSAGES["missing_required_columns"].format(missing_cols=missing_cols)
        )

    df_copy = df.copy()
    if overwrite_init:
        df_copy.drop(columns=init_cols, inplace=True, errors="ignore")

    for col in init_cols:
        if col not in df_copy.columns:
            df_copy[col] = pd.NA

    return df_copy[final_order]


def format_numeric_value(val: Any, ndigits: int = 4) -> Any:
    """
    Round to *ndigits* (default = 4), strip insignificant zeros,
    and return an ``int`` when the rounded value is whole.
    """
    if pd.isna(val):
        return val

    try:
        num = float(val)
    except (ValueError, TypeError):
        return val

    num = round(num, ndigits)
    if num.is_integer():
        return int(num)
    else:
        return num


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
