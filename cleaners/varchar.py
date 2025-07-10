import logging
from typing import Tuple

import pandas as pd
import numpy as np

from ..ce_utils import log_step


@log_step
def clean_varchar_categorical(
    df: pd.DataFrame,
    value_column: str = "attribute_value",
    logger: logging.Logger = logging.getLogger(__name__),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cleans and classifies string values as 'varchar', 'categorical', or 'categorical2'.

    Returns:
        A tuple of three DataFrames: (passed_df, mod_df, remaining_df).
        - passed_df: Rows that were successfully classified.
        - mod_df: Always an empty DataFrame for this cleaner.
        - remaining_df: Rows that did not meet the criteria (e.g., are purely numeric).
    """
    if df.empty:
        return df, df.copy(), df.copy()

    work_df = df.copy()
    stripped = work_df[value_column].astype(str).str.strip()

    # Identify candidates: strings that contain no digits or start with a letter
    is_candidate_mask = (
        work_df[value_column].notna()
        & stripped.ne("")
        & (
            stripped.str.contains(r"^\D*$", regex=True)
            | stripped.str.match(r"^[A-Za-z].*\d", na=False)
        )
    )

    candidate_df = work_df[is_candidate_mask].copy()
    remaining_df = work_df[~is_candidate_mask].copy()

    logger.info(f"{len(candidate_df)} rows matched regex; {len(remaining_df)} did not.")

    if candidate_df.empty:
        return candidate_df, candidate_df.copy(), remaining_df

    candidate_df["value1"] = candidate_df[value_column].str.lower()
    candidate_df["data_type"] = pd.NA

    # Logic for classification based on commas and word count
    comma_mask = candidate_df[value_column].astype(str).str.contains(",", na=False)

    # With comma: >3 words in any part -> varchar, else -> categorical2
    if comma_mask.any():
        splits = candidate_df.loc[comma_mask, value_column].astype(str).str.split(",")
        max_words = splits.apply(
            lambda parts: max(len(p.strip().split()) for p in parts) if parts else 0
        )
        candidate_df.loc[comma_mask, "data_type"] = np.where(
            max_words > 3, "varchar", "categorical2"
        )

    # No comma: >3 words -> varchar, else -> categorical
    no_comma_mask = ~comma_mask
    if no_comma_mask.any():
        word_counts = (
            candidate_df.loc[no_comma_mask, value_column]
            .astype(str)
            .str.split()
            .str.len()
        )
        candidate_df.loc[no_comma_mask, "data_type"] = np.where(
            word_counts > 3, "varchar", "categorical"
        )

    # For this function, mod_df is always empty. Return an empty frame for consistency.
    mod_df = pd.DataFrame(columns=candidate_df.columns)

    return candidate_df, mod_df, remaining_df
