import logging
import re
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..ce_helpers import build_sep_pattern, get_compiled_regex
from ..ce_utils import check_required_columns, log_step
from .numerical import clean_numerical_unit


@log_step
def clean_dimension_values(
    df: pd.DataFrame,
    unit_df: pd.DataFrame,
    *,
    value_column: str = "attribute_value",
    separators: List[str] | None = None,
    logger: logging.Logger = logging.getLogger(__name__),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse *dimension* strings like

        ``10 x 20 x 30 cm``
        ``5×7 in``

    where *x* (lower/upper/Unicode) is the delimiter.  Each numeric fragment is
    cleaned by :pyfunc:`clean_numerical_unit` **and kept in its own row**, all
    linked back to the original record via a ``key_value`` column.  The
    original full text is preserved in *attribute_value* for every output row.

    Only values that match **exactly** 2‑ or 3‑component dimension grammar are
    processed.  Others fall through untouched to *remaining_df*.
    """

    if separators is None:
        separators = ["x", "×", "X"]  # handle common ASCII & Unicode variants

    # 0. Guard clause --------------------------------------------------
    if df.empty:
        empty_cols = list(df.columns) + ["key_value"]
        return (
            pd.DataFrame(columns=empty_cols),
            pd.DataFrame(columns=empty_cols + ["mod_reason"]),
            pd.DataFrame(columns=empty_cols),
        )

    # 1. Compile grammar regex ----------------------------------------
    num_re = get_compiled_regex("numeric_with_optional_unit").pattern.strip("^$")
    sep_pat = build_sep_pattern(separators)
    dim_re = re.compile(rf"^\s*{num_re}(?:{sep_pat}{num_re}){{1,2}}\s*$", re.I | re.X)

    work = df.copy()
    work["_matches_dim"] = (
        work[value_column].astype(str).apply(lambda s: bool(dim_re.match(s.strip())))
    )

    cand_df = work[work["_matches_dim"]].copy()
    remain_df = (
        work[~work["_matches_dim"]]
        .drop(columns=["_matches_dim"], errors="ignore")
        .copy()
    )
    logger.info(f"{len(cand_df)} rows matched regex; {len(remain_df)} did not.")

    if cand_df.empty:
        empty_cols = list(df.columns) + ["key_value"]
        return (
            pd.DataFrame(columns=empty_cols),
            pd.DataFrame(columns=empty_cols + ["mod_reason"]),
            remain_df,
        )

    # 2. Explode numeric parts ---------------------------------------
    cand_df["_orig_attr"] = cand_df[value_column]
    cand_df["_idx"] = cand_df.index

    split_re = re.compile(sep_pat, re.I)
    cand_df["_parts"] = (
        cand_df[value_column].astype(str).apply(lambda s: re.split(split_re, s))
    )

    exploded = cand_df.explode("_parts").copy()
    exploded[value_column] = exploded["_parts"].str.strip()
    exploded["key_value"] = exploded["_idx"]

    num_input = exploded.drop(
        columns=["_parts", "_matches_dim", "_idx"], errors="ignore"
    )

    pass_df, mod_df, rem_df = clean_numerical_unit(
        num_input, unit_df=unit_df, value_column=value_column, logger=logger
    )

    # Any rem_df rows indicate endpoints that failed numeric grammar – they
    # belong in *remain_df* so downstream logic can decide what to do.
    if not rem_df.empty:
        remain_df = pd.concat([remain_df, rem_df], ignore_index=True)

    # 3. Restore original attr text ----------------------------------
    for _df in (pass_df, mod_df):
        if not _df.empty:
            _df[value_column] = cand_df.loc[_df["key_value"], "_orig_attr"].values

    # Clean up helper columns
    for _df in (pass_df, mod_df, remain_df):
        _df.drop(columns=["_orig_attr"], errors="ignore", inplace=True)

    # 4. Column sanity -----------------------------------------------
    pass_df = check_required_columns(df=pass_df, logger=logger)
    mod_df = check_required_columns(df=mod_df, logger=logger)
    remain_df = check_required_columns(df=remain_df, logger=logger)

    return pass_df, mod_df, remain_df
