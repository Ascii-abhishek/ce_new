import logging
import re
from typing import List, Tuple

import pandas as pd

from ..ce_helpers import build_sep_pattern, get_compiled_regex, standardize_unit
from ..ce_utils import check_required_columns
from .numerical import clean_numerical_unit


def _cleanup_range(
    data_df: pd.DataFrame, logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validates the structure of range data. Swaps values to ensure order
    and syncs unit fields. Does NOT standardize units.
    """
    if data_df.empty:
        return data_df, data_df.copy()

    df_work = data_df.copy()

    # Logic to swap values/polarities as before
    mask_swap_polarity = (df_work["polarity1"].astype(str).fillna("") == "+") & (
        df_work["polarity2"].astype(str).fillna("") == "-"
    )
    if mask_swap_polarity.any():
        rows = df_work.loc[mask_swap_polarity].index
        df_work.loc[rows, ["polarity1", "polarity2"]] = df_work.loc[
            rows, ["polarity2", "polarity1"]
        ].values
        df_work.loc[rows, ["value1", "value2"]] = df_work.loc[
            rows, ["value2", "value1"]
        ].values

    # Sync units: fill missing unit from the other side
    df_work["unit1"] = df_work["unit1"].fillna(df_work["unit2"])
    df_work["unit2"] = df_work["unit2"].fillna(df_work["unit1"])

    # After syncing, check if units match. If not, they are modified.
    mismatch_mask = (
        df_work["unit1"].notna()
        & df_work["unit2"].notna()
        & (
            df_work["unit1"].astype(str).str.lower()
            != df_work["unit2"].astype(str).str.lower()
        )
    )

    mod_df = df_work[mismatch_mask].copy()
    if not mod_df.empty:
        mod_df["mod_reason"] = "Unit mismatch in range"

    passed_df = df_work[~mismatch_mask].copy()

    return passed_df.reset_index(drop=True), mod_df.reset_index(drop=True)


def clean_range_with_to_and_hyphen(
    df: pd.DataFrame,
    unit_df: pd.DataFrame,
    *,
    value_column: str = "attribute_value",
    separators: List[str] | None = None,
    logger: logging.Logger = logging.getLogger(__name__),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generic range parser that re‑uses *numeric_with_optional_unit* for each
    endpoint while preserving the original text in *attribute_value*.

    Parameters
    ----------
    df : DataFrame
        Input records.
    unit_df : DataFrame
        Standard unit reference table for :pyfunc:`cleanup_range`.
    value_column : str, default ``"attribute_value"``
        Column holding the raw text to parse.
    separators : list[str], default ``["to", "-"]``
        Plain strings that separate the two numeric tokens (case‑insensitive for
        alphabetic separators).  Each separator is treated literally (regex
        escaped) and matched with optional surrounding whitespace.  Example::

            separators=["to", "–", "-"]

    logger : logging.Logger
        Injected logger.

    Returns
    -------
    (passed, modified, remaining) : tuple[DataFrame, DataFrame, DataFrame]
        Where *passed* has ``data_type='range1'`` ready for downstream, *modified*
        captures rows that failed numeric/unit validation, and *remaining*
        contains values that never matched the range grammar at all.
    """

    if separators is None:
        separators = ["to"]  # , "-"]

    # ------------------------------------------------------------------
    # 0. Early‑exit on empty input
    # ------------------------------------------------------------------
    if df.empty:
        base_cols = list(df.columns) + [
            "polarity1",
            "value1",
            "unit1",
            "polarity2",
            "value2",
            "unit2",
            "data_type",
            "key_value",
        ]
        return (
            pd.DataFrame(columns=base_cols),
            pd.DataFrame(columns=base_cols + ["mod_reason"]),
            pd.DataFrame(columns=df.columns),
        )

    # ------------------------------------------------------------------
    # 1. Build range regex
    # ------------------------------------------------------------------
    num_pattern = get_compiled_regex("numeric_with_optional_unit").pattern
    num_pattern = num_pattern.lstrip("^").rstrip("$")  # un‑anchor embedded pattern

    sep_pattern = build_sep_pattern(separators)

    range_regex = re.compile(
        rf"^\s*({num_pattern}){sep_pattern}({num_pattern})\s*$",  # noqa: W605,E501
        re.IGNORECASE | re.VERBOSE,
    )

    work = df.copy()
    work["_is_range"] = (
        work[value_column]
        .astype(str)
        .apply(lambda x: bool(range_regex.match(x.strip())))
    )

    candidate_df = work[work["_is_range"]].copy()
    remaining_df = work[~work["_is_range"]].drop(columns=["_is_range"]).copy()

    logger.info(
        f"{len(candidate_df)} rows matched clean_range regex; {len(remaining_df)} did not."
    )

    # ------------------------------------------------------------------
    # 2. Return early if nothing matched
    # ------------------------------------------------------------------
    if candidate_df.empty:
        base_cols = list(df.columns) + [
            "polarity1",
            "value1",
            "unit1",
            "polarity2",
            "value2",
            "unit2",
            "data_type",
            "key_value",
        ]
        return (
            pd.DataFrame(columns=base_cols),
            pd.DataFrame(columns=base_cols + ["mod_reason"]),
            remaining_df,
        )

    # ------------------------------------------------------------------
    # 3. Explode each candidate into endpoint rows
    # ------------------------------------------------------------------
    candidate_df["_orig_value"] = candidate_df[value_column]
    candidate_df["_orig_idx"] = candidate_df.index  # linkage key

    # Build *exactly the same* separator pattern without anchors for splitting
    split_pattern = re.compile(sep_pattern, flags=re.IGNORECASE)
    candidate_df["_parts"] = (
        candidate_df[value_column]
        .astype(str)
        .apply(lambda s: re.split(split_pattern, s, maxsplit=1))
    )

    exploded = candidate_df.explode("_parts").copy()
    exploded["_parts"] = exploded["_parts"].astype(str).str.strip()
    exploded["key_value"] = exploded["_orig_idx"]

    # Prepare input for clean_numerical_unit
    numeric_input = exploded.drop(
        columns=[value_column, "_is_range", "_orig_idx"], errors="ignore"
    ).rename(columns={"_parts": value_column})

    numeric_pass, numeric_mod, numeric_remain = clean_numerical_unit(
        numeric_input, unit_df=unit_df, value_column=value_column, logger=logger
    )

    # Collate modifications & remaining mismatches

    dfs_to_concat_remain = [df for df in [remaining_df, numeric_remain] if not df.empty]
    if dfs_to_concat_remain:
        remaining_df = pd.concat(dfs_to_concat_remain, ignore_index=True)

    # ------------------------------------------------------------------
    # 4. Re‑assemble cleaned endpoints & restore original attribute text
    # ------------------------------------------------------------------
    range_records = []
    for _, grp in numeric_pass.groupby("key_value", sort=False):
        grp_sorted = grp.sort_index()
        lhs = grp_sorted.iloc[0].to_dict()
        rhs = grp_sorted.iloc[1].to_dict() if len(grp_sorted) > 1 else {}

        combined = lhs.copy()
        combined["polarity2"] = rhs.get("polarity1", pd.NA)
        combined["value2"] = rhs.get("value1", pd.NA)
        combined["unit2"] = rhs.get("unit1", pd.NA)

        combined[value_column] = candidate_df.loc[combined["key_value"], "_orig_value"]
        combined["data_type"] = "range1"

        range_records.append(combined)

    # ------------------------------------------------------------------
    # 5. Validate range semantics & unit compatibility
    # ------------------------------------------------------------------
    range_df = pd.DataFrame(range_records)
    all_mod_dfs = [df for df in [numeric_mod] if not df.empty]
    if not range_df.empty:
        range_pass, range_mod = _cleanup_range(range_df, logger=logger)
        if not range_mod.empty:
            all_mod_dfs.append(range_mod)
    else:
        range_pass = pd.DataFrame(columns=range_df.columns)

    if not range_pass.empty:
        passed_final, unit_mod = standardize_unit(range_pass, unit_df, "unit1", logger)
        if not unit_mod.empty:
            unit_mod["mod_reason"] = "Invalid unit in range"
            all_mod_dfs.append(unit_mod)
        # Sync unit2 with standardized unit1
        if not passed_final.empty:
            passed_final["unit2"] = passed_final["unit1"]
    else:
        passed_final = range_pass

    mod_df = (
        pd.concat(all_mod_dfs, ignore_index=True) if all_mod_dfs else pd.DataFrame()
    )

    # ------------------------------------------------------------------
    # 6. Drop helper columns and enforce required column set
    # ------------------------------------------------------------------
    helper_cols = ["_orig_value", "key_value"]
    range_pass = range_pass.drop(columns=helper_cols, errors="ignore")
    mod_df = mod_df.drop(columns=helper_cols, errors="ignore")
    remaining_df = remaining_df.drop(columns=helper_cols, errors="ignore")

    range_pass = check_required_columns(df=range_pass, logger=logger)
    mod_df = check_required_columns(df=mod_df, logger=logger)
    remaining_df = check_required_columns(df=remaining_df, logger=logger)

    return range_pass, mod_df, remaining_df
