import logging
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .ce_helpers import *
from .ce_utils import (
    check_required_columns,
    format_numeric_value,
    safe_str_to_float,
    setup_logger,
)
from .generate_display import build_display_values
from utils.db_utils import df_from_query
from utils.pandas_read_data_utils import ensure_pd_df
from .ce_constants import REQUIRED_COLUMNS


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


def _build_sep_pattern(separators: List[str]) -> str:
    """Builds a regex pattern for a list of separators."""
    escaped = sorted([re.escape(s.strip()) for s in separators], key=len, reverse=True)
    return rf"\s*(?:{'|'.join(escaped)})\s*"


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


#######################################################################################################
#######################################################################################################


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

    candidate_df["value1"] = candidate_df[value_column]
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


def clean_thread(
    df: pd.DataFrame,
    unit_df: pd.DataFrame,
    *,
    value_column: str = "attribute_value",
    logger: logging.Logger = logging.getLogger(__name__),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits unified thread specifications into two numeric tokens and feeds them
    through ``clean_numerical_unit``.  Two grammars are recognised:

    ▸ **Imperial (TPI)**  – “#10-32”, “3/8 - 16”, “5/8 in - 11”…
      → *thread dia.* | *threads per inch*

    ▸ **Metric (pitch)**  – “M12 × 1.5", “m10 x 1.25” …
      → *thread dia.* | *thread pitch*

    The function explodes every match into **two rows**, tied together by a
    ``key_value`` link, and then restores the original text so downstream code
    still sees the full spec.
    """

    # ------------------------------------------------------------------ 0. guard
    if df.empty:
        cols = REQUIRED_COLUMNS.get("base_df")
        return (
            pd.DataFrame(columns=cols),
            pd.DataFrame(columns=cols),
            pd.DataFrame(columns=cols),
        )

    # ------------------------------------------------------------------ 1. scope
    mask = (
        df["attribute_name"]
        .astype(str)
        .str.contains("thread|screw size", case=False, na=False)
    )
    cand_df = df[mask].copy()
    remain_df = df[~mask].copy()

    logger.info(
        f"{len(cand_df)} rows hit attribute “thread”; {len(remain_df)} did not."
    )

    if cand_df.empty:  # nothing matched at all
        base_cols = list(df.columns)
        return (
            pd.DataFrame(columns=base_cols),
            pd.DataFrame(columns=base_cols + ["mod_reason"]),
            remain_df,
        )

    # ------------------------------------------------------------------ 2. regexes
    num_pat = get_compiled_regex("numeric_with_optional_unit").pattern.strip("^$")
    grp1_re = re.compile(
        rf"""
        ^\s*             # optional leading space
        \#?              # optional hash
        (?P<dia>{num_pat})  # the diameter token (full pattern)
        \s*-\s*          # hyphen with optional spaces
        (?P<tpi>[0-9]+)  # threads-per-inch: plain integer
        \s*$             # optional trailing space
        """,
        re.I | re.X,
    )
    grp2_re = get_compiled_regex("thread_metric")
    # ------------------------------------------------------------------ 3. explode
    exploded: list[dict[str, Any]] = []
    not_match: list[pd.Series] = []

    for idx, row in cand_df.iterrows():
        raw = str(row[value_column]).strip()

        m1 = grp1_re.match(raw)
        m2 = grp2_re.match(raw)

        if m1:  # ── imperial
            dia = m1.group("dia").lstrip("#").strip()
            tpi = m1.group("tpi").strip()
            parts = [
                ("thread dia.", dia),
                ("threads per inch", tpi),
            ]

        elif m2:  # ── metric
            core = re.sub(r"^\s*[mM]\s*", "", raw)
            left, right = map(str.strip, re.split(r"[x×]", core, 1))
            right = re.sub(r'(?i)(?:mm|")\s*$', "", right)
            parts = [
                ("thread dia.", left),
                ("thread pitch", right),
            ]
        else:
            not_match.append(row)
            continue

        for child, token in parts:
            rec = row.to_dict()
            rec["child_attribute"] = child
            rec[value_column] = token
            rec["key_value"] = idx
            rec["_orig_attr"] = raw
            exploded.append(rec)

    if not exploded:  # nothing parseable; shove back for later steps
        remain_df = pd.concat([remain_df, pd.DataFrame(not_match)], ignore_index=True)
        base_cols = list(df.columns)
        return (
            pd.DataFrame(columns=base_cols),
            pd.DataFrame(columns=base_cols + ["mod_reason"]),
            remain_df,
        )

    exploded_df = pd.DataFrame(exploded)

    # ------------------------------------------------------------------ 4. numeric clean
    passed, mod, num_remain = clean_numerical_unit(
        exploded_df, unit_df=unit_df, value_column=value_column, logger=logger
    )

    if not num_remain.empty:  # push failures forward
        remain_df = pd.concat([remain_df, num_remain], ignore_index=True)

    # ------------------------------------------------------------------ 5. restore original text
    def _restore_original(df_sub: pd.DataFrame) -> pd.DataFrame:
        """
        Re-injects the full spec back into *attribute_value* by
        matching on (key_value, child_attribute).  Works even if the
        index got reset by downstream cleaners.
        """
        if df_sub.empty:
            return df_sub

        # Build a two-column key → original-spec map once
        map_orig = exploded_df.set_index(["key_value", "child_attribute"])[
            "_orig_attr"
        ].to_dict()

        df_sub[value_column] = [
            map_orig.get((kv, ca), orig)  # fallback keeps whatever was there
            for kv, ca, orig in zip(
                df_sub["key_value"],
                df_sub["child_attribute"],
                df_sub[value_column],
            )
        ]
        return df_sub

    passed = _restore_original(passed)
    mod = _restore_original(mod)

    # ------------------------------------------------------------------ 6. tidy helpers
    for _df in (passed, mod, remain_df):
        _df.drop(columns=["_orig_attr"], errors="ignore", inplace=True)

    # ------------------------------------------------------------------ 7. column order / fill-ins
    passed = check_required_columns(df=passed, logger=logger)
    mod = check_required_columns(df=mod, logger=logger)
    remain_df = check_required_columns(df=remain_df, logger=logger)  # default ctx

    # ------------------------------------------------------------------ 8. set display_value for thread specs

    for _df in (passed, mod):
        if not _df.empty:
            _df["display_value"] = _df["attribute_value"]

    passed["polarity1"] = np.where(
        (passed["attribute_value"].str.strip().str.startswith("#"))
        & (passed["child_attribute"] == "thread dia."),
        "#",
        passed["polarity1"],
    )

    return passed, mod, remain_df


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
    sep_pat = _build_sep_pattern(separators)
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

    sep_pattern = _build_sep_pattern(separators)

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

    logger.info(f"{len(candidate_df)} rows matched clean_range regex; {len(remaining_df)} did not.")

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


#######################################################################################################
#######################################################################################################


def run_cleanup_pipeline(
    input_data_to_clean: str,
    output_path: str = ".",
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    """
    Runs the full CE cleanup pipeline and saves intermediate outputs.
    """
    recreate_base_dir(base_dir=output_path)
    logger = setup_logger(output_path) if logger is None else logger

    raw_df = ensure_pd_df(input_data_to_clean)
    logger.info(f"▶️ Starting Cleaning Engine Pipeline on {len(raw_df)} rows.")

    df_cleaned = ce_start_cleanup(raw_df, logger)
    remain = df_cleaned
    logger.info(
        f"Initial cleanup complete. Starting sequential cleaners on {len(remain)} rows."
    )

    unit_df = df_from_query("select * from hercules_db.ce_unit_mapping")
    unit_df = check_required_columns(df=unit_df, df_name="unit_df", logger=logger)

    # --- Pipeline Steps ---
    def log_step(name, passed, mod, remain):
        logger.info(f"Function '{name}':")
        logger.info(f"  - Passed: {len(passed):>6} rows")
        logger.info(f"  - For mod: {len(mod):>5} rows")
        logger.info(f"  - Remaining: {len(remain):>6} rows")
        logger.info("-" * 40)

    # Step 1: Varchar and Categorical
    logger.info(f"1️⃣: clean_varchar_categorical on {len(remain)} rows...")
    passed, mod, remain = clean_varchar_categorical(df=remain, logger=logger)
    passed = build_display_values(passed, unit_df, logger=logger)
    save_dfs("clean_varchar_categorical", passed, mod, output_path, logger=logger)
    log_step("clean_varchar_categorical", passed, mod, remain)

    # Step 2: Numerical with/without Unit
    logger.info(f"2️⃣: clean_numerical_unit on {len(remain)} rows...")
    passed, mod, remain = clean_numerical_unit(
        df=remain, unit_df=unit_df, logger=logger
    )
    passed = build_display_values(passed, unit_df, logger=logger)
    save_dfs("clean_numerical_unit", passed, mod, output_path, logger=logger)
    log_step("clean_numerical_unit", passed, mod, remain)

    # Step 3: Thread Spec Filter
    logger.info(f"3️⃣: clean_thread on {len(remain)} rows...")
    passed, mod, remain = clean_thread(df=remain, unit_df=unit_df, logger=logger)
    save_dfs("clean_thread", passed, mod, output_path, logger=logger)
    log_step("clean_thread", passed, mod, remain)

    # Step 4: Dimension Values (e.g., 10 x 20)
    logger.info(f"4️⃣: clean_dimension_values on {len(remain)} rows...")
    passed, mod, remain = clean_dimension_values(
        df=remain, unit_df=unit_df, logger=logger
    )
    passed = build_display_values(passed, unit_df, logger=logger)
    save_dfs("clean_dimension_values", passed, mod, output_path, logger=logger)
    log_step("clean_dimension_values", passed, mod, remain)

    # Step 5: Range Values (e.g., 100 to 200)
    logger.info(f"5️⃣: clean_range_with_to_and_hyphen on {len(remain)} rows...")
    passed, mod, remain = clean_range_with_to_and_hyphen(
        df=remain, unit_df=unit_df, logger=logger
    )

    passed = build_display_values(passed, unit_df, logger=logger)
    save_dfs("clean_range_with_to_and_hyphen", passed, mod, output_path, logger=logger)
    log_step("clean_range_with_to_and_hyphen", passed, mod, remain)

    # Final Step: Save any remaining unprocessable rows
    logger.info(
        f"Pipeline finished. {len(remain)} rows could not be processed by any cleaner."
    )
    if not remain.empty:
        remain["mod_reason"] = "Unprocessed by any cleaner"
        remain_path = os.path.join(output_path, "mod", "final_remain.csv")
        remain.to_csv(remain_path, index=False)
        logger.info(f"Saved remaining rows to 'mod/final_remain.csv'.")

    logger.info("✅ Cleanup pipeline completed successfully.")
