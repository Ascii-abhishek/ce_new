import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ce_helpers import *
from utils.db_utils import df_from_query
from utils.pandas_read_data_utils import ensure_pd_df


def _calculate_mixed_fraction_value(
    whole_str: str,
    num_str: str,
    den_str: str,
    logger: logging.Logger,
    context: str,
) -> Tuple[float, str]:
    """
    Calculates the absolute value of a mixed fraction (e.g., "1 1/2").
    Returns (value, error_message). If there's an error, value is np.nan.
    """
    whole = safe_str_to_float(whole_str, logger, f"whole part in {context}")
    numerator = safe_str_to_float(num_str, logger, f"numerator in {context}")
    denominator = safe_str_to_float(den_str, logger, f"denominator in {context}")

    if np.isnan(whole) or np.isnan(numerator) or np.isnan(denominator):
        return np.nan, "Invalid number in mixed fraction components"
    if denominator == 0.0:
        return np.nan, "Division by zero in mixed fraction"
    total = abs(whole) + abs(numerator) / abs(denominator)
    return total, ""


def _calculate_simple_fraction_value(
    num_str: str,
    den_str: str,
    logger: logging.Logger,
    context: str,
) -> Tuple[float, str]:
    """
    Calculates the absolute value of a simple fraction (e.g., "3/4").
    Returns (value, error_message). If there's an error, value is np.nan.
    """
    numerator = safe_str_to_float(num_str, logger, f"numerator in {context}")
    denominator = safe_str_to_float(den_str, logger, f"denominator in {context}")

    if np.isnan(numerator) or np.isnan(denominator):
        return np.nan, "Invalid number in simple fraction components"
    if denominator == 0.0:
        return np.nan, "Division by zero in simple fraction"
    total = abs(numerator) / abs(denominator)
    return total, ""


def _convert_decimal_to_float(
    val_str: str,
    logger: logging.Logger,
    context: str,
) -> Tuple[float, str]:
    """
    Converts a decimal or integer string (possibly containing commas) to absolute float.
    Returns (value, error_message). If there's an error, value is np.nan.
    """
    if pd.isna(val_str) or not isinstance(val_str, str) or not val_str.strip():
        return np.nan, "Empty or invalid decimal/integer string"

    cleaned = val_str.replace(",", "").strip()
    if not cleaned:
        return np.nan, "Empty string after removing commas"
    if cleaned.count(".") > 1:
        return np.nan, "Invalid decimal format: multiple decimal points"

    # Remove leading sign, but keep track of it if needed
    if cleaned[0] in ["+", "-"]:
        cleaned = cleaned[1:].strip()

    if not cleaned:
        return np.nan, "Empty string after stripping sign/commas"

    value = safe_str_to_float(cleaned, logger, f"value part '{cleaned}' in {context}")
    if np.isnan(value):
        return np.nan, "Invalid character(s) in decimal/integer string"
    return abs(value), ""


def _process_row_for_numerical_unit(
    row: pd.Series,
    regex_cols_map: Dict[str, str],
    logger: logging.Logger,
) -> Tuple[float, str]:
    """
    Given a row with extracted regex groups, compute the numeric value (absolute) and
    return (value, error_reason). If no pattern matches, returns (np.nan, reason).
    """
    context = f"attribute_value '{row.get('attribute_value', '')}'"
    val = np.nan
    reason = ""

    # Check for mixed fraction first
    if pd.notna(row[regex_cols_map["mixed_whole"]]):
        val, reason = _calculate_mixed_fraction_value(
            row[regex_cols_map["mixed_whole"]],
            row[regex_cols_map["mixed_num"]],
            row[regex_cols_map["mixed_den"]],
            logger,
            context + " (mixed fraction)",
        )
    # Then check for simple fraction
    elif pd.notna(row[regex_cols_map["simple_num"]]):
        val, reason = _calculate_simple_fraction_value(
            row[regex_cols_map["simple_num"]],
            row[regex_cols_map["simple_den"]],
            logger,
            context + " (simple fraction)",
        )
    # Then check for decimal/integer
    elif pd.notna(row[regex_cols_map["decimal"]]):
        val, reason = _convert_decimal_to_float(
            row[regex_cols_map["decimal"]],
            logger,
            context + " (decimal/integer)",
        )
    else:
        reason = "No recognizable numeric format"
    return val, reason


def _build_sep_pattern(separators: List[str]) -> str:
    """Return a non‑capturing group pattern like ``(?:to|-)`` with optional
    surrounding whitespace.  All *separators* are regex‑escaped and trimmed so
    callers can pass plain strings such as ``["to", "-"]``.
    """
    escaped = [re.escape(s.strip()) for s in separators]
    alternation = "|".join(sorted(escaped, key=len, reverse=True))  # longer first
    return rf"\s*(?:{alternation})\s*"  # include surrounding whitespace flexibly


def _cleanup_range(
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
        vals1 = sub["value1"].apply(safe_str_to_float, args=(logger, "value1 in cleanup_range"),
                                    )
        vals2 = sub["value2"].apply(safe_str_to_float, args=(logger, "value2 in cleanup_range"),
                                    )
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

#######################################################################################################
#######################################################################################################


def clean_varchar_categorical(
    df: pd.DataFrame,
    value_column: str = "attribute_value",
    logger: logging.Logger = logging.getLogger(__name__),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cleans attribute values to classify them as 'varchar', 'categorical', or 'categorical2'.

    This function processes string values that do not contain any numbers.
    The logic is as follows:
    1.  If a value contains a comma:
        - It splits the value by the comma.
        - If any resulting part has more than 3 words, it's classified as 'varchar'.
        - Otherwise, it's classified as 'categorical2'.
    2.  If a value does not contain a comma:
        - If it has more than 3 words, it's classified as 'varchar'.
        - Otherwise (3 or fewer words), it's classified as 'categorical'.

    Args:
        df: The input DataFrame to process.
        value_column: The name of the column containing the values to clean.
        logger: The logger instance for logging messages.

    Returns:
        A tuple of three DataFrames: (passed_df, mod_df, remaining_df).
        - passed_df: Rows that were successfully classified.
        - mod_df: Always empty for this cleaner, returned for pipeline consistency.
        - remaining_df: Rows that did not meet the criteria for this cleaner (e.g., contained numbers).
    """
    logger.info(f"Starting clean_varchar_categorical on {len(df)} rows.")
    if df.empty:
        # Define expected columns for empty DataFrames
        passed_cols = list(df.columns) + ["value1", "data_type"]
        mod_cols = list(df.columns) + ["mod_reason"]
        remaining_cols = list(df.columns)
        return (
            pd.DataFrame(columns=passed_cols),
            pd.DataFrame(columns=mod_cols),
            pd.DataFrame(columns=remaining_cols),
        )

    work_df = df.copy()

    stripped = work_df[value_column].astype(str).str.strip()
    is_candidate_mask = (
        work_df[value_column].notna()                 # not null
        & stripped.ne("")                             # not an empty string
        & (                                           # ── numeric rule ──
            stripped.str.contains(r"^\D*$", regex=True)          # A) no digits at all
            | stripped.str.match(r"^[A-Za-z].*\d", na=False)     # B) has digits but 1st char is a letter
        )
    )

    candidate_df = work_df[is_candidate_mask].copy()
    remaining_df = work_df[~is_candidate_mask].copy()

    if candidate_df.empty:
        logger.info("No candidate rows for varchar/categorical cleaning.")
        passed_cols = list(df.columns) + ["value1", "data_type"]
        mod_cols = list(df.columns) + ["mod_reason"]
        return (
            pd.DataFrame(columns=passed_cols),
            pd.DataFrame(columns=mod_cols),
            df,
        )

    candidate_df["value1"] = candidate_df[value_column]
    candidate_df["data_type"] = pd.NA

    # Condition 1: Value contains a comma
    comma_mask = candidate_df[value_column].astype(str).str.contains(",", na=False)

    # For rows with commas, split by comma and check word counts of each part
    if comma_mask.any():
        splits = candidate_df.loc[comma_mask, value_column].astype(str).str.split(",")
        # Check if any split part has more than 3 words
        max_words_in_split = splits.apply(lambda parts: max(len(p.strip().split()) for p in parts) if parts else 0)

        # If any part > 3 words -> varchar
        is_varchar_mask = max_words_in_split > 3
        candidate_df.loc[comma_mask & is_varchar_mask, "data_type"] = "varchar"
        candidate_df.loc[comma_mask & ~is_varchar_mask, "data_type"] = "categorical2"

    # Condition 2: Value does not contain a comma
    no_comma_mask = ~comma_mask
    if no_comma_mask.any():
        word_counts = candidate_df.loc[no_comma_mask, value_column].astype(str).str.split().str.len()

        # If > 3 words -> varchar
        is_varchar_mask = word_counts > 3
        candidate_df.loc[no_comma_mask & is_varchar_mask, "data_type"] = "varchar"

        # If <= 3 words -> categorical
        candidate_df.loc[no_comma_mask & ~is_varchar_mask, "data_type"] = "categorical"

    passed_df = candidate_df

    logger.info(
        f"Finished clean_varchar_categorical: {len(passed_df)} passed, "
        f"{len(remaining_df)} remaining."
    )

    return passed_df, remaining_df


def clean_numerical_unit(
    df: pd.DataFrame,
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
    logger.info(f"Starting clean_numerical_unit on {len(df)} rows.")
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
    no_match_mask = combined["mixed_whole"].isna() & combined["simple_num"].isna() & combined["decimal"].isna()
    remaining_df = combined.loc[no_match_mask, [*df_working.columns]].copy()
    passed_intermediate = combined.loc[~no_match_mask].copy()

    logger.info(f"{len(passed_intermediate)} rows matched regex; {len(remaining_df)} did not.")

    if not passed_intermediate.empty:
        # Determine polarity from sign
        passed_intermediate["polarity1"] = passed_intermediate["sign"].replace({"": pd.NA, np.nan: pd.NA})
        # Assign unit directly from regex group
        passed_intermediate["unit1"] = passed_intermediate["unit"].where(passed_intermediate["unit"].notna(), pd.NA)

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
            lambda row: _process_row_for_numerical_unit(row, regex_cols_map, logger), axis=1
        )
        passed_intermediate[["value1", "mod_reason"]] = pd.DataFrame(
            values_reasons.tolist(), index=passed_intermediate.index
        )

        # Coerce value1 to numeric
        passed_intermediate["value1"] = pd.to_numeric(passed_intermediate["value1"], errors="coerce")

        # Determine data_type based on presence of unit
        has_unit = passed_intermediate["unit1"].notna() & (passed_intermediate["unit1"] != "")
        passed_intermediate["data_type"] = np.where(has_unit, "numerical_with_unit", "numerical_without_unit")

        # Split into passed (no errors) and modified_for_error (errors present)
        error_mask = passed_intermediate["mod_reason"].notna() & (passed_intermediate["mod_reason"] != "")
        mod_df = passed_intermediate.loc[error_mask].copy()
        passed_df = passed_intermediate.loc[~error_mask].copy()
    else:
        passed_df = pd.DataFrame(columns=combined.columns)
        mod_df = pd.DataFrame(columns=combined.columns)

    # Reorder and drop helper columns
    passed_df = check_required_columns(passed_df, logger)
    mod_df = check_required_columns(mod_df, logger)
    remaining_df = check_required_columns(remaining_df, logger)

    logger.info(
        f"Finished clean_numerical_unit: {len(passed_df)} passed, "
        f"{len(mod_df)} had errors, {len(remaining_df)} did not match."
    )
    return passed_df, mod_df, remaining_df


def clean_thread(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        cols = list(df.columns)
        return (
            pd.DataFrame(columns=cols),
            pd.DataFrame(columns=cols + ["mod_reason"]),
            pd.DataFrame(columns=cols),
        )

    mask = df["attribute_name"].astype(str).str.contains("thread", case=False, na=False)
    mod_df = df.loc[mask].copy()
    if not mod_df.empty:
        mod_df["mod_reason"] = "spec with thread"
    remaining_df = df.loc[~mask].copy()

    passed_df = pd.DataFrame(columns=df.columns)
    return passed_df, mod_df, remaining_df


def clean_dimension_values(
    df: pd.DataFrame,
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
    work["_matches_dim"] = work[value_column].astype(str).apply(lambda s: bool(dim_re.match(s.strip())))

    cand_df = work[work["_matches_dim"]].copy()
    remain_df = work[~work["_matches_dim"]].drop(columns=["_matches_dim"], errors="ignore").copy()

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
    cand_df["_parts"] = cand_df[value_column].astype(str).apply(lambda s: re.split(split_re, s))

    exploded = cand_df.explode("_parts").copy()
    exploded[value_column] = exploded["_parts"].str.strip()
    exploded["key_value"] = exploded["_idx"]

    num_input = (
        exploded
        .drop(columns=["_parts", "_matches_dim", "_idx"], errors="ignore")
    )

    pass_df, mod_df, rem_df = clean_numerical_unit(num_input, value_column=value_column, logger=logger)

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
    pass_df = check_required_columns(pass_df, logger)
    mod_df = check_required_columns(mod_df, logger)
    remain_df = check_required_columns(remain_df, logger)

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
        separators = ["to", "-"]

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
    work["_is_range"] = work[value_column].astype(str).apply(
        lambda x: bool(range_regex.match(x.strip()))
    )

    candidate_df = work[work["_is_range"]].copy()
    remaining_df = work[~work["_is_range"]].drop(columns=["_is_range"]).copy()

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
    candidate_df["_parts"] = candidate_df[value_column].astype(str).apply(
        lambda s: re.split(split_pattern, s, maxsplit=1)
    )

    exploded = candidate_df.explode("_parts").copy()
    exploded["_parts"] = exploded["_parts"].astype(str).str.strip()
    exploded["key_value"] = exploded["_orig_idx"]

    # Prepare input for clean_numerical_unit
    numeric_input = (
        exploded
        .drop(columns=[value_column, "_is_range", "_orig_idx"], errors="ignore")
        .rename(columns={"_parts": value_column})
    )

    numeric_pass, numeric_mod, numeric_remain = clean_numerical_unit(
        numeric_input, value_column=value_column, logger=logger
    )

    # Collate modifications & remaining mismatches
    mod_df = numeric_mod.copy()
    if not numeric_remain.empty:
        remaining_df = pd.concat([remaining_df, numeric_remain], ignore_index=True)

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

    range_df = pd.DataFrame(range_records)

    # ------------------------------------------------------------------
    # 5. Validate range semantics & unit compatibility
    # ------------------------------------------------------------------
    range_pass, range_mod = _cleanup_range(range_df, unit_df, logger=logger)
    mod_df = pd.concat([mod_df, range_mod], ignore_index=True)

    # ------------------------------------------------------------------
    # 6. Drop helper columns and enforce required column set
    # ------------------------------------------------------------------
    helper_cols = ["_orig_value", "key_value"]
    range_pass = range_pass.drop(columns=helper_cols, errors="ignore")
    mod_df = mod_df.drop(columns=helper_cols, errors="ignore")
    remaining_df = remaining_df.drop(columns=helper_cols, errors="ignore")

    range_pass = check_required_columns(range_pass, logger)
    mod_df = check_required_columns(mod_df, logger)
    remaining_df = check_required_columns(remaining_df, logger)

    return range_pass, mod_df, remaining_df


def run_cleanup_pipeline(
    raw_df: pd.DataFrame,
    unit_df: pd.DataFrame,
    base_dir: str = ".",
) -> None:
    """Run the full CE cleanup pipeline and materialise intermediate outputs.

    The sequence is:
      1. ce_start_cleanup
      2. clean_numerical_unit
      3. clean_thread
      4. clean_dimension_values
      5. clean_range_with_to_and_plus_minus

    *passed* frames go to ``<base_dir>/passed``; *mod* frames (including the final
    *remain*) to ``<base_dir>/mod``.
    """

    # start‑of‑pipe cleanup
    df_after_start = ce_start_cleanup(raw_df, base_dir)
    remain = df_after_start

    # 0 - varchar and categorical
    passed, remain = clean_varchar_categorical(remain)
    save_dfs("clean_varchar_categorical", passed_df=passed, mod_df=None, base_dir=base_dir)

    # 1 – numerical units -------------------------------------------------------
    passed, mod, remain = clean_numerical_unit(remain)
    save_dfs("clean_numerical_unit", passed, mod, base_dir)

    # 2 – thread spec filter ----------------------------------------------------
    passed, mod, remain = clean_thread(remain)
    save_dfs("clean_thread", passed, mod, base_dir)

    # 3 – dimension (X × Y) values --------------------------------------------
    passed, mod, remain = clean_dimension_values(remain)
    save_dfs("clean_dimension_values", passed, mod, base_dir)

    # # 4 – ranges ("100 to 200") ------------------------------------------------
    passed, mod, remain = clean_range_with_to_and_hyphen(remain, unit_df)
    save_dfs("clean_range_with_to_and_hyphen", passed, mod, base_dir)

    # 5 – whatever is *still* left becomes final remain and is saved as mod -----
    if not remain.empty:
        remain_path = os.path.join(base_dir, "mod", "final_remain.csv")
        remain.to_csv(remain_path, index=False)


if __name__ == '__main__':
    df = ensure_pd_df("/home/abhishek/projects/cleaning/data/sealmaster_part2_for_cleaning.xlsx")
    unit_df = df_from_query("select units, display from hercules_db.ce_unit_mapping")
    run_cleanup_pipeline(df, unit_df)
