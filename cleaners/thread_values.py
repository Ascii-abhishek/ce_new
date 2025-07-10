import logging
import re
from typing import Any, Tuple

import numpy as np
import pandas as pd

from ..ce_helpers import get_compiled_regex
from ..ce_utils import check_required_columns, log_step
from ..constants import REQUIRED_COLUMNS
from .numerical import clean_numerical_unit


@log_step
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
