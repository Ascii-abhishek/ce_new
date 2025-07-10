import logging
import os

import pandas as pd
from utils.db_utils import df_from_query
from utils.pandas_read_data_utils import ensure_pd_df

from .ce_helpers import ce_start_cleanup, recreate_base_dir, save_dfs
from .ce_utils import check_required_columns, setup_logger
from .cleaners import (clean_dimension_values, clean_numerical_unit,
                       clean_range_with_to_and_hyphen, clean_thread,
                       clean_varchar_categorical)
from .display import build_display_values


def run_cleanup_pipeline(
    input_data_to_clean: str,
    output_path: str = ".",
    logger: logging.Logger | None = None,
) -> None:
    # ——— Setup ——————————————————————————————————————————————————————————
    recreate_base_dir(base_dir=output_path)
    logger = setup_logger(output_path) if logger is None else logger

    raw_df = ensure_pd_df(input_data_to_clean)
    total = len(raw_df)
    logger.info(f"▶️ Pipeline start: total raw rows = {total}")

    # ——— Initial cleanup ——————————————————————————————————————————————————
    df0 = ce_start_cleanup(raw_df, logger)
    logger.info(f"✔ After ce_start_cleanup: {len(df0)} rows ready for cleaners")

    unit_df = df_from_query("select * from hercules_db.ce_unit_mapping")
    unit_df = check_required_columns(df=unit_df, df_name="unit_df", logger=logger)

    # ——— Define steps ————————————————————————————————————————————————————
    steps = [
        (
            "clean_varchar_categorical",
            lambda df: clean_varchar_categorical(df, logger=logger),
            True,
        ),
        (
            "clean_numerical_unit",
            lambda df: clean_numerical_unit(df, unit_df=unit_df, logger=logger),
            True,
        ),
        (
            "clean_thread",
            lambda df: clean_thread(df, unit_df=unit_df, logger=logger),
            False,
        ),
        (
            "clean_dimension_values",
            lambda df: clean_dimension_values(df, unit_df=unit_df, logger=logger),
            True,
        ),
        (
            "clean_range_with_to_and_hyphen",
            lambda df: clean_range_with_to_and_hyphen(
                df, unit_df=unit_df, logger=logger
            ),
            True,
        ),
    ]

    remain = df0
    for idx, (name, fn, needs_display) in enumerate(steps, start=1):
        logger.info(f"{idx}. ▶️ Entering `{name}` with {len(remain)} rows")
        passed, mod, new_remain = fn(remain)

        # If this step produces display_values, unpack them
        if needs_display and not passed.empty:
            disp_passed, disp_mod = build_display_values(passed, unit_df, logger)
            # merge any new mod‐rows from display building
            mod = pd.concat([mod, disp_mod], ignore_index=True)
            passed = disp_passed

        # save everything
        save_dfs(name, passed, mod, output_path, logger=logger)

        # log summary for this function
        logger.info(f"✔ `{name}` summary:")
        logger.info(f"    → passed   : {len(passed)} rows")
        logger.info(f"    → mod      : {len(mod)} rows")
        logger.info(f"    → remaining: {len(new_remain)} rows")
        logger.info("-" * 60)

        remain = new_remain

    # ——— Final remaining ——————————————————————————————————————————————————
    logger.info(f"🏁 Pipeline end: {len(remain)} rows unprocessed")
    if not remain.empty:
        remain["mod_reason"] = "Unprocessed by any cleaner"
        path = os.path.join(output_path, "mod", "final_remain.csv")
        remain.to_csv(path, index=False)
        logger.info(f"Saved final remaining to: {path}")

    logger.info("✅ Cleanup pipeline completed.")


