REQUIRED_COLUMNS = {
    "base_df": ["int_pid", "l3_name", "attribute_name", "attribute_value"],
    "unit_df": ["units", "display", "type", "space", "separator", "range"],
}

INIT_COLUMNS = {
    "base_df": [
        "data_type",
        "key_value",
        "child_attribute",
        "polarity1",
        "value1",
        "unit1",
        "polarity2",
        "value2",
        "unit2",
        "display_value",
        "mod_reason",
    ]
}

ERROR_MESSAGES = {
    "invalid_df_name": "'{df_name}' is not a valid dataframe identifier.",
    "missing_required_columns": "Missing required columns: {missing_cols}",
}
