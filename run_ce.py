from modules.ce_new.ce_new import run_cleanup_pipeline


if __name__ == "__main__":
    input_data_to_clean = "/home/abhishek/projects/genie/extra/cleaning_test_3.xlsx"
    output_path = "/home/abhishek/projects/genie/extra/output"
    run_cleanup_pipeline(input_data_to_clean, output_path)
