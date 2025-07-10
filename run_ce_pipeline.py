from modules.ce_new.pipeline import run_cleanup_pipeline


if __name__ == "__main__":
    input_data_to_clean = "/home/abhishek/projects/genie/extra/with_l3_file.csv"
    output_path = "/home/abhishek/projects/genie/extra/output"
    run_cleanup_pipeline(input_data_to_clean, output_path)
