import yaml
import os
import json
import logging

def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)

    return content

def create_directory(dirs: list):
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def save_local_df(data, data_path):
    data.to_csv(data_path, index=False)
    print(f"Data is saved at {data_path}")    

def save_reports(report: dict, report_path: str, indentation=4):
    with open(report_path, "w") as f:
        json.dump(report, f, indent=indentation)
    print(f"reports are saved at {report_path}")

def get_timestamp(name):
    timestamp = time.asctime().replace(" ", "_").replace(":", "")

    uniquename = f"{name}_{timestamp}"
    return uniquename