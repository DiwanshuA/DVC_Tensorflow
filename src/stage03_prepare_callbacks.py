import os 
from src.utils.all_utils import read_yaml, create_directory
from src.utils.models import get_VGG16_model, prepare_model
import argparse
import pandas as pd
import shutil
from tqdm import tqdm
import logging
import io


logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "running_logs.log"), level=logging.INFO, format=logging_str, filemode='a')

def prepare_callbacks(config_path):
    config = read_yaml(config_path)
    #params = read_yaml(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    tensorboard_log_dir = os.path.join(artifacts_dir, artifacts["TENSORBOARD_ROOT_LOG_DIR"])

    checkpoint_dir = os.path.join(artifacts_dir, artifacts["CHECKPOINT_DIR"])

    callback_dir = os.path.join(artifacts_dir, artifacts["CALLBACKS_DIR"])

    create_directory([
    tensorboard_log_dir,
    checkpoint_dir,
    callback_dir
    ])


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    #args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info(">>>>>>>Stage Three started to prepare callbacks")
        prepare_callbacks(config_path=parsed_args.config)
        logging.info("Stage Three Completed! Callbacks prepared>>>>>>>>\n\n")
    except Exception as e:
        logging.exception(e)
        raise e