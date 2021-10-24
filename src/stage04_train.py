import os 
from src.utils.all_utils import read_yaml, create_directory
from src.utils.models import load_full_model
from src.utils.callbacks import get_callbacks
from src.utils.data_management import train_valid_generator
import argparse
import pandas as pd
from tqdm import tqdm
import logging


logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "running_logs.log"), level=logging.INFO, format=logging_str, filemode='a')

def train_model(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    data_dir = artifacts["DATA_DIR"]

    train_model_dir = os.path.join(artifacts_dir, "TRAINED_MODEL_DIR")

    create_directory([train_model_dir])

    untrained_full_model_path = os.path.join(artifacts_dir,
    artifacts["BASE_MODEL_DIR"],
    artifacts["UPDATED_BASE_MODEL_NAME"]
    )

    model = load_full_model(untrained_full_model_path)

    callback_dir_path = os.path.join(artifacts_dir, artifacts["CALLBACKS_DIR"])
    callbacks = get_callbacks(callback_dir_path)

    train_generator, valid_generator = train_valid_generator(
        data_dir=data_dir,
        IMAGE_SIZE=tuple(params["IMAGE_SIZE"][:-1]),
        BATCH_SIZE=params["BATCH_SIZE"],
        do_data_augmentation=params["AUGMENTATION"]
    )


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info(">>>>>>>Stage Four started")
        train_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info("Stage Four Completed!>>>>>>>>\n\n")
    except Exception as e:
        logging.exception(e)
        raise e