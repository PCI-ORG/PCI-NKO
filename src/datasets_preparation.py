import os
import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from datasets import Dataset, DatasetDict, load_from_disk
import logging
import random
import pandas as pd
from config import PROJ_FOLDER, WHOLE_SETS_PATH
from model_config import INITIAL_TRAIN_YEARS, PREDICTION_PERIOD, STEP_MONTHS


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Function to parse date string and validate
def parse_date(date_str):
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        logging.error(f"Date format error with '{date_str}'. Expected format: YYYY-MM-DD.")
        raise

# Function to filter data by date
def safe_date_filter(example, start_date, end_date):
    date = example['publication_date']
    return isinstance(date, datetime) and start_date <= date <= end_date

# Slice the dataset according to date range
def slice_dataset(dataset, start_date, end_date):
    if not (isinstance(start_date, datetime) and isinstance(end_date, datetime)):
        raise ValueError("Start date and end date must be datetime objects")
    
    return {
        key: dataset[key].filter(lambda example: safe_date_filter(example, start_date, end_date))
        for key in ['train', 'validate', 'test']
    }

# Oversample the training dataset
def oversample(dataset):
    true_docs = [doc for doc in dataset if doc['front_page'] == 1]
    false_docs = [doc for doc in dataset if doc['front_page'] == 0]

    count_true = len(true_docs)
    count_false = len(false_docs)

    if count_true == 0:
        logging.warning("No 'True' instances found in the dataset. Returning original dataset.")
        return pd.DataFrame(false_docs)

    oversample_factor = count_false // count_true
    oversampled_true_docs = true_docs * oversample_factor
    difference = count_false - len(oversampled_true_docs)

    if difference > 0:
        oversampled_true_docs += random.sample(true_docs, difference)

    balanced_train_set = oversampled_true_docs + false_docs
    random.shuffle(balanced_train_set)

    return pd.DataFrame(balanced_train_set)

# Prepare training and prediction datasets
def prepare_datasets(start_date_train, end_date_train, start_date_pred, end_date_pred, dataset):
    train_data = slice_dataset(dataset, parse_date(start_date_train), parse_date(end_date_train))
    oversampled_train_data = oversample(train_data['train'])
    
    train_dataset = DatasetDict({
        'train': Dataset.from_pandas(oversampled_train_data),
        'validate': train_data['validate'],
        'test': train_data['test']
    })
    
    pred_data = pd.concat([subset.to_pandas() for _, subset in slice_dataset(dataset, parse_date(start_date_pred), parse_date(end_date_pred)).items()])
    return train_dataset, pred_data

# Save datasets to disk
def save_datasets(train_dataset, pred_dataset, start_date_train, end_date_train, start_date_pred, end_date_pred):
    current_date = datetime.now().strftime('%Y-%m-%d')
    save_dir = os.path.join(PROJ_FOLDER, "data", "temp", f'{INITIAL_TRAIN_YEARS}Y{PREDICTION_PERIOD}M{STEP_MONTHS}S')
    os.makedirs(save_dir, exist_ok=True)

    train_filename = f"train_{start_date_train.strftime('%Y-%m')}_{end_date_train.strftime('%Y-%m')}"
    pred_filename = f"pred_{start_date_pred.strftime('%Y-%m')}_{end_date_pred.strftime('%Y-%m')}"

    train_dataset.save_to_disk(os.path.join(save_dir, train_filename))
    Dataset.from_pandas(pred_dataset).save_to_disk(os.path.join(save_dir, pred_filename))

    logging.info(f"Datasets saved as {train_filename} and {pred_filename} in {save_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and save datasets for training and prediction using specified date ranges.")
    parser.add_argument("start_date_train", type=str, help="Start date for the training dataset in YYYY-MM-DD format.")
    parser.add_argument("end_date_train", type=str, help="End date for the training dataset in YYYY-MM-DD format.")
    parser.add_argument("start_date_pred", type=str, help="Start date for the prediction dataset in YYYY-MM-DD format.")
    parser.add_argument("end_date_pred", type=str, help="End date for the prediction dataset in YYYY-MM-DD format.")
    
    args = parser.parse_args()

    whole_datasets_path = WHOLE_SETS_PATH
    if not whole_datasets_path:
        raise ValueError("WHOLE_SETS_PATH is not set in config.py.")

    whole_datasets = load_from_disk(whole_datasets_path)

    # Prepare datasets based on the date ranges provided via args
    train_dataset, pred_dataset = prepare_datasets(
        args.start_date_train,
        args.end_date_train,
        args.start_date_pred,
        args.end_date_pred,
        whole_datasets
    )

    # Save the datasets
    save_datasets(
        train_dataset,
        pred_dataset,
        parse_date(args.start_date_train),
        parse_date(args.end_date_train),
        parse_date(args.start_date_pred),
        parse_date(args.end_date_pred)
    )

    logging.info("Datasets have been prepared and are ready for training and prediction.")
