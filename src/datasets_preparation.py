import os
import argparse
from datetime import datetime
from datasets import Dataset, DatasetDict, load_from_disk
import logging
import random
from config import PROJ_FOLDER
from model_config import INITIAL_TRAIN_YEARS, PREDICTION_PERIOD, STEP_MONTHS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        logging.error(f"Date format error with '{date_str}'. Expected format: YYYY-MM-DD.")
        raise

def safe_date_filter(example, start_date, end_date):
    """Safely filter examples based on publication date."""
    date = example['publication_date']
    return isinstance(date, datetime) and start_date <= date <= end_date

def slice_dataset(dataset, start_date, end_date):
    if not (isinstance(start_date, datetime) and isinstance(end_date, datetime)):
        raise ValueError("Start date and end date must be datetime objects")
    
    return {
        key: dataset[key].filter(lambda example: safe_date_filter(example, start_date, end_date))
        for key in ['train', 'validate', 'test']
    }

def oversample(dataset):
    """Oversample 'True' instances in the training dataset based on the 'front_page' column."""
    true_docs = dataset.filter(lambda x: x['front_page'] == 1)
    false_docs = dataset.filter(lambda x: x['front_page'] == 0)

    count_true = len(true_docs)
    count_false = len(false_docs)
    oversample_factor = count_false // count_true
    oversampled_true_docs = Dataset.concatenate([true_docs] * oversample_factor)
    difference = count_false - len(oversampled_true_docs)

    if difference > 0:
        oversampled_true_docs = Dataset.concatenate([
            oversampled_true_docs,
            true_docs.shuffle(seed=42).select(range(difference))
        ])

    balanced_train_set = Dataset.concatenate([oversampled_true_docs, false_docs])
    balanced_train_set = balanced_train_set.shuffle(seed=42)
    return balanced_train_set

def prepare_datasets(start_date_train, end_date_train, start_date_pred, end_date_pred, dataset):
    train_data = slice_dataset(dataset, parse_date(start_date_train), parse_date(end_date_train))
    oversampled_train_data = oversample(train_data['train'])
    
    train_dataset = DatasetDict({
        'train': oversampled_train_data,
        'validate': train_data['validate'],
        'test': train_data['test']
    })
    
    # For prediction data, convert to Dataset
    pred_dataset = Dataset.concatenate([
        slice_dataset(dataset, parse_date(start_date_pred), parse_date(end_date_pred))[key]
        for key in ['train', 'validate', 'test']
    ])

    # Print label distribution
    unique_front_pages = set(pred_dataset['front_page'])
    print("Prediction Dataset Label Distribution:")
    print(unique_front_pages)

    return train_dataset, pred_dataset

## Save to DatasetDict
def save_datasets(train_dataset, pred_dataset, start_date_train, end_date_train, start_date_pred, end_date_pred):
    save_dir = os.path.join(PROJ_FOLDER, "data", "temp", f'{INITIAL_TRAIN_YEARS}Y{PREDICTION_PERIOD}M{STEP_MONTHS}S')
    os.makedirs(save_dir, exist_ok=True)

    train_filename = f"train_{start_date_train.strftime('%Y-%m')}_{end_date_train.strftime('%Y-%m')}"
    pred_filename = f"pred_{start_date_pred.strftime('%Y-%m')}_{end_date_pred.strftime('%Y-%m')}"

    train_dataset.save_to_disk(os.path.join(save_dir, train_filename))
    pred_dataset.save_to_disk(os.path.join(save_dir, pred_filename))

    # Print unique front_page values
    unique_front_pages = set(pred_dataset['front_page'])
    print("Saving Prediction Dataset Label Distribution:")
    print(unique_front_pages)

    logging.info(f"Datasets saved as {train_filename} and {pred_filename} in {save_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and save datasets for training and prediction using specified date ranges.")
    parser.add_argument("start_date_train", type=str, help="Start date for the training dataset in YYYY-MM-DD format.")
    parser.add_argument("end_date_train", type=str, help="End date for the training dataset in YYYY-MM-DD format.")
    parser.add_argument("start_date_pred", type=str, help="Start date for the prediction dataset in YYYY-MM-DD format.")
    parser.add_argument("end_date_pred", type=str, help="End date for the prediction dataset in YYYY-MM-DD format.")
    
    args = parser.parse_args()

    # Assuming the training dataset directory is set in an environment variable
    whole_datasets = load_from_disk(os.path.join(PROJ_FOLDER, "data/processed/whole_sets_2018-01_to_2024-01"))

    train_dataset, pred_dataset = prepare_datasets(
        args.start_date_train,
        args.end_date_train,
        args.start_date_pred,
        args.end_date_pred,
        whole_datasets
    )

    save_datasets(
        train_dataset,
        pred_dataset,
        parse_date(args.start_date_train),
        parse_date(args.end_date_train),
        parse_date(args.start_date_pred),
        parse_date(args.end_date_pred)
    )

    logging.info("Datasets have been prepared and are ready for training and prediction.")
