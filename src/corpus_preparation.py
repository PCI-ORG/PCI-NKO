import argparse
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
from pathlib import Path  # 添加这行
from datasets import Dataset, DatasetDict
from config import PROJ_FOLDER
from dotenv import set_key, load_dotenv

# Function to parse date string and validate
def parse_date(date_str):
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        print(f"Date format error with '{date_str}'. Expected format: YYYY-MM-DD.")
        return None  # Return None if the format is incorrect

# Function to get valid date from user input
def get_valid_date(prompt, default_date):
    """Keep prompting the user until a valid date is provided or return the default."""
    while True:
        date_str = input(f"{prompt} (default is {default_date}): ") or default_date
        date = parse_date(date_str)
        if date:
            return date_str  # Return valid date string when correct

# Function to read and preprocess data
def read_data(file_path):
    df = pd.read_csv(file_path)
    df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
    df.dropna(subset=['publication_date', 'text', 'front_page'], inplace=True)
    return df

# Function to get the date range for the dataset
def get_date_range(df):
    min_date = df['publication_date'].min().replace(day=1)
    max_date = df['publication_date'].max().replace(day=1) + pd.DateOffset(months=1)
    return min_date.strftime('%Y-%m-%d'), max_date.strftime('%Y-%m-%d')

# Function to select specific columns from the dataframe
def select_columns(df, columns_to_keep):
    return df[columns_to_keep]

# Function to perform stratified date split
def stratified_date_split(df, start_date, end_date, block_size=20, train_ratio=14, validate_ratio=3, test_ratio=3):
    assert train_ratio + validate_ratio + test_ratio == block_size
    train_frames, validate_frames, test_frames = [], [], []

    min_date = max(pd.to_datetime(start_date), df['publication_date'].min())
    max_date = min(pd.to_datetime(end_date), df['publication_date'].max())
    
    current_date = min_date
    while current_date <= max_date:
        end_date = current_date + pd.Timedelta(days=block_size)
        block = df[(df['publication_date'] >= current_date) & (df['publication_date'] < end_date)]
        days = block['publication_date'].dt.date.unique()
        np.random.shuffle(days)
        
        train_days = days[:train_ratio] if len(days) > train_ratio else days
        valid_days = days[train_ratio:train_ratio + validate_ratio] if len(days) > train_ratio + validate_ratio else days[len(train_days):]
        test_days = days[train_ratio + validate_ratio:train_ratio + validate_ratio + test_ratio] if len(days) > train_ratio + validate_ratio + test_ratio else days[len(train_days) + len(valid_days):]
        
        if len(train_days) > 0:
            train_frames.append(block[block['publication_date'].dt.date.isin(train_days)])
        if len(valid_days) > 0:
            validate_frames.append(block[block['publication_date'].dt.date.isin(valid_days)])
        if len(test_days) > 0:
            test_frames.append(block[block['publication_date'].dt.date.isin(test_days)])
        
        current_date = end_date
    
    train = pd.concat(train_frames) if train_frames else pd.DataFrame()
    validate = pd.concat(validate_frames) if validate_frames else pd.DataFrame()
    test = pd.concat(test_frames) if test_frames else pd.DataFrame()
    
    train.reset_index(drop=True, inplace=True)
    validate.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    
    return train, validate, test

# Function to save the datasets
def save_to_datasets(train, validate, test, save_path, dataset_name):
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train),
        'validate': Dataset.from_pandas(validate),
        'test': Dataset.from_pandas(test)
    })
    dataset_dict.save_to_disk(os.path.join(save_path, dataset_name))

# Function to write the path to config.py file
def add_dataset_path_to_config(dataset_path):
    config_file = Path(PROJ_FOLDER) / "src" / "config.py"
    dataset_path_str = str(Path(dataset_path)).replace("\\", "/")
    new_line = f"WHOLE_SETS_PATH = Path(r'{dataset_path_str}')\n"
    
    with config_file.open('r') as file:
        lines = file.readlines()
    
    whole_sets_path_exists = False
    for i, line in enumerate(lines):
        if line.strip().startswith("WHOLE_SETS_PATH") and not line.strip().startswith("#"):
            whole_sets_path_exists = True
            lines[i] = new_line
            break
    
    if not whole_sets_path_exists:
        lines.append(f"\n# Added by corpus_preparation.py\n{new_line}")
    
    with config_file.open('w') as file:
        file.writelines(lines)
    
    action = "Updated" if whole_sets_path_exists else "Added"
    print(f"{action} WHOLE_SETS_PATH in config.py to: {dataset_path_str}")
    
    print(f"WHOLE_SETS_PATH in config.py is set to: {dataset_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare datasets for analysis based on specified date range")
    parser.add_argument("--file_path", type=str, help="File path for the input CSV data file")
    parser.add_argument("--start_date", type=str, help="Start date for the dataset processing in YYYY-MM-DD format")
    parser.add_argument("--end_date", type=str, help="End date for the dataset processing in YYYY-MM-DD format")
    parser.add_argument("--save_path", type=str, default=f"{PROJ_FOLDER}/data/processed/", help="Directory to save the processed datasets")
    
    args = parser.parse_known_args()[0]

    if not args.file_path:
        args.file_path = input("Please enter the file path for the input CSV data file: ")

    df = read_data(args.file_path)
    df = select_columns(df, ['publication_date', 'text', 'front_page'])
    
    date_range = get_date_range(df)
    
    if not args.start_date:
        args.start_date = get_valid_date(f"Please enter the start date in YYYY-MM-DD format", date_range[0])
    
    if not args.end_date:
        args.end_date = get_valid_date(f"Please enter the end date in YYYY-MM-DD format", date_range[1])
    
    train, validate, test = stratified_date_split(df, args.start_date, args.end_date)
    dataset_name = f"whole_sets_{pd.to_datetime(args.start_date).strftime('%Y-%m')}_to_{pd.to_datetime(args.end_date).strftime('%Y-%m')}"
    dataset_path = os.path.join(args.save_path, dataset_name)

    save_to_datasets(train, validate, test, args.save_path, dataset_name)
    print("Datasets successfully saved.")

    # Update the .env file with the path to the datasets
    add_dataset_path_to_config(dataset_path)
