import argparse
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
from datasets import Dataset, DatasetDict
from config import PROJ_FOLDER

def read_data(file_path):
    df = pd.read_csv(file_path)
    df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
    df.dropna(subset=['publication_date', 'text', 'front_page'], inplace=True)
    return df

def get_date_range(df):
    min_date = df['publication_date'].min().strftime('%Y-%m-%d')
    max_date = df['publication_date'].max().strftime('%Y-%m-%d')
    return min_date, max_date

def select_columns(df, columns_to_keep):
    return df[columns_to_keep]

def stratified_date_split(df, start_date, end_date, block_size=20, train_ratio=14, validate_ratio=3, test_ratio=3):
    # Ensure the ratios add up to the block size
    assert train_ratio + validate_ratio + test_ratio == block_size
    
    # Initialize lists to store the dataframes
    train_frames = []
    validate_frames = []
    test_frames = []
    
    # Get the minimum and maximum dates
    min_date = max(pd.to_datetime(start_date), df['publication_date'].min())
    max_date = min(pd.to_datetime(end_date), df['publication_date'].max())
    
    current_date = min_date
    while current_date <= max_date:
        end_date = current_date + pd.Timedelta(days=block_size)
        
        # Select the block of data
        block = df[(df['publication_date'] >= current_date) & (df['publication_date'] < end_date)]
        
        # Get the unique dates in the block
        days = block['publication_date'].dt.date.unique()
        
        # Shuffle the dates
        np.random.shuffle(days)
        
        # Split the dates into train, validate, and test sets
        train_days = days[:train_ratio] if len(days) > train_ratio else days
        valid_days = days[train_ratio:train_ratio + validate_ratio] if len(days) > train_ratio + validate_ratio else days[len(train_days):]
        test_days = days[train_ratio + validate_ratio:train_ratio + validate_ratio + test_ratio] if len(days) > train_ratio + validate_ratio + test_ratio else days[len(train_days) + len(valid_days):]
        
        # Append the dataframes to the respective lists
        if len(train_days) > 0:
            train_frames.append(block[block['publication_date'].dt.date.isin(train_days)])
        if len(valid_days) > 0:
            validate_frames.append(block[block['publication_date'].dt.date.isin(valid_days)])
        if len(test_days) > 0:
            test_frames.append(block[block['publication_date'].dt.date.isin(test_days)])
        
        # Update the current date
        current_date = end_date
    
    # Concatenate the dataframes
    train = pd.concat(train_frames) if train_frames else pd.DataFrame()
    validate = pd.concat(validate_frames) if validate_frames else pd.DataFrame()
    test = pd.concat(test_frames) if test_frames else pd.DataFrame()
    
    # Reset the index
    train.reset_index(drop=True, inplace=True)
    validate.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    
    return train, validate, test


def save_to_datasets(train, validate, test, save_path, dataset_name):
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train),
        'validate': Dataset.from_pandas(validate),
        'test': Dataset.from_pandas(test)
    })
    dataset_dict.save_to_disk(os.path.join(save_path, dataset_name))  # Save the whole DatasetDict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare datasets for analysis based on specified date range")
    parser.add_argument("file_path", type=str, help="File path for the input CSV data file")
    parser.add_argument("--start_date", type=str, help="Start date for the dataset processing in YYYY-MM-DD format")
    parser.add_argument("--end_date", type=str, help="End date for the dataset processing in YYYY-MM-DD format")
    parser.add_argument("--save_path", type=str, default=f"{PROJ_FOLDER}/data/processed/", help="Directory to save the processed datasets")
    args = parser.parse_known_args()[0]
    df = read_data(args.file_path)
    df = select_columns(df, ['publication_date', 'text', 'front_page'])
    if not args.start_date or not args.end_date:
        args.start_date, args.end_date = get_date_range(df)
    train, validate, test = stratified_date_split(df, args.start_date, args.end_date)
    dataset_name = f"whole_sets_{pd.to_datetime(args.start_date).strftime('%Y-%m')}_to_{pd.to_datetime(args.end_date).strftime('%Y-%m')}"
    save_to_datasets(train, validate, test, args.save_path, dataset_name)
    print("Datasets successfully saved.")
