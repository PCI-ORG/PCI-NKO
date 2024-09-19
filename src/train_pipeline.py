import os
import sys
import subprocess
import csv
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from colorlog import ColoredFormatter
from config import PROJ_FOLDER
from model_config import  INITIAL_TRAIN_START, INITIAL_TRAIN_YEARS, FINAL_TRAIN_START, STEP_MONTHS, MAX_PRED_END, scheme_name

def setup_logger():
    formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

setup_logger()


def generate_date_pairs(start_str, initial_train_years, step_months, final_train_start, max_pred_end=MAX_PRED_END):
    """Generate train and prediction date pairs based on the input parameters."""
    # Convert string dates to datetime objects
    current_train_start = datetime.strptime(start_str, "%Y-%m-%d")
    final_train_start = datetime.strptime(final_train_start, "%Y-%m-%d")
    max_pred_end = datetime.strptime(max_pred_end, "%Y-%m-%d")

    # Loop through dates, generating pairs until the final train start date is reached
    while current_train_start <= final_train_start:
        train_start = current_train_start.strftime('%Y-%m-%d')
        # Set train_end to one year after train_start minus one day
        train_end = (current_train_start + relativedelta(years=initial_train_years) - timedelta(days=1)).strftime('%Y-%m-%d')
        pred_start = (current_train_start + relativedelta(years=initial_train_years)).strftime('%Y-%m-%d')
        pred_end_date = (current_train_start + relativedelta(years=initial_train_years, months=step_months))
        
        # Ensure pred_end does not exceed the maximum allowed date
        if pred_end_date > max_pred_end:
            pred_end_date = max_pred_end
        
        pred_end = pred_end_date.strftime('%Y-%m-%d')
        # Yield the date pair for training and prediction
        yield train_start, train_end, pred_start, pred_end
        # Move to the next step
        current_train_start += relativedelta(months=step_months)

def main():
    # current_date = datetime.now().strftime("%Y-%m-%d")
    results_dir = Path(PROJ_FOLDER) / "results"
    results_dir.mkdir(parents=True, exist_ok=True) 
    results_file = results_dir / f"{scheme_name}_all_results.csv"

    if not results_file.exists():
        with results_file.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Train Start', 'Train End', 'Pred Start', 'Pred End', 'Test F1', 'Pred F1'])

    for train_start, train_end, pred_start, pred_end in generate_date_pairs(INITIAL_TRAIN_START, INITIAL_TRAIN_YEARS, STEP_MONTHS, FINAL_TRAIN_START):
        print(f"Training dataset: {train_start} to {train_end}, Prediction dataset: {pred_start} to {pred_end}")
        
        try:
             subprocess.run([sys.executable, Path('datasets_preparation.py'), train_start, train_end, pred_start, pred_end], check=True)
            print("\033[92m" + "Datasets have been prepared for training and prediction." + "\033[0m")

            subprocess.run([sys.executable, Path('model.py'), train_start, train_end, pred_start, pred_end], check=True)
    
        except subprocess.CalledProcessError as e:
            print("\033[91m" + f"Error occurred while running subprocess: {e}" + "\033[0m")

        except Exception as e:
            print("\033[91m" + f"An unexpected error occurred: {str(e)}" + "\033[0m")
    
    print("All results have been saved to results/. ")
        
if __name__ == "__main__":
    main()
