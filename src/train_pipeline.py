import subprocess
import os
import csv
from datetime import datetime
from dateutil.relativedelta import relativedelta
from config import PROJ_FOLDER
from model_config import  INITIAL_TRAIN_START, INITIAL_TRAIN_YEARS, FINAL_TRAIN_START, STEP_MONTHS, MAX_PRED_END, scheme_name

initial_train_start, initial_train_years, step_months, final_train_start = INITIAL_TRAIN_START, INITIAL_TRAIN_YEARS, STEP_MONTHS, FINAL_TRAIN_START

def generate_date_pairs(start_str, initial_train_years, step_months, final_train_start, max_pred_end=MAX_PRED_END):
    # Convert string dates to datetime objects
    current_train_start = datetime.strptime(start_str, "%Y-%m-%d")
    final_train_start = datetime.strptime(final_train_start, "%Y-%m-%d")
    max_pred_end = datetime.strptime(max_pred_end, "%Y-%m-%d")

    # Loop through dates, generating pairs until the final train start date is reached
    while current_train_start <= final_train_start:
        train_start = current_train_start.strftime('%Y-%m-%d')
        train_end = (current_train_start + relativedelta(years=initial_train_years)).strftime('%Y-%m-%d')
        pred_start = (current_train_start + relativedelta(years=initial_train_years, days=1)).strftime('%Y-%m-%d')
        pred_end_date = (current_train_start + relativedelta(years=initial_train_years, months=3))
        # Ensure pred_end does not exceed the maximum allowed date
        if pred_end_date > max_pred_end:
            pred_end_date = max_pred_end
        
        pred_end = pred_end_date.strftime('%Y-%m-%d')
        yield train_start, train_end, pred_start, pred_end
        current_train_start += relativedelta(months=step_months)

def main():
    # current_date = datetime.now().strftime("%Y-%m-%d")
    results_file = os.path.join(PROJ_FOLDER, "results",  f"{scheme_name}_all_results.csv")

    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Train Start', 'Train End', 'Pred Start', 'Pred End', 'Test F1', 'Pred F1'])

    for train_start, train_end, pred_start, pred_end in generate_date_pairs(INITIAL_TRAIN_START, INITIAL_TRAIN_YEARS, STEP_MONTHS, FINAL_TRAIN_START):
        print(f"Training dataset: {train_start} to {train_end}, Prediction dataset: {pred_start} to {pred_end}")
        
        # subprocess.run(['python', 'datasets_preparation.py', train_start, train_end, pred_start, pred_end], check=True)

        # # Print message to indicate that the datasets have been prepared in green
        # print("\033[92m" + f"Datasets have been prepared for training and prediction." + "\033[0m")

        # subprocess.run(['python', 'model.py', train_start, train_end, pred_start, pred_end], check=True)
        try:
   
            subprocess.run(['python', 'datasets_preparation.py', train_start, train_end, pred_start, pred_end], check=True)
            print("\033[92m" + "Datasets have been prepared for training and prediction." + "\033[0m")

            subprocess.run(['python', 'model.py', train_start, train_end, pred_start, pred_end], check=True)
    
        except subprocess.CalledProcessError as e:
            print("\033[91m" + f"Error occurred while running subprocess: {e}" + "\033[0m")

        except Exception as e:
            print("\033[91m" + f"An unexpected error occurred: {str(e)}" + "\033[0m")
    
    print("All results have been saved to DataFrame.")
        
if __name__ == "__main__":
    main()
