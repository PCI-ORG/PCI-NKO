import os
import json
import numpy as np
from datasets import load_from_disk
import evaluate as ev
from config import PROJ_FOLDER, LOGS_FOLDER, HYPERPARAMETER_SEARCH_LOGS, FULL_TRAINING_LOGS
from model_config import model_name, scheme_name, model_init, get_tokenizer


def create_logs_folders():
    os.makedirs(LOGS_FOLDER, exist_ok=True)
    os.makedirs(HYPERPARAMETER_SEARCH_LOGS, exist_ok=True)
    os.makedirs(FULL_TRAINING_LOGS, exist_ok=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return ev.load('f1').compute(predictions=predictions, references=labels)

# Load the data
def load_datasets(train_path, pred_path):
    return load_from_disk(train_path), load_from_disk(pred_path)

# Tokenize the data
def preprocess_function(examples):
    tokenizer = get_tokenizer()
    tokenized_examples = tokenizer(examples["text"], truncation=True, padding=True, max_length=256)
    tokenized_examples["labels"] = examples["front_page"]
    return tokenized_examples

# Save and load hyperparameters
def save_hp(hyperparams, file_path):
    with open(file_path, 'w') as f:
        json.dump(hyperparams, f)

def load_hp(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# Save and load the best model and hyperparameters    
def save_best_model(model, start_date_str, best_f1, best_hyperparams):
    best_model_path = os.path.join(PROJ_FOLDER, "models", model_name, scheme_name, start_date_str, "best_model")
    os.makedirs(best_model_path, exist_ok=True)
    model.save_pretrained(best_model_path)
    print(f"\033[94mBest model saved to {best_model_path} with F1 score: {best_f1}.\033[0m")
    
    # Save the best hyperparameters for the time period
    hyperparams_folder = os.path.join(PROJ_FOLDER, 'models', model_name, scheme_name, start_date_str,'hyperparameters')
    os.makedirs(hyperparams_folder, exist_ok=True)
    hyperparams_path = os.path.join(hyperparams_folder, f"{start_date_str}_best_hyperparams.json")
    save_hp(best_hyperparams, hyperparams_path)


def load_best_hyperparams(start_date_str):
    hyperparams_folder = os.path.join(PROJ_FOLDER, "models", model_name, scheme_name, start_date_str, "hyperparameters")
    hyperparams_path = os.path.join(hyperparams_folder, f"{start_date_str}_best_hyperparams.json")
    return load_hp(hyperparams_path)