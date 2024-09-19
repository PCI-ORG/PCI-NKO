import os
import json
import optuna
import argparse
import evaluate
import random
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_from_disk
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import torch
from util import create_logs_folders, compute_metrics, save_best_model, load_best_hyperparams, LoggingCallback, read_logs, preprocess_function
from config import PROJ_FOLDER, HYPERPARAMETER_SEARCH_LOGS, FULL_TRAINING_LOGS
from model_config import MODEL_CHECKPOINT, NUM_LABELS, MAX_LENGTH, MAX_ATTEMPTS, N_TRIALS, INITIAL_TRAIN_START, model_name, scheme_name, model_path, get_tokenizer, get_device, model_init, get_optuna_settings
from datasets_preparation import prepare_datasets, save_datasets

device = get_device()
tokenizer = get_tokenizer()
task_evaluator = evaluate.evaluator("text-classification")

def get_previous_date_str(start_date_str, step_months=1, initial_train_start=INITIAL_TRAIN_START):
        current_start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        initial_train_start_date = datetime.strptime(initial_train_start, '%Y-%m-%d')
        
        previous_start_date = current_start_date - relativedelta(months=step_months)
        
        if previous_start_date < initial_train_start_date:
            return None
        
        previous_start_date_str = previous_start_date.strftime('%Y-%m-%d')
        return previous_start_date_str
        

def run_hyperparameter_search(start_date_str,
                              encoded_dataset,  
                              random_seed,
                              n_trials=N_TRIALS):
    create_logs_folders()
    # Calculate previous hyperparameters path
    previous_date_str = get_previous_date_str(start_date_str)
    initial_hyperparams = None # Alternative: load_best_hyperparams(previous_date_str)
    
    if not initial_hyperparams:
        print("Warning: No initial hyperparameters found, starting from default settings.")

    def objective(trial):
        hyperparams = get_optuna_settings(trial, initial_hyperparams=initial_hyperparams)
        output_dir = os.path.join(HYPERPARAMETER_SEARCH_LOGS, model_name, scheme_name, start_date_str)
        os.makedirs(output_dir, exist_ok=True)
        
        log_file = os.path.join(output_dir, f'hyperparameter_search_trial_{trial.number}.jsonl')
        logging_callback = LoggingCallback(log_file=log_file)
        
        search_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1", 
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=10,
            logging_strategy="epoch",
            save_total_limit=2,
            report_to=None,
            seed=random_seed,  # Use the provided random seed
            **hyperparams  
        )
        
        model = model_init().to(device)
        search_trainer = Trainer(
            model=model,
            args=search_args,
            train_dataset=encoded_dataset["train"].shard(index=1, num_shards=10),  # Use 1/10 of the training data for faster training
            eval_dataset=encoded_dataset["validate"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3), logging_callback]
        )
        
        search_trainer.train()
        eval_results = search_trainer.evaluate()
        
        return eval_results["eval_f1"]

    study = optuna.create_study(direction="maximize", study_name="Model Hyperparameter Optimization", sampler=optuna.samplers.TPESampler(seed=random_seed))
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Print the hyperparameter search logs summary
    for trial in study.trials:
        log_file = os.path.join(HYPERPARAMETER_SEARCH_LOGS, model_name, scheme_name, start_date_str, f'hyperparameter_search_trial_{trial.number}.jsonl')
        if os.path.exists(log_file):
            logs = read_logs(log_file)
            print(f"\nTrial {trial.number} summary:")
            if logs['evaluation']:
                last_eval = logs['evaluation'][-1]
                print(f"Final F1: {last_eval['eval_f1']:.4f} at step {last_eval['step']}")

    return study.best_trial

def run_full_training(start_date_str, encoded_dataset):
    best_f1 = 0
    best_model_dir = None
    best_hyperparams = None
    max_attempts = MAX_ATTEMPTS

    for attempt in range(max_attempts):
        print(f"\n\033[91mAttempt {attempt + 1}...\033[0m")
        print("\n\033[91mRunning hyperparameter search...\033[0m")
        
        random_seed = random.randint(1, 10000)
        print(f"Using random seed: {random_seed}")
        
        best_trial = run_hyperparameter_search(start_date_str, encoded_dataset, random_seed)
        attempt_hyperparams = best_trial.params
        print(f"Using hyperparameters: {attempt_hyperparams}")

        output_dir = os.path.join(HYPERPARAMETER_SEARCH_LOGS, start_date_str)
        os.makedirs(output_dir, exist_ok=True)
        
        log_file = os.path.join(output_dir, f'training_logs_attempt_{attempt+1}.jsonl')
        logging_callback = LoggingCallback(log_file=log_file)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            per_device_eval_batch_size=128,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1", 
            learning_rate=attempt_hyperparams['learning_rate'],
            per_device_train_batch_size=attempt_hyperparams['per_device_train_batch_size'],
            num_train_epochs=30,
            weight_decay=attempt_hyperparams['weight_decay'],
            warmup_steps=attempt_hyperparams['warmup_steps'],
            lr_scheduler_type="cosine_with_restarts",
            gradient_accumulation_steps=attempt_hyperparams['gradient_accumulation_steps'],
            max_grad_norm=attempt_hyperparams['max_grad_norm'],
            adam_epsilon=attempt_hyperparams['adam_epsilon'],
            push_to_hub=False,
            report_to=None,
            logging_dir=output_dir,
            save_total_limit=2,
            seed=random_seed  # Set the seed for training
        )

        model = model_init().to(device)
        trainer = Trainer(
            model=model,  
            args=training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["validate"],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3), logging_callback]
        )
        trainer.train() 
        test_result = trainer.evaluate(encoded_dataset["test"])
        eval_f1 = test_result['eval_f1']

        if eval_f1 > best_f1:
            best_f1 = eval_f1
            best_model_dir = output_dir
            best_hyperparams = attempt_hyperparams
            print(f"New best F1 score: {best_f1} achieved on attempt {attempt + 1}")

        # Print the training logs summary
        logs = read_logs(log_file)
        print("\nTraining Loss Summary:")
        for entry in logs['training'][-5:]:  # Print the last 5 loss values
            print(f"Step {entry['step']}: Loss = {entry['loss']:.4f}")
        print("\nEvaluation F1 Summary:")
        for entry in logs['evaluation'][-5:]:  # Print the last 5 F1 scores
            print(f"Step {entry['step']}: F1 = {entry['eval_f1']:.4f}")

    if best_model_dir:
        save_best_model(model, start_date_str, best_f1, best_hyperparams)
        print(f"\033[94mBest model and hyperparameters saved to disk with F1 score: {best_f1}.\033[0m")
    
    return best_hyperparams, best_f1

def run_prediction_evaluation(start_date_train, encoded_pred_dataset):
    best_model_path = os.path.join(model_path, start_date_train, "best_model")

    model = model_init(best_model_path).to(device)

    task_evaluator = evaluate.evaluator("text-classification")
    eval_results = task_evaluator.compute(
        model_or_pipeline=model,
        tokenizer=get_tokenizer(),
        data=encoded_pred_dataset,
        input_column="text",
        label_column="front_page",
        label_mapping={'LABEL_0': 0, 'LABEL_1': 1},
        metric='f1'
    )
    pred_f1_score = eval_results['f1']

    print(f"\n\033[91mEval results on prediction dataset: {pred_f1_score}\033[0m")

    result = {'f1': pred_f1_score}
    print(json.dumps(result))
    return pred_f1_score

def parse_args():
    parser = argparse.ArgumentParser(description="Run model training and evaluation.")
    parser.add_argument("start_date_train", type=str, help="Start date for the training dataset in YYYY-MM-DD format.")
    parser.add_argument("end_date_train", type=str, help="End date for the training dataset in YYYY-MM-DD format.")
    parser.add_argument("start_date_pred", type=str, help="Start date for the prediction dataset in YYYY-MM-DD format.")
    parser.add_argument("end_date_pred", type=str, help="End date for the prediction dataset in YYYY-MM-DD format.")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of trials for hyperparameter optimization.")
    parser.add_argument("--previous_hyperparams_path", type=str, default=None, help="Path to the JSON file containing the best hyperparameters from the last period.")
    return parser.parse_args()

def main(args):
    start_date_train = datetime.strptime(args.start_date_train, '%Y-%m-%d')
    end_date_train = datetime.strptime(args.end_date_train, '%Y-%m-%d')
    start_date_pred = datetime.strptime(args.start_date_pred, '%Y-%m-%d')
    end_date_pred = datetime.strptime(args.end_date_pred, '%Y-%m-%d')

    # Load the file to store results
    # current_date = datetime.now().strftime("%Y-%m-%d")
    all_res_file = os.path.join(PROJ_FOLDER, "results",  f"{scheme_name}_all_results.csv")
    all_res = pd.read_csv(all_res_file)

    data_dir = os.path.join(PROJ_FOLDER, "data", "temp", scheme_name)

    train_filename = f"train_{start_date_train.strftime('%Y-%m')}_{end_date_train.strftime('%Y-%m')}"
    pred_filename = f"pred_{start_date_pred.strftime('%Y-%m')}_{end_date_pred.strftime('%Y-%m')}"

    train_dataset_path = os.path.join(data_dir, train_filename)
    pred_dataset_path = os.path.join(data_dir, pred_filename)

    # Load encoded datasets
    encoded_train_dataset = load_from_disk(train_dataset_path).map(preprocess_function, batched=True)
    encoded_pred_dataset = load_from_disk(pred_dataset_path).map(preprocess_function, batched=True, remove_columns=["publication_date"])

    # Hyperparameter search and model training
    best_hyperparams, best_f1 = run_full_training(args.start_date_train, encoded_train_dataset)


    # Print the training results
    print(f"Training completed for the period starting {start_date_train.strftime('%Y-%m-%d')}.")
    print(f"Best F1: {best_f1}, Best Hyperparameters: {best_hyperparams}")

    
    # Evaluate the model on prediction dataset
    print("\nEvaluating prediction dataset...")
    prediction_f1 = run_prediction_evaluation(start_date_train.strftime('%Y-%m-%d'), encoded_pred_dataset)

    # Announce the evaluation results
    print(f"\n\033[93mModel training completed for:\nTraining dataset: {start_date_train.strftime('%Y-%m-%d')} to {end_date_train.strftime('%Y-%m-%d')}\nPrediction dataset: {start_date_pred.strftime('%Y-%m-%d')} to {end_date_pred.strftime('%Y-%m-%d')}\nPrediction F1 score: {prediction_f1:.4f}\033[0m")

    # Concatenate the results to the dataframe
    res = pd.DataFrame([[start_date_train.strftime('%Y-%m-%d'), end_date_train.strftime('%Y-%m-%d'), 
                         start_date_pred.strftime('%Y-%m-%d'), end_date_pred.strftime('%Y-%m-%d'), 
                         best_f1, prediction_f1]], 
                         columns=['Train Start', 'Train End', 'Pred Start', 'Pred End', 'Test F1', 'Pred F1'])
    
    all_res = pd.concat([all_res, res], ignore_index=True)
    all_res.to_csv(all_res_file, index=False)

    # Print the full training history
    for attempt in range(MAX_ATTEMPTS):
        log_file = os.path.join(HYPERPARAMETER_SEARCH_LOGS, args.start_date_train, f'training_logs_attempt_{attempt+1}.jsonl')
        if os.path.exists(log_file):
            print(f"\nFull training history for attempt {attempt+1}:")
            logs = read_logs(log_file)
            print("\nTraining Loss:")
            for entry in logs['training']:
                print(f"Step {entry['step']}: Loss = {entry['loss']:.4f}")
            print("\nEvaluation F1 Scores:")
            for entry in logs['evaluation']:
                print(f"Step {entry['step']}: F1 = {entry['eval_f1']:.4f}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
