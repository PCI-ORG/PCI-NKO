from transformers import AlbertConfig, AlbertForSequenceClassification, AutoModelForSequenceClassification, RobertaConfig, TrainingArguments, BertTokenizerFast
from transformers import TFAutoModelForSequenceClassification
from config import PROJ_FOLDER
from optuna.trial import Trial
import torch
import sys
import os

# Model Checkpoint
MODEL_CHECKPOINT = "klue/roberta-base" # Alternative: "kykim/albert-kor-base"
INITIAL_TRAIN_START = "2018-01-01"
INITIAL_TRAIN_YEARS = 4 #year(s)
FINAL_TRAIN_START = "2019-11-01"
PREDICTION_PERIOD = 3 # months
STEP_MONTHS = 1 # month(s)
MAX_PRED_END = "2024-02-01" 

model_name = MODEL_CHECKPOINT.split("/")[-1]
scheme_name = f"{INITIAL_TRAIN_YEARS}Y{PREDICTION_PERIOD}M{STEP_MONTHS}S"
model_path = os.path.join(PROJ_FOLDER, "models", model_name, scheme_name)


NUM_LABELS = 2
MAX_LENGTH = 256

# Setup hyperparameter search
OPTUNA_HYPERPARAMETERS = {
    "batch_size": [8, 16],
    "num_train_epochs": [6, 10],
    "learning_rate": (1e-5, 1e-4),
    "weight_decay": (0, 0.3),
    "warmup_steps": (500, 2000),
    "gradient_accumulation_steps": (1, 4),
    "max_grad_norm": (0.5, 2.0),
    "adam_epsilon": (1e-8, 1e-6)
}

MAX_ATTEMPTS = 3
N_TRIALS = 10


def get_tokenizer():
    return BertTokenizerFast.from_pretrained(MODEL_CHECKPOINT)

def get_device():
    """Determine which device to use (MPS, CUDA, or CPU)."""
    print(f"System platform: {sys.platform}")
    
    if sys.platform == 'darwin':  # Check if the operating system is macOS
        print("Checking for Apple MPS...")
        if torch.backends.mps.is_available():
            print("MPS is available.")
            return torch.device("mps")
        else:
            print("MPS is NOT available.")
    
    print("Checking for CUDA...")
    if torch.cuda.is_available():
        cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        print(f"CUDA is available. Using GPU: {cuda_devices}")
        return torch.device("cuda")
    
    print("Neither MPS nor CUDA is available. Using CPU.")
    return torch.device("cpu")


def model_init(best_model_path=None):
    device = get_device()  # Get the appropriate device
    config = RobertaConfig.from_pretrained( # Or 'AlbertConfig.from_pretrained(' for Albert model  
        MODEL_CHECKPOINT,
        num_labels=NUM_LABELS,
        hidden_dropout_prob = 0.2,  # Dropout for hidden layers
        attention_probs_dropout_prob = 0.2  # Dropout for attention probabilities
    )
    if best_model_path:
        model = AutoModelForSequenceClassification.from_pretrained(
            best_model_path,
            config=config,
            local_files_only=True,  
            ignore_mismatched_sizes=True  
        ).to(device)
    else:
        # Load model from the default MODEL_CHECKPOINT
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, config=config).to(device)
    return model


def get_optuna_settings(trial, initial_hyperparams=None):
    hyperparams = {
        "per_device_train_batch_size": trial.suggest_categorical(
            name="per_device_train_batch_size",
            choices=OPTUNA_HYPERPARAMETERS["batch_size"],
        ),
        "num_train_epochs": trial.suggest_int(
            name="num_train_epochs",
            low=OPTUNA_HYPERPARAMETERS["num_train_epochs"][0],
            high=OPTUNA_HYPERPARAMETERS["num_train_epochs"][1],
        ),
        "learning_rate": trial.suggest_float(
            name="learning_rate",
            low=OPTUNA_HYPERPARAMETERS["learning_rate"][0],
            high=OPTUNA_HYPERPARAMETERS["learning_rate"][1],
            log=True,
        ),
        "weight_decay": trial.suggest_float(
            name="weight_decay",
            low=OPTUNA_HYPERPARAMETERS["weight_decay"][0],
            high=OPTUNA_HYPERPARAMETERS["weight_decay"][1],
        ),
        "warmup_steps": trial.suggest_int(
            name="warmup_steps",
            low=OPTUNA_HYPERPARAMETERS["warmup_steps"][0],
            high=OPTUNA_HYPERPARAMETERS["warmup_steps"][1],
        ),
        "gradient_accumulation_steps": trial.suggest_int(
            name="gradient_accumulation_steps",
            low=OPTUNA_HYPERPARAMETERS["gradient_accumulation_steps"][0],
            high=OPTUNA_HYPERPARAMETERS["gradient_accumulation_steps"][1],
        ),
        "max_grad_norm": trial.suggest_float(
            name="max_grad_norm",
            low=OPTUNA_HYPERPARAMETERS["max_grad_norm"][0],
            high=OPTUNA_HYPERPARAMETERS["max_grad_norm"][1],
        ),
        "adam_epsilon": trial.suggest_float(
            name="adam_epsilon",
            low=OPTUNA_HYPERPARAMETERS["adam_epsilon"][0],
            high=OPTUNA_HYPERPARAMETERS["adam_epsilon"][1],
            log=True,
        )
    }
    # Option to inject initial_hyperparams if they are provided
    if initial_hyperparams:
        for key, value in initial_hyperparams.items():
            if key in hyperparams and isinstance(hyperparams[key], (int, float)):
                hyperparams[key] = value  # Update with initial hyperparam value if needed

    return hyperparams

device = get_device()
print(f"Selected device: {device}")

