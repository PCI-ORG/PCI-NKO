import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Project Folders
PROJ_FOLDER = Path('/your/project/folder') # Set this to your project folder
LOGS_FOLDER = PROJ_FOLDER / "logs"
HYPERPARAMETER_SEARCH_LOGS = LOGS_FOLDER / "hyperparameter_search_logs"
FULL_TRAINING_LOGS = LOGS_FOLDER / "full_training_logs"

# Ensure directories exist
for folder in [LOGS_FOLDER, HYPERPARAMETER_SEARCH_LOGS, FULL_TRAINING_LOGS]:
    folder.mkdir(parents=True, exist_ok=True)

# CUDA configuration
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES")

# Set CUDA_VISIBLE_DEVICES if available
if CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# This line will be added by corpus_preparation.py
# WHOLE_SETS_PATH = Path('...')