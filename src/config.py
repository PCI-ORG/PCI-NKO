import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Project Folder
PROJ_FOLDER = os.getenv("PROJ_FOLDER")
LOGS_FOLDER = os.getenv("LOGS_FOLDER")
HYPERPARAMETER_SEARCH_LOGS = os.getenv("HYPERPARAMETER_SEARCH_LOGS")
FULL_TRAINING_LOGS = os.getenv("FULL_TRAINING_LOGS")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES") 

# Check if all required environment variables are set
required_vars = ["PROJ_FOLDER", "LOGS_FOLDER", "HYPERPARAMETER_SEARCH_LOGS", "FULL_TRAINING_LOGS"]
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Set CUDA_VISIBLE_DEVICES if available
if CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES