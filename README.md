_Last updated on September 21, 2024._

Website: [policychangeindex.org/overview-PCI-NKO.html](https://policychangeindex.org/overview-PCI-NKO.html)

Authors: [Zhiqiang Ji](https://www.linkedin.com/in/zhiqiangji117/) and [Weifeng Zhong](https://www.weifengzhong.com)

---------------------------------------------

## Introduction

The Policy Change Index for North Korea (PCI-NKO) is an innovative project that aims to predict North Korea's policy changes by analyzing its state propaganda. The PCI-NKO utilizes deep learning and large language models (LLMs) to detect and interpret changes in how the Rodong Sinmun, North Korea's official newspaper, prioritizes different policies. The method involves the following steps:

1. Collect and label Rodong Sinmun articles, including essential metadata such as publication date, title, content, and page number.
2. Train a deep learning model to predict front-page articles for every four-year period.
3. Deploy the model to assess editorial changes in the subsequent three months.
4. Define the PCI-NKO as the difference in the algorithm's performance between the training and deployment periods.
5. Use large language models to interpret anomalous articles detected in the three-month window.
6. Repeat the analysis monthly, resulting in a monthly PCI-NKO from April 2022 to February 2024.

This repository provides the code to implement steps 2-6 of the PCI-NKO workflow. Due to copyright considerations, we do not provide the training data. However, the same workflow can be applied to text classification tasks with binary labels and temporal information, such as publication dates. Interested researchers can use their own data for replication.

## Quick Start

For those familiar with Python projects, here's a quick guide to get started:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/PCI-NKO.git
   cd PCI-NKO
   ```

2. Set up the configuration:
   - Copy `.env.sample` to `.env` and edit it with your API keys
   - Modify `src/config.py` to set the project's root directory

3. Create and activate the Conda environment:
   ```
   conda env create -f environment.yml
   conda activate PCI-NKO
   ```
4. Install PyTorch and additional packages:
   - Visit [PyTorch official website](https://pytorch.org/get-started/locally/) to get the correct installation command for your system
   - Run the PyTorch installation command, for example:
     ```
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```
   - Install additional required packages:
     ```
     pip install transformers accelerate datasets
     ```
4. Prepare the corpus:
   ```
   cd src
   python corpus_preparation.py
   ```

5. Run the training pipeline:
   ```
   python train_pipeline.py
   ```

6. Analyze the results:
   - Open and run `analysis.ipynb` in Jupyter Lab
   - Follow the instructions in the notebook for detailed analysis

For more detailed instructions, please refer to the full documentation below.

## Setting Up the Environment

### System and Hardware Requirements
Model training requires substantial VRAM. The scripts have been successfully tested on the following systems:

- Apple M3 Max with 40-core GPU and 64GB RAM
- AWS EC2 instance with NVIDIA V100 (16GB VRAM)

If your system has less than 16GB VRAM, you may need to adjust the training parameters. Conversely, with more powerful hardware, you can increase the complexity of computations or parallel batch size. Training parameters can be modified in /src/model_config.py.

**Note:** The training process requires significant disk space, primarily for generated logs. Depending on the size of your source data and training settings, the required disk space may vary considerably. Please allocate at least 15GB of free disk space or adjust the logging configuration according to your training needs.

### Environment Setup

For Windows or macOS systems, the quickest setup method is to use [Anaconda](https://www.anaconda.com/download/success) to create a new environment, then manually install PyTorch and other packages like Transformers using pip. This approach ensures smooth operation across different systems and CUDA hardware versions.
Follow these steps to set up your environment:
1. Clone this repository:
    ```bash
    git clone https://github.com/PCI-ORG/PCI-NKO.git
    cd PCI-NKO
    ```
2. Set up the API key:
Copy `.env.sample` to `.env` and edit it to set your OpenRouter API Key or OpenAI API Key.
**Note:** [OpenRouter](https://openrouter.ai/) provides a platform to easily try various open-source and closed-source large language models. Registration is recommended. If you want to use a different large language model, OpenRouter is compatible with the [OpenAI python library](https://github.com/openai/openai-python), minimizing necessary code changes. If using a non-OpenAI model (e.g., [Anthropic](https://www.anthropic.com/api)), you'll need to install new packages and modify the API call in `analysis_util.py`.

3. Modify `config.py` to set the project's root directory.

4. Create a new Conda environment using the environment.yml file:
    ```bash
    conda env create -f environment.yml
    ```

5. Activate the environment:
    ```bash
    conda activate PCI-NKO
    ```     
6. Install PyTorch (visit [PyTorch official website](https://pytorch.org/get-started/locally/) to get the correct installation command for your system)

7. Install additional packages compatible with PyTorch:
    ```bash
    pip install transformers accelerate datasets
    ```

8. Test the hardware configuration:
    ```bash
    cd src
    python test_device.py
    ```

This will confirm whether you're using CUDA, Apple's Metal Performance Shaders (MPS), or CPU. CPU is strongly discouraged for model training. If your GPU is not recognized, please check your PyTorch installation.

- **Note:** If CUDA devices are available, uncomment the last line in the .env file.


For those who prefer using `pyenv` or Python's built-in `venv` for creating virtual environments, we also provide a `requirements.txt` file. You can use this as an alternative to the Conda setup (steps 1-5 above). After creating your virtual environment with [pyenv](https://github.com/pyenv/pyenv),  [pyenv-win](https://github.com/pyenv-win/pyenv-win), or [venv](https://docs.python.org/3/library/venv.html), you can install the required packages using:

```bash
pip install -r requirements.txt
``` 
After this, proceed with steps 6-8 to complete your setup.

By following these steps, you should have a working environment ready for running the project scripts.


## Data Preparation and Model Training

### Data Preparation

We assume the raw data is a CSV file where each item represents an article with its publication date, content, and whether it appeared on the front page (represented by 1 or 0).

1. Create a `/data` folder in the root directory and place the raw data file (e.g., `demo_set.csv`) inside it.
2. Navigate to the `/src` directory and run:  
    ```bash
    python corpus_preparation.py
    ```
This script converts the data into the Hugging Face datasets format, which is more suitable for model training. You'll need to input:
- The path to the raw data file (e.g., `../data/demo_set.csv`)
- The start date of the training window
- The end date of the prediction window
You can choose not to use all of your text data for a project. For instance, if the source file contains articles from 2018-01-01 to 2024-02-01, you have the option to study a specific period, such as 2022-01-01 to 2024-02-01. This flexibility enables focused analysis on particular time periods, which is especially useful when working with datasets that span long periods (e.g., [PCI-China](https://policychangeindex.org/overview-PCI-China.html) covers more than 70 years of data).

The script will generate a directory starting with "whole_sets_" in the /data/processed directory, containing three subdirectories: `train`, `validate`, and `test`. Articles are divided into 20-day blocks based on their publication date, and within each block, randomly assigned to train, validate, and test subsets in a 14:3:3 ratio. This division schema can be modified in `corpus_preparation.py`.

Due to the imbalance between front-page and non-front-page articles (approximately 1:11), oversampling is performed on the `train` subset before training. The relevant functions are in the `datasets_preparation.py` script.

**Note:** This `corpus_preparation.py` script typically needs to be run only once at the beginning of a new project, as it doesn't affect the subsequent workflow.

### Model Training

Model training settings are stored in the `model_config.py` script. Key settings include:

1. `MODEL_CHECKPOINT`: The default is "klue/roberta-base", a pre-trained model for Korean that performs well in text classification tasks. For platforms with limited computational resources, "kykim/albert-kor-base" is an alternative option. Note that the performance of these models might be affected by the linguistic differences between South and North Korean texts.
2. Default training schema:
    ```
    INITIAL_TRAIN_START = "2018-01-01"
    INITIAL_TRAIN_YEARS = 4 # year(s)
    FINAL_TRAIN_START = "2019-11-01"
    PREDICTION_PERIOD = 3 # months
    STEP_MONTHS = 1 # month(s)
    MAX_PRED_END = "2024-02-01" 
    ```
As mentioned above, we used 4 years of data to train the model and predict the next 3 months as one episode. Then we moved forward by 1 month to start the next episode. Please feel free to modify the parameters to fit your needs. 
**Note:** The schema is also used to identify the results in the folder. For example, the combination above will result in a folder named "4Y3M1S" in the "models" directory.

3. Hyperparameter optimization: We use Optuna to find the best model parameters. The process involves:
    - Extracting 10% of the training data for Optuna search
    - Using the resulting hyperparameters as a starting point to train the model with the entire train set
    - Repeating this process 3 times (3 attempts) and keeping the best result as the final model
The `MAX_ATTEMPTS` parameter significantly affects the training time. `N_TRIALS` also influences training time but is moderated by an early stopping mechanism.

After configuring these settings, you're ready to proceed with model training.

Navigate to the /src directory and run:
```bash
python train_pipeline.py
```

>**IMPORTANT:** 
>- Depending on your data, training method, and hardware environment, the training process may last for **several hours or even days**. It is strongly recommended to run this in an environment where it won't be interrupted.
>- The training process generates a large number of log files. Keep a close eye on the disk space used by the logs folder to prevent the disk from filling up and causing the training to halt unexpectedly.

### Training Results

Once the training is complete:
1. The `/results` folder will contain a summary of each episode's model performance, including F1 score on the test set for the training window (previous four years) and F1 score for the prediction window (subsequent 3 months).
2. The best model for each episode will be saved in `/models/roberta-base/{your scheme}/`.

## Result Analysis

We use the Jupyter notebook `analysis.ipynb` for result analysis, providing greater flexibility. To maintain readability, most of the required functions are stored in analysis_util.py. The overall analysis process includes the following steps:

1. **Identify Abnormal Episodes**： Visualize the model training results and the changes in the Policy Change Index (PCI) during the prediction window. Identify episodes with notable abnormalities.
2. **Data Extraction and LLM Processing**: Extract data for the identified abnormal episodes. Use Large Language Models (LLMs) to:
    - Translate articles into English
    - Generate English summaries
    - Tag articles, specifying the policy area they belong to
3. **In-depth Episode Analysis**: For each noteworthy episode, compare the False Omission Rate (FOR) differences between various policy areas in the training window and the prediction window. This comparison helps identify focused policy areas. 
4. **Focused Policy Area Analysis**: After identifying noteworthy policy areas, we zoom in further, use LLMs to analyze the content of sample articles in these areas, and generate a brief analysis report.

For each episode, the LLM analysis results, including a FOR comparison chart and a report in docx format, are saved in the `/results/analysis_results/` directory.

**Remember** that the LLM-generated analysis is not a substitute for domain knowledge and qualitative research. It is intended only as supplementary material for further investigation. It alone is not sufficient to draw any conclusions about North Korean policy changes. However, it may help researchers identify trends worth closer attention.

## Citing the PCI-NKO

Please cite the source of the latest PCI-NKO by the website: https://policychangeindex.org.

Please stay tune for our upcoming research paper on the subject.