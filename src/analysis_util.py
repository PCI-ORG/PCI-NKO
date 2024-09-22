import os
import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import random
from datasets import load_from_disk
import pyarrow as pa
from model_config import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import torch
from tqdm import tqdm
import time
from docx import Document
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openrouter_key = os.getenv("OPENROUTER_API_KEY")

# Define the policy areas
ALL_POLICY_AREAS = [
    'Economic and Industrial Policy', 
    'Agriculture and Food Security',
    'Infrastructure and Public Works',
    'Public Health and Safety', 
    'Defense and National Security', 
    #'Nuclear Power',
    'Social Policy and Ideology',
    'Labor and Employment', 
    'Education and Human Capital', 
    'Foreign Relations and Diplomacy', 
    'Science and Technology', 
    'Miscellaneous'
]

# Define LLM models
# candidate models as of Aug 2024
or_model1 = "meta-llama/llama-3.1-8b-instruct"
or_model2 = "meta-llama/llama-3.1-70b-instruct"
or_model3 = "openai/gpt-4o-mini-2024-07-18"
or_model4 = "openai/gpt-4o-2024-08-06"
or_model5 = "anthropic/claude-3.5-sonnet"
or_model6 = "meta-llama/llama-3.1-405b-instruct"
or_model7 = "google/gemini-pro-1.5-exp"
or_model8 = "deepseek/deepseek-chat"

# Set models for translation and analysis tasks separately
translation_model = or_model3
backup_model = or_model4
analysis_model = or_model5

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_key,
)

#####################################################################

## 1. For plotting the results
def plot_results(df, event_dates, output_dir='../results/figures'):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    output_file_1 = os.path.join(output_dir, 'f1_scores_over_time.png')
    output_file_2 = os.path.join(output_dir, 'f1_diff_over_time.png')

    # Read in the results
    results = df

    # Get the columns for plotting
    expected_columns = ['Pred End', 'Test F1', 'Pred F1']
    plot_df = results[expected_columns].copy()

    # Convert the date to datetime
    plot_df['Pred End'] = pd.to_datetime(plot_df['Pred End'])

    # Extract the year and month and store in new columns in %Y-%m format
    plot_df['year_month'] = plot_df['Pred End'].dt.strftime('%Y-%m')

    # Plot the results
    plt.figure(figsize=(12, 6), facecolor='white')
    sns.set_style("white")  # 使用白色背景，不包含网格线

    ax = sns.lineplot(data=plot_df, x='Pred End', y='Test F1', label='Test F1', color='navy')
    sns.lineplot(data=plot_df, x='Pred End', y='Pred F1', label='Pred F1', color='crimson')

    # 移除网格线
    ax.grid(False)

    # Add vertical lines for important events and event descriptions
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    num_events = len(event_dates)
    event_positions = [y_min + i * y_range / (num_events + 1) for i in range(1, num_events + 1)]

    for (event_date, event_desc), event_position in zip(event_dates, event_positions):
        ax.axvline(event_date, color='gray', linestyle='--', alpha=0.5)
        ax.text(event_date, event_position, event_desc, rotation=0, ha='left', va='center', fontsize=8)

    plt.title('Test and Prediction F1 Over Time', fontsize=16)
    plt.xlabel('Prediction End Date', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.xticks(rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Set x-axis ticks to every two months
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Adjust the x-axis limits to show the desired time range
    ax.set_xlim([datetime(2022, 2, 1), datetime(2024, 4, 30)])

    plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
    plt.savefig(output_file_1, dpi=300, bbox_inches='tight', facecolor='white')  # Set facecolor to 'white'
    plt.show()

    # Second plot to show the difference between the test and prediction F1
    plt.figure(figsize=(12, 6), facecolor='white')
    sns.set_style("white")  # 使用白色背景，不包含网格线

    plot_df['F1 Diff'] = plot_df['Test F1'] - plot_df['Pred F1']
    ax = sns.lineplot(data=plot_df, x='Pred End', y='F1 Diff')

    # 移除网格线
    ax.grid(False)

    # Add vertical lines for important events and event descriptions
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    num_events = len(event_dates)
    event_positions = [y_min + i * y_range / (num_events + 1) for i in range(1, num_events + 1)]

    for (event_date, event_desc), event_position in zip(event_dates, event_positions):
        ax.axvline(event_date, color='gray', linestyle='--', alpha=0.5)
        ax.text(event_date, event_position, event_desc, rotation=0, ha='left', va='center', fontsize=8)

    plt.ylabel('Policy Change Index for North Korea', fontsize=14)
    plt.axhline(0, color='red', linestyle='--')  # Add a horizontal line at 0
    plt.xticks(rotation=45)

    # Set x-axis ticks to every two months
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Adjust the x-axis limits to show the desired time range
    ax.set_xlim([datetime(2022, 2, 1), datetime(2024, 4, 30)])

    plt.tight_layout()
    plt.savefig(output_file_2, dpi=300, bbox_inches='tight', facecolor='white')  # Set facecolor to 'white'
    plt.show()


########################################################### 
## 2. LLM analysis

# Generate dates
def generate_dates(date1, train_years=INITIAL_TRAIN_YEARS, pred_months=PREDICTION_PERIOD):
    train_start = pd.Timestamp(date1).normalize()
    train_end = (train_start + pd.DateOffset(years=train_years)).normalize()
    pred_start = train_end
    pred_end = (pred_start + pd.DateOffset(months=pred_months)).normalize()

    # Generate date strings
    train_start_YM = train_start.strftime('%Y-%m')
    train_end_YM = train_end.strftime('%Y-%m')
    pred_start_YM = pred_start.strftime('%Y-%m')
    pred_end_YM = pred_end.strftime('%Y-%m')
    return train_start, train_end, pred_start, pred_end, train_start_YM, train_end_YM, pred_start_YM, pred_end_YM

def get_previous_month(date_str):
    """
    Takes a date string in 'YYYY-MM' format and returns the previous month in the same format. 
    """
    # Parse the input string to a datetime object
    current_date = datetime.strptime(date_str, '%Y-%m')
    
    # Subtract one month
    previous_month = current_date - relativedelta(months=1)
    
    # Format the result back to 'YYYY-MM' string
    return previous_month.strftime('%Y-%m')

# Extract data by date range
def extract_data_by_date_range(df, start_date, end_date, random_seed=None):
    df['publication_date'] = pd.to_datetime(df['publication_date'])
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    mask = (df['publication_date'] >= start_date) & (df['publication_date'] < end_date)
    filtered_df = df.loc[mask]
    
    if random_seed is not None:
        np.random.seed(random_seed)
        
        filtered_df = filtered_df.sample(frac=1, random_state=random_seed)
        
    return filtered_df

# Load data by date range
def load_and_split_data(df, train_start, train_years=INITIAL_TRAIN_YEARS, pred_months=PREDICTION_PERIOD):
    df['publication_date'] = pd.to_datetime(df['publication_date']).dt.normalize()
    train_start, train_end, pred_start, pred_end = generate_dates(train_start, train_years, pred_months)[:4]
    train_df = df[(df['publication_date'] >= train_start) & (df['publication_date'] < train_end)].copy().reset_index(drop=True)
    pred_df = df[(df['publication_date'] >= pred_start) & (df['publication_date'] < pred_end)].copy().reset_index(drop=True)
    return train_df, pred_df

# Convert test set to pandas DataFrame
def test_set_to_dataframe(dataset_path):
    print("Loading dataset...")
    dataset = load_from_disk(dataset_path)
    print("Dataset loaded successfully.")
    
    print("Converting test set to pandas DataFrame...")
    test_df = dataset['test'].to_pandas()
    
    # Convert publication_date to datetime
    test_df['publication_date'] = pd.to_datetime(test_df['publication_date'], errors='coerce')
    
    print("Conversion completed.")
    return test_df

# The three functions below are for evaluating the model
def preprocess_batch(texts, tokenizer, max_length=256):
    return tokenizer(
        texts, 
        truncation=True, 
        padding=True, 
        max_length=max_length, 
        return_tensors="pt"
    )

def classify_texts(df, model, tokenizer, device, label_column='front_page', batch_size=32, max_length=256, threshold=0.5, invert_predictions=False):
    model.eval()
    all_predictions = []
    
    for i in tqdm(range(0, len(df), batch_size)):
        batch_texts = df['text'][i:i+batch_size].tolist()
        inputs = preprocess_batch(batch_texts, tokenizer, max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
        
        # Ensure predictions is two-dimensional
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        all_predictions.append(predictions)
    
    # Concatenate all batches of predictions
    all_predictions = np.vstack(all_predictions)
    
    # Apply threshold and possibly invert predictions
    if invert_predictions:
        df['result'] = (all_predictions[:, 0] < threshold).astype(int)
    else:
        df['result'] = (all_predictions[:, 0] >= threshold).astype(int)
    
    # Print some debugging information
    print(f"Shape of 'result' column: {df['result'].shape}")
    # print(f"First few values of 'result' column: {df['result'].head()}")
    print(f"Distribution of 'result' column: {df['result'].value_counts(normalize=True)}")
    print(f"Shape of '{label_column}' column: {df[label_column].shape}")
    # print(f"First few values of '{label_column}' column: {df[label_column].head()}")
    print(f"Distribution of '{label_column}' column: {df[label_column].value_counts(normalize=True)}")
    
    # Use numpy for element-wise operations
    df['prediction_type'] = np.select([
        (df[label_column] == 1) & (df['result'] == 1),
        (df[label_column] == 0) & (df['result'] == 0),
        (df[label_column] == 0) & (df['result'] == 1),
        (df[label_column] == 1) & (df['result'] == 0)
    ], ['TP', 'TN', 'FP', 'FN'], default='Unknown')
    
    return df

def calculate_metrics(df, label_column='front_page'):
    y_true = df[label_column]
    y_pred = df['result']
    
    # Ensure y_true and y_pred are one-dimensional arrays
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    
    accuracy = (y_true == y_pred).mean()
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

# Sample for FOR
def sample_for_FOR(train_df, pred_df, tn_fn_ratio=1, random_seed=23):

    if random_seed is not None:
        np.random.seed(random_seed)
        
    pred_fn = pred_df[pred_df['prediction_type'] == 'FN']
    pred_tn = pred_df[pred_df['prediction_type'] == 'TN']
    
    pred_tn_sampled = pred_tn.sample(n=min(len(pred_tn), len(pred_fn) * tn_fn_ratio), replace=False, random_state=random_seed)
    pred_sample = pd.concat([pred_fn, pred_tn_sampled])
    
    train_fn = train_df[train_df['prediction_type'] == 'FN']
    train_tn = train_df[train_df['prediction_type'] == 'TN']
    
    train_fn_sampled = train_fn.sample(n=min(len(train_fn), len(pred_fn)), replace=False, random_state=random_seed)
    train_tn_sampled = train_tn.sample(n=min(len(train_tn), len(train_fn_sampled) * tn_fn_ratio), replace=False, random_state=random_seed)
    train_sample = pd.concat([train_fn_sampled, train_tn_sampled])

    print("Train sample count：")
    print(train_sample['prediction_type'].value_counts())
    print("\nPred sample count：")
    print(pred_sample['prediction_type'].value_counts())
    
    return train_sample, pred_sample

# Translate text using LLM with interruption-proof
def RD_translate(text, primary_model=translation_model, backup_model=backup_model):
    max_retries = 3
    base_delay = 2
    
    def try_translation(model):
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a professional translator specializing in Korean to English translation, with expertise in North Korean official documents and news articles."},
                        {"role": "user", "content": f"Translate the following North Korean text from Korean to English. Maintain the original tone and style while ensuring clarity in English. Pay attention to specific political terms and expressions common in North Korean media: {text}"}
                    ],
                    max_tokens=1000,
                    temperature=0.5
                )
                return completion.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Error occurred with model {model}: {e}. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Failed after {max_retries} attempts with model {model}. Error: {e}")
                    return None
    
    # Try with primary model
    result = try_translation(primary_model)
    if result:
        return result
    
    print(f"Switching to backup model {backup_model}")
    
    # Try with backup model
    result = try_translation(backup_model)
    if result:
        return result
    
    return f"Error in translation: Failed with both models"

def generate_counts(sample_df, all_policies):
    fn_counts = sample_df[sample_df['prediction_type'] == 'FN']['Policy Area'].value_counts()
    tn_counts = sample_df[sample_df['prediction_type'] == 'TN']['Policy Area'].value_counts()
    
    counts = []
    
    for policy in all_policies:
        fn = fn_counts.get(policy, 0) 
        tn = tn_counts.get(policy, 0)
        total = fn + tn
        counts.append((fn, total))
    
    return counts


def calculate_false_omission_rate(df):
    grouped = df.groupby('Policy Area')
    false_omission_rates = grouped['prediction_type'].apply(lambda x: 
        sum(x == 'FN') / (sum(x == 'FN') + sum(x == 'TN'))
        if len(x) > 0 else 0 
    )
    false_omission_rates = false_omission_rates.reindex(ALL_POLICY_AREAS, fill_value=0)
    # Convert series to dataframe
    return false_omission_rates

# Note: Translation can take a long time (30 min for approx 500 rows).
# This function is to resume the translation process from interrupted point.
def translate_csv(input_file, output_file, batch_size=20):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' does not exist.")

    df_input = pd.read_csv(input_file)
    total_rows = len(df_input)
    print(f"Total rows in input file: {total_rows}")

    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        print(f"Resuming from existing output file: {output_file}")
        completed_translations = df['Translation_EN-US'].notna().sum()
    else:
        df = df_input.copy()
        df['Translation_EN-US'] = ""
        df['Translation_EN-US'] = df['Translation_EN-US'].astype(str)  # Ensure string type
        df.to_csv(output_file, index=False)
        print(f"Created new output file: {output_file}")
        completed_translations = 0

    print(f"Translations completed: {completed_translations}")

    with tqdm(total=total_rows, desc="Translating", unit="row", initial=completed_translations) as pbar:
        for index, row in df.iterrows():
            if pd.isna(row['Translation_EN-US']) or row['Translation_EN-US'] == "":
                korean_text = row['text']
                english_translation = RD_translate(korean_text)
                df.at[index, 'Translation_EN-US'] = english_translation
                pbar.update(1)
                if (pbar.n % batch_size == 0) or (pbar.n == total_rows):
                    df.to_csv(output_file, index=False)
                    print(f"Translated and saved up to row {pbar.n}")

    print("Translation process completed.")
    df.to_csv(output_file, index=False)
    print("Final save completed.")

# Summarizing and tagging texts
def clean_text(text):
    return re.sub(r'^\s*\n+', '', text.strip())

def process_entry(entry, analysis_model=analysis_model, max_retries=3):
    prompt = f"""Analyze the following news article from North Korea and provide:
News text: '{entry}'
    
1. A concise summary in 3-5 bullet points.
2. Tags in two categories:
   a) 1 relevant policy area selected from the following list:
   ['Economic and Industrial Policy', 'Agriculture and Food Security', 'Infrastructure and Public Works',
     'Public Health and Safety', 'Defense and National Security', 'Nuclear Power', 'Social Policy and Ideology',
       'Labor and Employment', 'Education and Human Capital', 'Foreign Relations and Diplomacy', 'Science and Technology', 'Miscellaneous']
   b) 3-5 specific keywords or concepts mentioned in the article
Use the following format:
Summary:
- 
- 
- 
Tags:
[Policy Area 1], [Keyword 1], [Keyword 2], [Keyword 3], [Keyword 4], [Keyword 5]
"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=analysis_model,
                messages=[
                    {"role": "system", "content": "You are an expert analyst on North Korean politics and policies. Your task is to summarize news articles and identify key policy areas. Provide the summary and tags directly without any introductory text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=350,
                n=1,
                temperature=0.2,
            )
            
            result = response.choices[0].message.content.strip()
            
            # Check if "Tags:" exists in the result
            if "Tags:" in result:
                summary, tags = result.split("Tags:")
                summary = clean_text(summary.replace("Summary:", ""))
                tags = clean_text(tags)
                tags = re.sub(r'\[|\]', '', tags)
                policy_area = tags.split(",")[0].strip()
                
                # If tags or policy area is "N/A", retry
                if tags == "N/A" or policy_area == "N/A":
                    print(f"Attempt {attempt + 1}: Tags or Policy Area is N/A. Retrying...")
                    continue
                
                return summary, tags, policy_area
            else:
                print(f"Attempt {attempt + 1}: No Tags found in the response. Retrying...")
                continue
        
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error occurred: {e}. Retrying in 5 seconds...")
            time.sleep(5)
    
    print(f"Failed after {max_retries} attempts. Returning error message.")
    return "Error in processing", "Error in processing", "Error in processing"

# Process the entire dataframe
def process_df(df):
    tqdm.pandas(desc="Processing entries")
    df['Summary'], df['Tags'], df['Policy Area'] = zip(*df['Translation_EN-US'].progress_apply(process_entry))
    return df

                
# Plotting FOR
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_false_omission_rates(train_for, pred_for, train_counts, pred_counts, train_date_range, pred_date_range, pred_start_date):
    plt.rcdefaults()
    sns.set_style("white", {'axes.facecolor': 'white'})
    
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
    
    x = np.arange(len(ALL_POLICY_AREAS))
    width = 0.35
    
    train_bars = ax.barh(x - width/2, train_for, width, label=train_date_range, color='#1f77b4', alpha=0.8)
    pred_bars = ax.barh(x + width/2, pred_for, width, label=pred_date_range, color='#ff7f0e', alpha=0.8)
    
    ax.set_xlabel('False Omission Rate', color='black')
    ax.set_title('False Omission Rate by Policy Area', fontsize=16, color='black')
    ax.set_yticks(x)
    ax.set_yticklabels(ALL_POLICY_AREAS, color='black')
    ax.legend(fontsize=12)
    
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    
    for text in ax.get_legend().get_texts():
        text.set_color('black')
    
    def autolabel(rects, counts):
        for rect, (fn, total) in zip(rects, counts):
            width = rect.get_width()
            # Add percentage label
            ax.annotate(f'{width:.0%}',
                        xy=(width, rect.get_y() + rect.get_height()*0.25),
                        xytext=(5, 0),  # 5 points horizontal offset
                        textcoords="offset points",
                        ha='left', va='bottom',
                        color='black')
            # Add fn / total label
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_y() + rect.get_height() / 2,
                    f'{fn} / {total}',
                    ha='center', va='center',
                    fontweight='bold', color='black',
                    fontsize=8)

    autolabel(train_bars, train_counts)
    autolabel(pred_bars, pred_counts)
    
    ax.grid(color='lightgray', linestyle='-', linewidth=0.5)
    
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    file_path = f'../results/analysis_results/{pred_start_date}_FOR_comparison.png'
    ensure_dir(file_path)
    plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    plt.close()

# Locate the policy areas that have significant changes in False Omission Rate for further analysis
def filter_policy_areas(for_changes, sample_counts, for_threshold=0.15, sample_threshold=10, additional_threshold=0.10):
    results = {
        'significant_change': [],
        'potential_change': [],
        'stable': [],
        'insufficient_data': []
    }

    for policy in for_changes.index:
        change = for_changes[policy]
        train_sample = sample_counts.loc[policy, 'Train']
        pred_sample = sample_counts.loc[policy, 'Prediction']
        total_sample = train_sample + pred_sample

        if total_sample >= sample_threshold:
            if change >= for_threshold:
                results['significant_change'].append((policy, change, total_sample))
            elif change >= additional_threshold:
                results['potential_change'].append((policy, change, total_sample))
            else:
                results['stable'].append((policy, change, total_sample))
        elif total_sample >= sample_threshold // 2:
            if change >= for_threshold:
                results['potential_change'].append((policy, change, total_sample))
            else:
                results['stable'].append((policy, change, total_sample))
        else:
            results['insufficient_data'].append((policy, change, total_sample))

    for key in results:
        results[key] = sorted(results[key], key=lambda x: abs(x[1]), reverse=True)

    return results

def analyze_policy_changes(train_sample, pred_sample, focused_policies, model=analysis_model, batch_size=None):
    def prepare_data(sample, policies):
        filtered_data = sample[sample['Policy Area'].isin(policies)]
        articles = []
        for _, row in filtered_data.iterrows():
            article = f"Summary: {row['Summary']}\nTags: {row['Tags']}\n"
            articles.append(article)
        return articles

    train_articles = prepare_data(train_sample, focused_policies)
    pred_articles = prepare_data(pred_sample, focused_policies)

    def analyze_full_batch(train_batch, pred_batch):
        prompt = f"""As an expert analyst on North Korean politics and policies, analyze these sets of article summaries and tags from two periods related to {', '.join(focused_policies)}:

        Period 1 (Training Sample):
        {'-' * 40}
        {"".join(train_batch)}
        {'-' * 40}

        Period 2 (Prediction Sample):
        {'-' * 40}
        {"".join(pred_batch)}
        {'-' * 40}

        Please provide a comprehensive analysis of the main themes, changes, and potential policy shifts observed. Focus on:
        1. Emerging or increasing themes in Period 2
        2. Decreasing or disappearing themes
        3. Any surprising elements or unexpected developments
        4. Implications for North Korea's priorities or challenges
        5. Consistent themes across both periods and their implications
        6. Potential future policy directions based on these trends

        Provide a structured analysis, using specific examples from the summaries and tags to support your observations and conclusions.
        """

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert analyst on North Korean politics and policies. Your task is to analyze sets of article summaries and tags from different periods and provide insights on potential policy changes and focus areas."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,
                    n=1,
                    temperature=0.3
                )
                return response.choices[0].message.content
            
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error occurred: {e}. Retrying in 5 seconds...")
                    sleep(5)
                else:
                    print(f"Failed after {max_retries} attempts. Error: {e}")
                    return f"An error occurred while analyzing the data: {str(e)}"

    if batch_size is None or (len(train_articles) <= batch_size and len(pred_articles) <= batch_size):
    
        return analyze_full_batch(train_articles, pred_articles)
    else:

        batch_analyses = []
        for i in range(0, max(len(train_articles), len(pred_articles)), batch_size):
            train_batch = train_articles[i:i+batch_size]
            pred_batch = pred_articles[i:i+batch_size]
            batch_analysis = analyze_full_batch(train_batch, pred_batch)
            batch_analyses.append(batch_analysis)

        final_prompt = f"""You have analyzed multiple batches of article summaries and tags from two periods related to North Korean policies in {', '.join(focused_policies)}. Based on these analyses, provide a comprehensive overview of the changes and trends observed. Please address:

        1. What are the most significant themes or topics that have become more prevalent in Period 2? What new themes have emerged, and which ones have decreased in importance?
        2. What might these changes suggest about potential policy shifts or new focus areas in North Korea? How do these align with known domestic and international developments?
        3. Are there any surprising elements in the Period 2 data that might indicate unexpected developments or policy changes? How do these differ from North Korea's traditional policy positions?
        4. How do these differences reflect changes in North Korea's priorities or challenges? What implications might these have for North Korea's domestic and foreign policy?
        5. What themes remain consistent across both periods? What might this suggest about North Korea's long-term strategies or core interests?
        6. Based on these trends, what predictions can you make about future policy directions in North Korea? How might these changes affect international engagement with North Korea?

        Please provide a structured analysis, synthesizing the information from all batches and using specific examples to support your observations and conclusions.

        Here are the summaries of each batch analysis:

        {'-' * 40}
        {"".join(batch_analyses)}
        {'-' * 40}
        """

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert analyst on North Korean politics and policies. Your task is to synthesize analyses of multiple batches of article summaries and tags to provide a comprehensive overview of policy changes and trends."},
                        {"role": "user", "content": final_prompt}
                    ],
                    max_tokens=2000,
                    n=1,
                    temperature=0.3
                )
                return response.choices[0].message.content
            
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error occurred: {e}. Retrying in 5 seconds...")
                    sleep(5)
                else:
                    print(f"Failed after {max_retries} attempts. Error: {e}")
                    return f"An error occurred while analyzing the data: {str(e)}"