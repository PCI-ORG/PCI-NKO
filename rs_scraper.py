import os
import requests
import time
from datetime import datetime, timedelta
from tqdm import tqdm

def download_pdf(url, output_dir, formatted_date):
    response = requests.get(url)
    if response.status_code == 200:
        os.makedirs(output_dir, exist_ok=True)
        filename = f'rd_{formatted_date}.pdf'
        with open(os.path.join(output_dir, filename), 'wb') as file:
            file.write(response.content)
        return True
    else:
        print(f"Error: Failed to download PDF from {url}")
        return False

def get_formatted_date(date):
    return date.strftime('%Y-%m-%d')

def get_date_range(start_date, end_date):
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    return dates

def scrape_rodong_sinmun(start_date, end_date=None):
    output_dir = datetime.now().strftime('%Y-%m-%d')

    if end_date:
        dates = get_date_range(start_date, end_date)
    else:
        dates = [start_date]
    # Download the PDFs:
    download_count = 0
    for date in tqdm(dates, desc="Downloading PDFs"):
        path_year = date.strftime("%Y")
        path_month = date.strftime("%m")
        formatted_date = get_formatted_date(date)
        url = f'https://kcnawatch.org/wp-content/uploads/sites/5/{path_year}/{path_month}/rodong-{formatted_date}.pdf'
        if download_pdf(url, output_dir, formatted_date):
            download_count += 1
        time.sleep(0.5)
    if download_count == 0:
      print("\nNo issue has been downloaded.")
    else:
      if end_date is not None:
          print(f"\nDownloaded a total of {download_count} issues from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")
      else:
          print(f"\nDownloaded the issue of {start_date.strftime('%Y-%m-%d')}.")

if __name__ == "__main__":
    start_date_input = input("Please enter the start date (MM/DD/YYYY): ")
    start_date = datetime.strptime(start_date_input, '%m/%d/%Y')

    end_date_input = input("Please enter the end date (MM/DD/YYYY), leave it blank if not applicable: ")
    if end_date_input:
        end_date = datetime.strptime(end_date_input, '%m/%d/%Y')
    else:
        end_date = None

    scrape_rodong_sinmun(start_date, end_date)
