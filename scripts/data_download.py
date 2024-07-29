import os
import pandas as pd
import openpyxl
"""
It downloads the data from the source and saves it locally
"""

def fetch_data():
    """
    Read the local Excel file
    """
    data_set = "./data/raw/global.xlsx"
    raw_data_dir = os.path.join(os.getcwd(), 'data', 'raw')
    
    # Ensure that the raw data directory exists
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)
    
    file_path = os.path.join(raw_data_dir, 'global.xlsx')
    
    # Check if the file exists
    if os.path.exists(file_path):
        # Read the Excel file
        data = pd.read_excel(file_path)
        print(data.head())
    else:
        print(f"File {file_path} does not exist.")

if __name__ == '__main__':
    """
    Download the data
    """
    fetch_data()