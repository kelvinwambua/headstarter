import pandas as pd
import os
import openpyxl

"""
Preprocess the raw data and save the processed data in the processed data directory.
"""

def preprocess_data():
    """
    Preprocess the raw data and save the processed data in the processed data directory.
    """
    raw_data_dir = os.path.join(os.getcwd(), 'data', 'raw')
    processed_data_dir = os.path.join(os.getcwd(), 'data', 'processed')
    
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    
    # Load data from the specific sheet '2023_dataset'
    file_path = os.path.join(raw_data_dir, 'global.xlsx')
    sheet_name = '2023_dataset'
    data = pd.read_excel(file_path, sheet_name=sheet_name)

    # 1. Drop rows with missing values
    data = data.dropna()

    # 2. Convert categorical columns to numeric (if any)

    if 'Category' in data.columns:
        data['Category'] = data['Category'].astype('category').cat.codes
    
    # 3. Normalize numeric columns (if needed)

    if 'Value' in data.columns:
        data['Value'] = (data['Value'] - data['Value'].mean()) / data['Value'].std()
    
    # Save the processed data
    processed_file_path = os.path.join(processed_data_dir, 'processed_data.csv')
    data.to_csv(processed_file_path, index=False)

if __name__ == "__main__":
    preprocess_data()