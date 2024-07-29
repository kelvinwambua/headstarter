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
    
    # Example: Load and preprocess data
    file_path = os.path.join(raw_data_dir, 'global.xlsx')
    
    if os.path.exists(file_path):
        data = pd.read_excel(file_path, engine='openpyxl')
        
        # Example preprocessing steps
        data.fillna(0, inplace=True)
        
        # Save processed data
        processed_file_path = os.path.join(processed_data_dir, 'processed_data.csv')
        data.to_csv(processed_file_path, index=False)
        print(f"Processed data saved to {processed_file_path}")
    else:
        print(f"File {file_path} does not exist.")

if __name__ == "__main__":
    """
    Run the data preprocessing script
    """
    preprocess_data()