import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

"""
Train a machine learning model on the processed data and save the model.
"""


def train_model():
    """
    Train a machine learning model on the processed data and save the model.
    """
    processed_data_dir = os.path.join(os.getcwd(), 'data', 'processed')
    model_dir = os.path.join(os.getcwd(), 'models')
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Load the processed data
    data_path = os.path.join(processed_data_dir, 'processed_data.csv')
    data = pd.read_csv(data_path)
    
    # Debug: Print the columns of the DataFrame
    print("Columns in the dataset:", data.columns)
    
    # Check if the actual target column exists in the data
    target_column = 'Criminality avg,'
    if target_column not in data.columns:
        raise KeyError(f"The column '{target_column}' is not found in the data.")
    
    # Split the data into features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # One-hot encode categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_columns]), columns=encoder.get_feature_names_out(categorical_columns))
    
    # Drop original categorical columns and concatenate encoded columns
    X = X.drop(categorical_columns, axis=1)
    X = pd.concat([X, X_encoded], axis=1)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    model_path = os.path.join(model_dir, 'random_forest_model.pkl')
    pd.to_pickle(model, model_path)
    
    print("Model training completed and saved to", model_path)

if __name__ == "__main__":
    train_model()