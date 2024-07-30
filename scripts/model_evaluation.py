import pandas as pd
from sklearn.metrics import classification_report
import joblib
import os
"""
The module evaluates the model on the processed data.
"""


def evaluate_model():
    """
    Load the processed data, model, predict, and evaluate the model.
    """
    processed_data_dir = os.path.join(os.getcwd(), 'data', 'processed')
    model_dir = os.path.join(os.getcwd(), 'models')
    
    # Load processed data
    file_path = os.path.join(processed_data_dir, 'processed_data.csv')
    data = pd.read_csv(file_path)
    
    # Example: Define features and target
    X = data.drop('target_column', axis=1)
    y = data['target_column']
    
    # Load model
    model_path = os.path.join(model_dir, 'random_forest_model.pkl')
    model = joblib.load(model_path)
    
    # Predict and evaluate
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))


if __name__ == "__main__":
    """
    Evaluate the model when the script runs
    """
    evaluate_model()
