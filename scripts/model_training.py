import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os
"""
The module trains a simple RandomForestClassifier model on the processed data.
"""


def train_model():
    """
    Train a RandomForestClassifier model on the processed data and save the model.
    """
    processed_data_dir = os.path.join(os.getcwd(), 'data', 'processed')
    model_dir = os.path.join(os.getcwd(), 'models')
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Load processed data
    file_path = os.path.join(processed_data_dir, 'processed_data.csv')
    data = pd.read_csv(file_path)
    
    # Example: Define features and target
    X = data.drop('target_column', axis=1)
    y = data['target_column']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Save model
    model_path = os.path.join(model_dir, 'random_forest_model.pkl')
    joblib.dump(model, model_path)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    """
    Train the model when the script runs
    """
    train_model()
