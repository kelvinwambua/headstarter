import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from sklearn.preprocessing import OneHotEncoder

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
    
    # Define features and target
    target_column = 'Criminality avg,'
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # One-hot encode categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    
    # Load the encoder used during training
    encoder_path = os.path.join(model_dir, 'encoder.pkl')
    if os.path.exists(encoder_path):
        encoder = joblib.load(encoder_path)
    else:
        encoder.fit(X[categorical_columns])
        joblib.dump(encoder, encoder_path)
    
    # Transform categorical columns
    X_encoded = pd.DataFrame(encoder.transform(X[categorical_columns]), columns=encoder.get_feature_names_out(categorical_columns))
    
    # Drop original categorical columns and concatenate encoded columns
    X = X.drop(categorical_columns, axis=1)
    X = pd.concat([X, X_encoded], axis=1)
    
    # Load the model
    model_path = os.path.join(model_dir, 'random_forest_model.pkl')
    model = joblib.load(model_path)
    
    # Predict
    y_pred = model.predict(X)
    
    # Evaluate the model
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

if __name__ == "__main__":
    evaluate_model()