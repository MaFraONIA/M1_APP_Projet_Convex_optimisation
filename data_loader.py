import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data():
    try:
        logging.info("Loading the breast cancer Wisconsin dataset.")
        # Load the dataset
        data = load_breast_cancer()
        X, y = data.data, data.target
        
        logging.info("Preprocessing the dataset.")
        # Preprocessing steps
        # Standardize features by removing the mean and scaling to unit variance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        logging.info("Data loading and preprocessing completed successfully.")
        return X_scaled, y
    except Exception as e:
        logging.error("An error occurred during data loading and preprocessing.", exc_info=True)
        # Optionally, return None or empty arrays depending on how you wish to handle errors
        return None, None