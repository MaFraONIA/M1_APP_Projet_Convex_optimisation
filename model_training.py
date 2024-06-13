import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from ista_algorithm import ISTA
from gradient_descent import logistic_regression_gd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_ista(X, y, lambda_, max_iter=100, tol=1e-6):
    """
    Trains a logistic regression model using the ISTA algorithm.
    
    Parameters:
    - X: numpy.ndarray, feature matrix.
    - y: numpy.ndarray, target vector.
    - lambda_: float, regularization parameter.
    - max_iter: int, maximum number of iterations.
    - tol: float, tolerance for convergence.
    
    Returns:
    - w: numpy.ndarray, the final weight vector.
    """
    try:
        w = ISTA(X, y, lambda_, max_iter, tol)
        logging.info("ISTA model training completed successfully.")
        return w
    except Exception as e:
        logging.error("An error occurred during ISTA model training", exc_info=True)
        raise

def train_gradient_descent(X, y, lr=0.01, lambda_=0.1, max_iter=1000, tol=1e-6):
    """
    Trains a logistic regression model using classic gradient descent.
    
    Parameters:
    - X: numpy.ndarray, feature matrix.
    - y: numpy.ndarray, target vector.
    - lr: float, learning rate.
    - lambda_: float, regularization parameter.
    - max_iter: int, maximum number of iterations.
    - tol: float, tolerance for convergence.
    
    Returns:
    - w: numpy.ndarray, the final weight vector.
    """
    try:
        w = logistic_regression_gd(X, y, lr, lambda_, max_iter, tol)
        logging.info("Gradient descent model training completed successfully.")
        return w
    except Exception as e:
        logging.error("An error occurred during gradient descent model training", exc_info=True)
        raise

def train_random_forest(X, y, n_estimators=100, max_depth=None, random_state=None):
    """
    Trains a Random Forest classifier.
    
    Parameters:
    - X: numpy.ndarray, feature matrix.
    - y: numpy.ndarray, target vector.
    - n_estimators: int, the number of trees in the forest.
    - max_depth: int, the maximum depth of the trees.
    - random_state: int, controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features).
    
    Returns:
    - The trained Random Forest model.
    """
    try:
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        rf.fit(X, y)
        logging.info("Random Forest model training completed successfully.")
        return rf
    except Exception as e:
        logging.error("An error occurred during Random Forest model training", exc_info=True)
        raise

def train_svm(X, y, C=1.0, kernel='rbf', random_state=None):
    """
    Trains a Support Vector Machine (SVM) classifier.
    
    Parameters:
    - X: numpy.ndarray, feature matrix.
    - y: numpy.ndarray, target vector.
    - C: float, regularization parameter.
    - kernel: string, specifies the kernel type to be used in the algorithm.
    - random_state: int, the seed of the pseudo random number generator used when shuffling the data for probability estimates.
    
    Returns:
    - The trained SVM model.
    """
    try:
        svm = SVC(C=C, kernel=kernel, probability=True, random_state=random_state)
        svm.fit(X, y)
        logging.info("SVM model training completed successfully.")
        return svm
    except Exception as e:
        logging.error("An error occurred during SVM model training", exc_info=True)
        raise

def train_neural_network(X, y, hidden_layer_sizes=(100,), activation='relu', max_iter=200, random_state=None):
    """
    Trains a Neural Network (MLPClassifier) classifier.
    
    Parameters:
    - X: numpy.ndarray, feature matrix.
    - y: numpy.ndarray, target vector.
    - hidden_layer_sizes: tuple, the ith element represents the number of neurons in the ith hidden layer.
    - activation: string, activation function for the hidden layer.
    - max_iter: int, maximum number of iterations.
    - random_state: int, determines the random number generation for weights and bias initialization, train-test split if early stopping is used, and batch sampling when solver='sgd' or 'adam'.
    
    Returns:
    - The trained Neural Network model.
    """
    try:
        nn = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter, random_state=random_state)
        nn.fit(X, y)
        logging.info("Neural Network model training completed successfully.")
        return nn
    except Exception as e:
        logging.error("An error occurred during Neural Network model training", exc_info=True)
        raise