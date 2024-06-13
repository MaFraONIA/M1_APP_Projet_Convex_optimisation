import logging
from data_loader import load_and_preprocess_data
from model_training import train_ista, train_gradient_descent, train_random_forest, train_svm, train_neural_network
from performance_evaluation import calculate_performance_metrics, plot_roc_curve, plot_confusion_matrix
from results_visualization import plot_roc_curves, plot_confusion_matrices
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    if X is None or y is None:
        logging.error("Data loading and preprocessing failed.")
        return
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define hyperparameters
    lambda_ = 0.1
    lr = 0.01
    max_iter = 100
    tol = 1e-6
    
    # Model training
    logging.info("Training models...")
    w_ista = train_ista(X_train, y_train, lambda_, max_iter, tol)
    w_gd = train_gradient_descent(X_train, y_train, lr, lambda_, max_iter, tol)
    model_rf = train_random_forest(X_train, y_train)
    model_svm = train_svm(X_train, y_train)
    model_nn = train_neural_network(X_train, y_train)
    
    # Generate predictions and probabilities for evaluation
    # For simplicity, these steps are assumed to be implemented within their respective training functions or a separate utility function
    # Placeholder for ISTA and Gradient Descent predictions and probabilities
    # ista_predictions, ista_probabilities = predict_ista(X_test, w_ista)
    # gd_predictions, gd_probabilities = predict_gd(X_test, w_gd)
    rf_predictions, rf_probabilities = model_rf.predict(X_test), model_rf.predict_proba(X_test)[:, 1]
    svm_predictions, svm_probabilities = model_svm.predict(X_test), model_svm.predict_proba(X_test)[:, 1]
    nn_predictions, nn_probabilities = model_nn.predict(X_test), model_nn.predict_proba(X_test)[:, 1]
    
    # Calculate performance metrics
    # Placeholder for ISTA and Gradient Descent metrics calculation
    # metrics_ista = calculate_performance_metrics(y_test, ista_predictions, ista_probabilities)
    # metrics_gd = calculate_performance_metrics(y_test, gd_predictions, gd_probabilities)
    metrics_rf = calculate_performance_metrics(y_test, rf_predictions, rf_probabilities)
    metrics_svm = calculate_performance_metrics(y_test, svm_predictions, svm_probabilities)
    metrics_nn = calculate_performance_metrics(y_test, nn_predictions, nn_probabilities)
    
    # Log metrics
    # logging.info(f"ISTA Metrics: {metrics_ista}")
    # logging.info(f"Gradient Descent Metrics: {metrics_gd}")
    logging.info(f"Random Forest Metrics: {metrics_rf}")
    logging.info(f"SVM Metrics: {metrics_svm}")
    logging.info(f"Neural Network Metrics: {metrics_nn}")
    
    # Plot ROC curves
    plot_roc_curves(y_test, [rf_probabilities, svm_probabilities, nn_probabilities], 
                    ['Random Forest', 'SVM', 'Neural Network'])
    
    # Plot confusion matrices
    plot_confusion_matrices(y_test, [rf_predictions, svm_predictions, nn_predictions], 
                             ['Random Forest', 'SVM', 'Neural Network'], class_names=['Benign', 'Malignant'])

if __name__ == "__main__":
    main()