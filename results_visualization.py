import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_roc_curves(y_true, model_predictions, model_names):
    """
    Plots ROC curves for multiple models for comparison.
    Parameters:
    - y_true: array-like, true labels.
    - model_predictions: list of array-like, predicted probabilities for the positive class from each model.
    - model_names: list of str, names of the models corresponding to the predictions.
    """
    try:
        plt.figure(figsize=(10, 8))
        for y_score, label in zip(model_predictions, model_names):
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.show()
        logging.info("ROC curves plotted successfully.")
    except Exception as e:
        logging.error("Error plotting ROC curves: %s", e, exc_info=True)
        raise

def plot_confusion_matrices(y_true, model_predictions, model_names, class_names):
    """
    Plots confusion matrices for multiple models for comparison.
    Parameters:
    - y_true: array-like, true labels.
    - model_predictions: list of array-like, predicted labels from each model.
    - model_names: list of str, names of the models corresponding to the predictions.
    - class_names: list of str, names of the classes.
    """
    try:
        for y_pred, label in zip(model_predictions, model_names):
            plt.figure(figsize=(6, 5))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title(f'Confusion Matrix: {label}')
            plt.show()
        logging.info("Confusion matrices plotted successfully.")
    except Exception as e:
        logging.error("Error plotting confusion matrices: %s", e, exc_info=True)
        raise