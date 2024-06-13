import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_performance_metrics(y_true, y_pred, y_score=None):
    """
    Calculate performance metrics including accuracy, recall, and F1-score.
    If y_score is provided, also calculate AUC.
    
    Parameters:
    - y_true : array-like, true labels.
    - y_pred : array-like, predicted labels.
    - y_score : array-like, target scores, can either be probability estimates of the positive class,
                confidence values, or non-thresholded measure of decisions (as returned by
                “decision_function” on some classifiers).
    
    Returns:
    - metrics : dict, containing calculated metrics.
    """
    try:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }
        
        if y_score is not None:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            metrics['auc'] = auc(fpr, tpr)
        
        logging.info("Performance metrics calculated successfully.")
        return metrics
    except Exception as e:
        logging.error("Error calculating performance metrics: %s", e, exc_info=True)
        raise

def plot_roc_curve(y_true, y_scores, labels):
    """
    Plot ROC curve for each model.
    
    Parameters:
    - y_true : array-like, true binary labels.
    - y_scores : list of array-like, target scores for each model.
    - labels : list of str, names of the models.
    """
    try:
        plt.figure(figsize=(10, 8))
        for y_score, label in zip(y_scores, labels):
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
        logging.info("ROC curve plotted successfully.")
    except Exception as e:
        logging.error("Error plotting ROC curve: %s", e, exc_info=True)
        raise

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot the confusion matrix.
    
    Parameters:
    - y_true : array-like, true labels.
    - y_pred : array-like, predicted labels.
    - class_names : list of str, names of the classes.
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.show()
        logging.info("Confusion matrix plotted successfully.")
    except Exception as e:
        logging.error("Error plotting confusion matrix: %s", e, exc_info=True)
        raise