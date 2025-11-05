import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(y_test, predictions):
    """
    Evaluate a model.
    Parameters:
    y_test (Series): True labels for the test set.
    predictions (array): Predictions from a model.

    Returns:
    dict: Evaluation metrics including confusion matrix and classification report.
    """
    #check if predictions are in {-1, 1} and convert to {0, 1}
    if set(np.unique(predictions)) == {-1, 1}:
        preds_binary = (predictions == -1).astype(int)
    else:
        preds_binary = predictions
    cm = confusion_matrix(y_test, preds_binary)
    cr = classification_report(y_test, preds_binary, output_dict=True)
    return {"confusion_matrix": cm, "classification_report": cr}

def print_evaluation(evaluation_results):
    """
    Print evaluation results.
    Parameters:
    evaluation_results (dict): Dictionary containing confusion matrix and classification report.
    """
    print("Confusion Matrix:")
    print(evaluation_results["confusion_matrix"])
    print("\nClassification Report:")
    for label, metrics in evaluation_results["classification_report"].items():
        if label in ['0', '1']:
            print(f"Label {label}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

def plot_confusion_matrix(cm, class_names=['Normal', 'Anomaly'],title="model"):
    """
    Plot confusion matrix using matplotlib.
    Parameters:
    cm (array): Confusion matrix.
    class_names (list): List of class names.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix'+" for "+title, fontsize=16, pad=20)
    plt.show()