import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix, accuracy_score
import seaborn as sns

def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_precision_recall_threshold(y_test, y_prob, thresholds_to_mark):
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    proportion_label_1 = [(y_prob >= t).mean() for t in thresholds]
    accuracies = [accuracy_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision[:-1], color='green', lw=2, label='Precision')
    plt.plot(thresholds, recall[:-1], color='red', lw=2, label='Recall')
    plt.plot(thresholds, proportion_label_1, color='blue', lw=2, label='Proportion classified as label 1')
    plt.plot(thresholds, accuracies, color='orange', lw=2, label='Accuracy')

    for t in thresholds_to_mark:
        precision_val = precision[np.searchsorted(thresholds, t, side="left")]
        recall_val = recall[np.searchsorted(thresholds, t, side="left")]
        proportion_val = (y_prob >= t).mean()
        accuracy_val = accuracy_score(y_test, (y_prob >= t).astype(int))
        plt.axvline(x=t, color='gray', linestyle='--', lw=1)
        plt.text(t, precision_val, f'{precision_val:.2f}', color='green', fontsize=10, ha='right')
        plt.text(t, recall_val, f'{recall_val:.2f}', color='red', fontsize=10, ha='right')
        plt.text(t, proportion_val, f'{proportion_val:.2f}', color='blue', fontsize=10, ha='right')
        plt.text(t, accuracy_val, f'{accuracy_val:.2f}', color='orange', fontsize=10, ha='right')

    plt.xlabel('Classification Threshold')
    plt.ylabel('Precision / Recall / Proportion / Accuracy')
    plt.title('Precision, Recall, Proportion, and Accuracy vs. Classification Threshold')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix", model_label="", threshold=None):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    full_title = title
    if model_label:
        full_title += f" ({model_label})"
    if threshold is not None:
        full_title += f" [Threshold = {threshold}]"
    else:
        full_title += f" [Default Threshold = 0.5]"
    plt.title(full_title)
    plt.show()

def plot_precision_recall_curve(y_test, y_prob, thresholds_to_mark=None):
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    plt.figure()
    plt.plot(recall, precision, color='purple', lw=2, label='Precision-Recall Curve')
    
    # Add points for specific thresholds
    if thresholds_to_mark:
        for t in thresholds_to_mark:
            idx = np.searchsorted(thresholds, t, side="left")
            if idx < len(precision):
                plt.scatter(recall[idx], precision[idx], color='red', label=f'Threshold = {t:.2f}')
                plt.text(recall[idx], precision[idx], f'({recall[idx]:.2f}, {precision[idx]:.2f})', fontsize=8, color='black')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()
