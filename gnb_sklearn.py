# Gaussian Naive Bayes using scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from common_functions import plot_roc_curve, plot_precision_recall_threshold, plot_confusion_matrix, plot_precision_recall_curve

# Load and prepare the dataset
df = pd.read_csv("loan.csv")
df.drop(columns=["ID", "ZIP Code", "Experience"], inplace=True)

X = df.drop("Personal Loan", axis=1)
y = df["Personal Loan"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and test scikit-learn model
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

# Print classification metrics once
print("Scikit-learn Gaussian Naive Bayes")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=2))

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix", model_label="From sklearn")

# Predict probabilities
y_prob = gnb.predict_proba(X_test)[:, 1]

# Plot ROC curve
plot_roc_curve(y_test, y_prob)

# Plot Precision-Recall curve with points for selected thresholds
specific_thresholds = [0.02, 0.05, 0.16, 0.5, 0.93]
plot_precision_recall_curve(y_test, y_prob, thresholds_to_mark=specific_thresholds)

# Compute Precision, Recall, and Proportion of data classified as label 1
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
proportion_label_1 = [(y_prob >= t).mean() for t in thresholds]

# Plot Precision, Recall, and Proportion vs. Classification Threshold on the same graph
plot_precision_recall_threshold(y_test, y_prob, specific_thresholds)

# Allow tuning of classification probability threshold
threshold = 0.02  # Default threshold
y_pred_tuned = (y_prob >= threshold).astype(int)

# Print classification metrics for tuned threshold
print(f"Classification metrics for threshold = {threshold}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tuned):.2f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred_tuned, digits=2))

# Plot confusion matrix for tuned threshold
plot_confusion_matrix(y_test, y_pred_tuned, title=f'Confusion Matrix', model_label="From sklearn", threshold=threshold)