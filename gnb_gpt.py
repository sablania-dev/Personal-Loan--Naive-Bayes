# AI-generated implementation of Gaussian Naive Bayes using scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from common_functions import plot_roc_curve, plot_precision_recall_threshold, plot_confusion_matrix, plot_precision_recall_curve

# Load dataset and preprocess
data = pd.read_csv("loan.csv")
data.drop(columns=["ID", "ZIP Code"], inplace=True)

features = data.drop("Personal Loan", axis=1)
target = data["Personal Loan"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model performance
print("AI-Generated Gaussian Naive Bayes")
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
print("Classification Report:")
print(classification_report(y_test, predictions, digits=2))

# Plot confusion matrix
plot_confusion_matrix(y_test, predictions, title="Confusion Matrix", model_label="AI-Generated")

# Predict probabilities
probabilities = model.predict_proba(X_test)[:, 1]

# Plot ROC curve
plot_roc_curve(y_test, probabilities)

# Plot Precision-Recall curve with threshold points
thresholds_to_highlight = [0.02, 0.05, 0.16, 0.5, 0.93]
plot_precision_recall_curve(y_test, probabilities, thresholds_to_mark=thresholds_to_highlight)

# Compute and plot Precision, Recall, and Proportion vs. Threshold
plot_precision_recall_threshold(y_test, probabilities, thresholds_to_highlight)

# Adjust classification threshold and evaluate
custom_threshold = 0.02
adjusted_predictions = (probabilities >= custom_threshold).astype(int)

print(f"Metrics for Threshold = {custom_threshold}")
print(f"Accuracy: {accuracy_score(y_test, adjusted_predictions):.2f}")
print("Classification Report:")
print(classification_report(y_test, adjusted_predictions, digits=2))

# Plot confusion matrix for adjusted threshold
plot_confusion_matrix(y_test, adjusted_predictions, title="Confusion Matrix", model_label="AI-Generated", threshold=custom_threshold)