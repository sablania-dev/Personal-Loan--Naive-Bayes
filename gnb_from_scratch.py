# Gaussian Naive Bayes from scratch

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from common_functions import plot_roc_curve, plot_precision_recall_threshold, plot_confusion_matrix, plot_precision_recall_curve

# Load and prepare the dataset
df = pd.read_csv("loan.csv")
df.drop(columns=["ID", "ZIP Code"], inplace=True)

X = df.drop("Personal Loan", axis=1)
y = df["Personal Loan"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = {}

        for c in self.classes:
            X_c = X[y == c]
            self.parameters[c] = {
                "mean": X_c.mean(axis=0),
                "var": X_c.var(axis=0),
                "prior": X_c.shape[0] / X.shape[0]
            }

    def _calculate_likelihood(self, mean, var, x):
        eps = 1e-6
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
        exponent = np.exp(-(x - mean)**2 / (2 * var + eps))
        return coeff * exponent

    def _calculate_posterior(self, x):
        posteriors = {}

        for c, params in self.parameters.items():
            prior = np.log(params["prior"])
            likelihoods = self._calculate_likelihood(params["mean"], params["var"], x)
            posteriors[c] = prior + np.sum(np.log(likelihoods + 1e-6))

        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        return np.array([self._calculate_posterior(x) for x in X])

# Train and test the custom model
gnb_custom = GaussianNaiveBayes()
gnb_custom.fit(X_train.values, y_train.values)
y_pred_custom = gnb_custom.predict(X_test.values)

print("Custom Gaussian Naive Bayes")
print("Accuracy:", accuracy_score(y_test, y_pred_custom))
print("Classification Report:")
print(classification_report(y_test, y_pred_custom))

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred_custom, title="Confusion Matrix", model_label="From Scratch")

# Predict probabilities (manually calculate probabilities for class 1)
def predict_proba_custom(model, X):
    probabilities = []
    for x in X:
        posteriors = {}
        for c, params in model.parameters.items():
            prior = np.log(params["prior"])
            likelihoods = model._calculate_likelihood(params["mean"], params["var"], x)
            posteriors[c] = prior + np.sum(np.log(likelihoods + 1e-6))
        exp_posteriors = {k: np.exp(v) for k, v in posteriors.items()}
        total = sum(exp_posteriors.values())
        probabilities.append(exp_posteriors[1] / total)  # Probability of class 1
    return np.array(probabilities)

# Predict probabilities
y_prob_custom = predict_proba_custom(gnb_custom, X_test.values)

# Plot ROC curve
plot_roc_curve(y_test, y_prob_custom)

# Plot Precision-Recall curve with points for selected thresholds
specific_thresholds = [0.02, 0.04, 0.1, 0.5, 0.93]
plot_precision_recall_curve(y_test, y_prob_custom, thresholds_to_mark=specific_thresholds)

# Compute Precision, Recall, and Proportion of data classified as label 1
precision, recall, thresholds = precision_recall_curve(y_test, y_prob_custom)
proportion_label_1 = [(y_prob_custom >= t).mean() for t in thresholds]

# Plot Precision, Recall, and Proportion vs. Classification Threshold on the same graph
plot_precision_recall_threshold(y_test, y_prob_custom, specific_thresholds)

# Allow tuning of classification probability threshold
threshold = 0.02  # Default threshold
y_pred_tuned_custom = (y_prob_custom >= threshold).astype(int)

# Print classification metrics for tuned threshold
print(f"Classification metrics for threshold = {threshold}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tuned_custom):.2f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred_tuned_custom, digits=2))

# Plot confusion matrix for tuned threshold
plot_confusion_matrix(y_test, y_pred_tuned_custom, title=f'Confusion Matrix', model_label="From Scratch", threshold=threshold)