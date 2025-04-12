# Personal Loan: Naive Bayes Classifier

This project is a part of the course CS60020: Foundations of Algorithm Design and Machine Learning, Spring 2025, IIT Kharagpur.
This project demonstrates the implementation and evaluation of a Gaussian Naive Bayes classifier for predicting whether a customer will accept a personal loan offer. The dataset used is `loan.csv`, which contains customer information and the target variable `Personal Loan`.

## Project Structure

The project contains the following files:

1. **`gnb_from_scratch.py`**  
   Implements Gaussian Naive Bayes from scratch without using machine learning libraries. It includes:
   - Training the model on the dataset.
   - Predicting probabilities and labels.
   - Plotting evaluation metrics such as ROC and Precision-Recall curves.
   - Allowing threshold tuning for classification.

2. **`gnb_sklearn.py`**  
   Uses scikit-learn's `GaussianNB` to implement the Gaussian Naive Bayes classifier. It includes:
   - Training and testing the model.
   - Generating evaluation metrics and visualizations.
   - Threshold tuning for classification.

3. **`gnb_gpt.py`**  
   An AI-generated implementation of Gaussian Naive Bayes using scikit-learn. It provides similar functionality to `gnb_sklearn.py` but is structured differently.

4. **`common_functions.py`**  
   Contains utility functions for plotting evaluation metrics:
   - ROC Curve
   - Precision-Recall Curve
   - Confusion Matrix
   - Precision, Recall, and Proportion vs. Threshold

## Dataset

The dataset `loan.csv` should be placed in the same directory as the scripts. It must include the following columns:
- Features: Customer attributes such as income, age, family size, etc.
- Target: `Personal Loan` (binary: 0 or 1)

Columns `ID` and `ZIP Code` are dropped during preprocessing.

## How to Run

1. **Pre-requisites**:
   - Python 3.7 or higher
   - Required libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

   Install dependencies using:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

2. **Run the scripts**:
   - To use the custom implementation:
     ```bash
     python gnb_from_scratch.py
     ```
   - To use scikit-learn's implementation:
     ```bash
     python gnb_sklearn.py
     ```
   - To use the AI-generated implementation:
     ```bash
     python gnb_gpt.py
     ```

3. **Output**:
   - Accuracy, classification reports, and confusion matrices will be printed.
   - Plots for ROC and Precision-Recall curves will be displayed.

## Custom Threshold Tuning

All scripts allow tuning the classification threshold to optimize for specific metrics (e.g., precision or recall). Adjust the `threshold` variable in the scripts to set a custom threshold.

## Visualizations

The following visualizations are generated:
- **ROC Curve**: Shows the trade-off between true positive rate and false positive rate.
- **Precision-Recall Curve**: Highlights the balance between precision and recall.
- **Confusion Matrix**: Displays the counts of true positives, true negatives, false positives, and false negatives.
- **Precision, Recall, and Proportion vs. Threshold**: Helps in selecting an optimal threshold.

## License

This project is for educational purposes and is not licensed for commercial use.