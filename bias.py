"""
Problem: Information Cascade in Machine Learning Models

In this exercise, you'll investigate how artificial bias in features affects model predictions.
We'll create an artificial feature that is increasingly correlated with the target variable
and observe how this affects the model's predictions.

Your tasks:
1. Complete the add_bias_column function to create an artificial 'Bias' feature with given correlation strength.
2. Complete the evaluate_model function to train and evaluate the model in terms of accuracy and correlation between 'Bias' and predictions.

The data contains banana quality measurements where:
- Target variable 'Quality' is binary ('Good' or 'Bad')
- Features include Size, Weight, Sweetness, etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

np.random.seed(12)

def add_bias_column(X, y, correlation_strength):
    """
    Add an artificial bias column with specified correlation to target.
    
    Args:
        X: Feature matrix
        y: Target vector
        correlation_strength: Proportion of samples where bias feature matches target
        
    Returns:
        X_biased: Feature matrix with added bias column
    """
    X_biased = X.copy()
    
    # TODO 1/4: Initialize bias column with random 0/1 values (~1 line)
    # Hint: Use np.random.randint

    # TODO 2/4: For positive class (y=1), set 'correlation_strength' proportion of positive classes to 1 to match y=1 (~3-5 lines)
    # Hint: Use np.random.choice to select indices
    
    # TODO 3/4: For negative class (y=0), set 'correlation_strength' proportion of negative classes to 0 to match y=0 (~3-5 lines)
    # Hint: Use np.random.choice to select indices (same as above)
    
    # TODO 4/4: Add 'Bias' column to dataframe (~1 line)
    X_biased['Bias'] = # TODO

    return X_biased

def evaluate_model(X, y, X_test, y_test):
    """
    Train and evaluate the model.
    
    Args:
        X: Training features
        y: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        accuracy: Model accuracy
        bias: Correlation between artificial feature and predictions
    """
    model = LogisticRegression()
    # TODO 1/4: Train logistic regression model (~1 line)
    
    # TODO 2/4: Get predictions (~1 line)
    
    # TODO 3/4: Calculate accuracy (~1 line)
    accuracy = # TODO

    # TODO 4/4: Calculate correlation between artificial feature and predictions (~1 line)
    # Hint: Use np.corrcoef
    prediction_bias = # TODO
    
    return accuracy, prediction_bias

def main():
    # Load and prepare data
    df = pd.read_csv('dataset.csv')
    X = df.drop('Quality', axis=1)
    y = (df['Quality'] == 'Good').astype(int)

    # Create datasets with different bias strengths
    correlation_strengths = [0, 0.2, 0.4, 0.6, 0.8, 1]
    datasets = []
    for strength in correlation_strengths:
        X_biased = add_bias_column(X, y, strength)
        datasets.append(X_biased)

    # Store results
    results = {
        'correlation_strength': [],
        'accuracy': [],
        'prediction_bias': []
    }
    
    # Evaluate model
    for i, dataset in enumerate(datasets):
        strength = correlation_strengths[i]
        
        X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.2, random_state=42)
        
        # Evaluate model
        accuracy, prediction_bias = evaluate_model(X_train, y_train, X_test, y_test)
        
        # Store results
        results['correlation_strength'].append(strength)
        results['accuracy'].append(accuracy)
        results['prediction_bias'].append(prediction_bias)
        
        print(f"Correlation strength: {strength}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Correlation between Bias and Predictions: {prediction_bias:.3f}")
        print("-" * 50)
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['correlation_strength'], results['prediction_bias'], 'o-')
    plt.xlabel('Correlation Strength')
    plt.ylabel('Correlation between Bias and Predictions')
    plt.title('Correlation between Bias and Predictions vs Correlation Strength')
    
    plt.subplot(1, 2, 2)
    plt.plot(results['correlation_strength'], results['accuracy'], 'o-', label='Accuracy')
    plt.xlabel('Correlation Strength')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Correlation Strength')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('bias_experiment_results.png')
    plt.show()

if __name__ == "__main__":
    main()
