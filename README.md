# Information Cascade in Machine Learning Models

This project investigates how artificial bias in features affects model predictions. It demonstrates the concept of information cascade in machine learning by creating an artificial feature that is increasingly correlated with the target variable and observing how this affects the model's predictions.

## Project Overview

The project uses a dataset of banana quality measurements where:
- Target variable 'Quality' is binary ('Good' or 'Bad')
- Features include Size, Weight, Sweetness, etc.

The dataset was sourced from https://www.kaggle.com/datasets/l3llff/banana?.

The main components of the project:
1. `add_bias_column`: Creates an artificial 'Bias' feature with specified bias strength
2. `evaluate_model`: Trains and evaluates the model in terms of accuracy and correlation between 'Bias' and predictions
3. Visualization of results showing how bias affects model performance

## Requirements

- Python 3.x
- Required packages:
  - numpy
  - pandas
  - matplotlib
  - scikit-learn

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure you have the `dataset.csv` file in the project directory
2. Run the main script:
```bash
python bias.py
```

The script will:
- Load and prepare the data
- Create datasets with different bias strengths (0, 0.2, 0.4, 0.6, 0.8, 1)
- Train and evaluate models for each bias strength
- Generate plots showing:
  - Correlation between Bias and Predictions vs Bias Strength
  - Accuracy vs Bias Strength

Results will be saved as 'bias_experiment_results.png'.