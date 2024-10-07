# Credit-Risk-Analysis-and-Prediction


This project performs an in-depth analysis of credit card risk, focusing on predicting loan default probabilities and generating corresponding credit scores. Using various machine learning techniques and data visualization tools, the project explores customer demographics, loan characteristics, and creditworthiness. Additionally, a pipeline for predicting credit risk based on historical loan data has been implemented using logistic regression and XGBoost models.

## Introduction

Credit risk analysis is essential for financial institutions to minimize losses from loan defaults. In this project, machine learning models such as Logistic Regression and XGBoost are used to predict loan defaults and assign credit scores to customers. The primary goals are:
- Predicting the likelihood of a customer defaulting on a loan.
- Analyzing key factors that contribute to credit risk.
- Visualizing the distribution of loan statuses and amounts across different categories.

## Dataset

The dataset used in this project contains loan-related and demographic data including:
- Personal attributes like age, income, home ownership, and employment length.
- Loan information such as loan amount, interest rate, loan grade, and loan percentage of income.
- Credit history details like default history and credit history length.

Missing values were handled through data imputation, and outliers were removed to ensure model robustness.

## Data Preprocessing

- **Null Value Removal**: Rows with missing values were removed.
- **Outlier Removal**: Outliers in age, employment length, and income were filtered out.
- **Feature Scaling**: Numeric features were standardized using `StandardScaler`.
- **Categorical Encoding**: Categorical features were one-hot encoded for use in machine learning models.

## Exploratory Data Analysis (EDA)

The project includes several visualizations for a better understanding of the data:
- **Box Plots**: Visualize loan percentages across different loan grades and statuses.
- **Distribution Plots**: Display the distribution of credit scores among customers.
- **Sankey Diagrams**: Show the flow of loan amounts by loan grades and statuses.
- **Heatmaps**: Present loan amounts across loan grades and default statuses.
- **Parallel Categories Diagrams**: Illustrate correlations between home ownership, loan intent, loan grades, and default status.

## Modeling and Prediction

### Models Used:
1. **Logistic Regression**: A standard model to predict binary outcomes like loan default.
2. **XGBoost Classifier**: An ensemble learning method optimized for handling imbalanced data.

### Steps:
- **SMOTE**: Synthetic Minority Oversampling Technique was applied to handle class imbalance.
- **ROC Curve and AUC Score**: Models were evaluated using the ROC curve and AUC score to compare performance.
- **Credit Score Calculation**: Credit scores were calculated using the formula:  
  `Credit Score = 300 + (550 * (1 - default_probability))`

The best-performing model was then used to predict credit risk across the entire dataset, which was saved to a new file: `dataset_with_predictions.csv`.

## Results

- **ROC-AUC Score**: Both models achieved competitive ROC-AUC scores, with XGBoost generally outperforming Logistic Regression.
- **Credit Scores**: Customers were assigned credit scores based on their default probabilities.
- **Visual Insights**: The visualizations provide a clear understanding of how loan characteristics and demographics influence credit risk.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-risk-analysis.git
   ```

2.Install the required dependencies:

```bash
Copy code
pip install -r requirements.txt
```

3.Download the dataset and place it in the repository folder.

### Usage

1.Run the data preprocessing and model training:

```bash
Copy code
python credit_risk_analysis.py
```

2.Generate visualizations:

```bash
Copy code
python visualization.py
```
The generated CSV files with predictions and credit scores will be saved in the project folder.

## Technologies Used
Python: Core programming language.
Pandas: For data manipulation and preprocessing.
Scikit-learn: For building machine learning models.
Imbalanced-learn (SMOTE): For handling class imbalance.
XGBoost: For advanced machine learning predictions.
Plotly: For interactive data visualizations.
Seaborn: For static visualizations.
Matplotlib: For plotting ROC curves and other charts.
