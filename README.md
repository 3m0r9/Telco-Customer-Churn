# Telco Customer Churn Prediction

This project aims to predict customer churn for a telecommunications company using machine learning techniques. By analyzing customer data, the model helps identify which customers are likely to leave the company, enabling proactive retention strategies.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Modeling](#modeling)
5. [Evaluation](#evaluation)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Results](#results)
9. [Contributors](#contributors)
10. [License](#license)
11. [Let's Connect](#lets-connect)

## Project Overview

Customer churn is a significant issue for telecommunications companies, and predicting churn is critical for retaining customers. In this project, we develop machine learning models to predict whether a customer will leave the company, based on a variety of features such as:
- Contract length
- Payment method
- Monthly charges
- Internet service type

We explore and compare different models to achieve the best prediction accuracy and other performance metrics.

## Dataset

The dataset used in this project is the [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), available on Kaggle. It contains customer information for a telecommunications company, including features that describe each customer and a target variable indicating whether the customer churned.

- **Features**: CustomerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges.
- **Target Variable**: Churn (Yes/No).

## Data Preprocessing

Before training the machine learning models, the following preprocessing steps are applied:
- **Handling Missing Data**: Missing values in the `TotalCharges` column are filled or removed.
- **Encoding Categorical Variables**: Categorical features like gender, contract type, and payment method are encoded using one-hot encoding or label encoding.
- **Feature Scaling**: Numerical features such as `MonthlyCharges` and `TotalCharges` are scaled to ensure uniformity for models like Logistic Regression or SVM.
- **Class Imbalance**: Since churn is often a minority class, techniques like oversampling or SMOTE are used to balance the dataset.

## Modeling

We build and evaluate several machine learning models to predict customer churn:
1. **Logistic Regression**: A simple linear model for binary classification.
2. **Random Forest**: An ensemble method based on decision trees.
3. **XGBoost**: A gradient boosting technique that is effective for tabular data.
4. **Support Vector Machine (SVM)**: A powerful classifier that aims to find the optimal boundary between classes.

### Hyperparameter Tuning
We perform grid search and cross-validation to fine-tune model hyperparameters, aiming to maximize the performance of each model.

## Evaluation

The performance of the models is evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **ROC-AUC Curve**

A confusion matrix is also used to evaluate how well the models predict each class (churn and no churn).

## Installation

To run this project on your local machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/3m0r9/Telco-Customer-Churn.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Telco-Customer-Churn
   ```
3. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the `data/` directory.
2. Preprocess the dataset:
   ```bash
   python preprocess_data.py --input data/Telco-Customer-Churn.csv --output data/processed_data.csv
   ```
3. Train the machine learning models:
   ```bash
   python train_model.py --input data/processed_data.csv --model logistic_regression
   python train_model.py --input data/processed_data.csv --model random_forest
   python train_model.py --input data/processed_data.csv --model xgboost
   python train_model.py --input data/processed_data.csv --model svm
   ```
4. Evaluate the models on the test set:
   ```bash
   python evaluate_model.py --input data/processed_data.csv --model logistic_regression
   ```

## Results

The results of the models are as follows:
- **Logistic Regression**: 79% accuracy.
- **Random Forest**: 82% accuracy.
- **XGBoost**: 84% accuracy.
- **SVM**: 81% accuracy.

Further details on precision, recall, F1-score, and ROC-AUC scores can be found in the `results/` directory. Confusion matrices and plots of the ROC curves are also available.

## Contributors

- **Imran Abu Libda** - [3m0r9](https://github.com/3m0r9)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Let's Connect

- **GitHub** - [3m0r9](https://github.com/3m0r9)
- **LinkedIn** - [Imran Abu Libda](https://www.linkedin.com/in/imran-abu-libda/)
- **Email** - [imranabulibda@gmail.com](mailto:imranabulibda@gmail.com)
- **Medium** - [Imran Abu Libda](https://medium.com/@imranabulibda_23845)
