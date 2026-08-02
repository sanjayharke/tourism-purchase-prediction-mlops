# Wellness Tourism Purchase Prediction — MLOps Project

[![Tourism Purchase Prediction Pipeline](https://github.com/sanjayharke/tourism-purchase-prediction-mlops/actions/workflows/pipeline.yml/badge.svg)](https://github.com/sanjayharke/tourism-purchase-prediction-mlops/actions/workflows/pipeline.yml)

## Project Overview

This project develops an end-to-end machine-learning and MLOps system for
predicting whether a prospective customer is likely to purchase a wellness
tourism package.

The project combines data validation, preprocessing, model training,
hyperparameter tuning, MLflow experiment tracking, automated GitHub Actions,
model packaging and deployment through Streamlit and Hugging Face Spaces.

## Business Objective

The objective is to help the tourism company identify high-priority prospective
customers. Customers whose predicted probability exceeds the selected operating
threshold can be prioritized for targeted marketing campaigns.

## Prediction Target

The target variable is `ProdTaken`:

- `0` — The customer did not purchase the tourism package
- `1` — The customer purchased the tourism package

## Machine-Learning Workflow

1. Register and validate the tourism dataset
2. Verify the schema, target variable and checksum
3. Remove identifier and unnecessary columns
4. Remove duplicate records
5. Handle missing numerical and categorical values
6. Create stratified training and testing datasets
7. Construct a reusable preprocessing and modelling pipeline
8. Tune a Random Forest model using `GridSearchCV`
9. Evaluate classification performance
10. Select an operating threshold using the F2 score
11. Track parameters, metrics and artifacts using MLflow
12. Package the trained model for deployment
13. Validate the model using an automated smoke test
14. Deploy the prediction interface

## Model and Decision Threshold

The final classifier is a Random Forest model tuned using cross-validated grid
search. The deployed model uses an operating threshold of approximately `0.22`.

The threshold was selected by maximizing the F2 score on out-of-fold training
predictions. This gives greater importance to identifying potential purchasers
than to avoiding an additional marketing contact.

## MLOps Components

### MLflow

MLflow is used to record:

- Experiment and run information
- Hyperparameters
- Classification metrics
- Model artifacts
- Evaluation reports
- Confusion matrix
- ROC and precision-recall curves
- Trained model package

### GitHub Actions

The automated workflow contains three sequential jobs:

1. `register-data`
2. `prepare-data`
3. `train-track-evaluate`

Every push to the main branch initiates dataset registration, preprocessing,
model training, evaluation, artifact generation and model validation.

### Deployment

The trained model is deployed through:

- Streamlit Community Cloud
- Hugging Face Spaces

## Public Links

- **GitHub repository:**  
  https://github.com/sanjayharke/tourism-purchase-prediction-mlops

- **Streamlit application:**  
  https://tourism-purchase-prediction-mlops-hzvvn8wxazfouu3gsnwqsq.streamlit.app/

- **Hugging Face Space:**  
  https://huggingface.co/spaces/sanjayharke/tourism-purchase-prediction-mlops

## Repository Structure

```text
tourism-purchase-prediction-mlops/
├── .github/
│   └── workflows/
│       └── pipeline.yml
├── mlruns/
├── tourism_project/
│   ├── data/
│   │   └── tourism.csv
│   ├── deployment/
│   │   ├── app.py
│   │   ├── model.joblib
│   │   └── requirements.txt
│   ├── model_building/
│   │   ├── data_register.py
│   │   ├── prep.py
│   │   ├── project_config.py
│   │   ├── smoke_test.py
│   │   └── train.py
│   └── requirements.txt
├── Dockerfile
└── README.md
