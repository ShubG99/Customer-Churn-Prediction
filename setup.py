from setuptools import setup, find_packages

setup(
    name="telecom_churn_prediction",
    version="1.0.0",
    author="Machine Learning Engineering Team",
    description="End-to-End Production Machine Learning Pipeline for Telecom Customer Churn Prediction",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "imbalanced-learn>=0.11.0",
        "optuna>=3.3.0",
        "shap>=0.42.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "mlflow>=2.8.0",
        "pyyaml>=6.0",
    ],
)
