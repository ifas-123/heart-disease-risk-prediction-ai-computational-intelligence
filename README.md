# 🫀 Heart Disease Risk Prediction System

An end-to-end Machine Learning pipeline and interactive Streamlit web application designed to predict cardiovascular disease risk based on clinical patient parameters. The system is powered by an **Optuna-Tuned LightGBM Classifier** and includes model persistence, a dynamic web interface, and multi-tier clinical risk classification.

---

## 📂 Repository Structure

```text
├── submissions/                           # Directory containing Kaggle evaluation submissions
├── app.py                                 # Streamlit web application source application code
├── heart_disease_model.joblib             # Serialized LightGBM model and feature metadata
├── train notebook.ipynb                   # Jupyter notebook for EDA, modeling, & tuning
├── requirements.txt                       # Required Python dependencies
├── train.csv                              # Training dataset
├── test.csv                               # Test dataset for inference
├── sample_submission.csv                  # Sample submission format file
├── tuned_lgb_submission.csv              # Tuned LightGBM prediction outputs
└── ultimate_tuned_ensemble_submission.csv # Ensemble prediction submission outputs

🌟 Key Features
Machine Learning Pipeline: Built using LightGBM with automated hyperparameter optimization via Optuna.

Model Serialization: Uses joblib to save both the trained estimator and exact feature ordering to prevent schema mismatches during inference.

Dynamic Web UI: Interactive Streamlit web interface with real-time field validation, human-readable dropdown options, and automated feature mapping.

3-Tier Risk Logic: Evaluates predicted probability into actionable diagnostic feedback:

🔴 High Risk: Probability >= 65%

🟡 Moderate Risk: Probability between 35% and 64%

🟢 Low Risk: Probability < 35%

Empirical Validation: Kaggle competition predictions generated and verified against test data.
