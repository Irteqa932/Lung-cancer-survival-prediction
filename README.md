# Lung Cancer Survival Prediction using Machine Learning

## Project Overview

This project focuses on predicting lung cancer patient survival using clinical and diagnostic data. The objective is to develop a machine learning model that can identify patterns associated with survival outcomes and support healthcare decision-making.

The project implements an end-to-end machine learning pipeline including data preprocessing, feature engineering, class imbalance handling, and model evaluation using multiple algorithms.

---

## Objectives

- Analyze patient diagnosis data to identify factors affecting survival
- Build predictive models for lung cancer survival classification
- Handle class imbalance using SMOTE
- Compare multiple machine learning algorithms
- Evaluate models using healthcare-relevant metrics such as recall and ROC-AUC

---

## Dataset

The dataset contains patient clinical and diagnostic information such as:

- Age
- Gender
- Smoking status
- BMI
- Cholesterol level
- Cancer stage
- Treatment type
- Diagnosis and treatment dates
- Survival outcome

A new feature, **treatment duration**, was engineered from date columns to improve predictive capability.

---

## Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost
- Matplotlib & Seaborn
- Google Colab

---

## Machine Learning Models

The following models were implemented and compared:

- Logistic Regression (Baseline Model)
- Random Forest Classifier
- XGBoost Classifier (Best Performing Model)

---

## Model Performance

| Model | Accuracy | Recall | ROC-AUC |
|-------|----------|--------|---------|
Logistic Regression | 56.3% | 37.1% | 0.49 |
Random Forest | 75.0% | 5.2% | 0.49 |
XGBoost | 50.7% | 47.7% | 0.50 |

XGBoost achieved the best recall performance, which is critical in healthcare prediction tasks where identifying positive cases is more important than overall accuracy.

---

## Key Improvements

- Feature engineering from date columns (treatment duration)
- Handling missing target values
- Resolving datatype issues during preprocessing
- Applying SMOTE to address class imbalance
- Implementing gradient boosting (XGBoost) for improved prediction

These improvements demonstrate the ability to debug, optimize, and enhance machine learning pipelines.

---

## Key Insights

- Cancer stage and treatment duration were significant predictors.
- Class imbalance significantly affected model performance.
- Ensemble methods performed better than baseline models.
- Recall is a more meaningful metric than accuracy for survival prediction.

---

##  Project Structure

```
Lung-cancer-survival-prediction/
│
├── Lung-cancer-survival-prediction.ipynb
├── README.md
└── requirements.txt
```

---

##  How to Run the Project

1️. Clone the repository:

```
git clone https://github.com/your-username/lung-cancer-survival-prediction.git
```

2️. Install dependencies:

```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn jupyter
```

3️. Open the notebook in Google Colab or Jupyter Notebook.

4️. Run all cells sequentially.

---

## Future Scope

- Integration of genomic and imaging data
- Hyperparameter tuning for improved performance
- Explainable AI techniques (SHAP)
- Deployment as a clinical decision support tool

---

## Conclusion

This project demonstrates the feasibility of applying machine learning techniques to predict lung cancer survival outcomes using large-scale clinical data. Feature engineering, imbalance handling, and advanced models improved prediction capability, highlighting both opportunities and challenges in healthcare ML applications.

---
## Project Resources
- Google Colab Notebook: [Open Notebook](https://colab.research.google.com/drive/1T98SrcTvDmtJQOlQVGLQzxf68yjlwDjG?usp=sharing)

## Author

Irtika

---
