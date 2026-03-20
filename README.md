# Titanic Survival Prediction

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-82--85%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

End-to-end ML pipeline predicting passenger survival on the Titanic — EDA, preprocessing, model training, evaluation, and serialization.

---

## Business Problem

Given passenger attributes (age, sex, class, fare), predict who survived. Binary classification mapping directly to real-world use cases like churn prediction and loan default.

---

## Dataset

| Property | Value |
|---|---|
| Source | Kaggle Titanic Competition |
| Rows | 891 passengers |
| Target | Survived (0 = No, 1 = Yes) |
| Key Features | Age, Sex, Pclass, Fare, SibSp, Parch, Embarked |

---

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~79% |
| **Random Forest** | **~82-85%** |

**Key Insights:**
- Women had a significantly higher survival rate than men
- First-class passengers were ~3x more likely to survive than third-class
- Children under 10 had higher survival rates across all classes

---

## Project Structure

```
titanic-survival-prediction/
├── main.py                  # Full ML pipeline
├── roc_curve_fix.py         # ROC curve helper
├── data/train.csv           # Kaggle dataset
├── models/titanic_model.pkl # Trained model
├── plots/                   # EDA visualizations
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/deepanshu0110/titanic-survival-prediction.git
cd titanic-survival-prediction
pip install -r requirements.txt
# Add train.csv to data/ from Kaggle
python main.py
```

---

## ML Pipeline Steps

1. Data Loading — Pandas CSV ingestion
2. EDA — survival distribution, correlation heatmap
3. Preprocessing — imputation, label encoding, scaling
4. Training — Logistic Regression + Random Forest with CV
5. Evaluation — accuracy, F1, confusion matrix, ROC-AUC
6. Serialization — best model saved as .pkl

---

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn

---

## License

MIT License