# Titanic Survival Prediction

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-82--85%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

End-to-end binary classification pipeline on the Kaggle Titanic dataset — EDA, preprocessing, model training, evaluation, and serialization.

---

## Business Problem

Given passenger attributes (age, sex, ticket class, fare), predict who survived. The same pipeline applies to real-world tasks: customer churn, loan default, fraud detection.

---

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~79% |
| **Random Forest** | **~82–85%** |

**Key Insights:**
- Women survived at ~3x the rate of men
- First-class passengers were ~3x more likely to survive than third-class
- Children under 10 had above-average survival across all classes

---

## Quickstart

```bash
git clone https://github.com/deepanshu0110/titanic-survival-prediction.git
cd titanic-survival-prediction
pip install -r requirements.txt
# Place Kaggle train.csv in data/
python main.py
```

---

## ML Pipeline

1. EDA — survival distribution, correlation heatmap
2. Preprocessing — imputation, encoding, scaling
3. Training — Logistic Regression + Random Forest with cross-validation
4. Evaluation — accuracy, F1, confusion matrix, ROC-AUC
5. Serialization — best model saved as .pkl

---

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn

---

## Author

**Deepanshu Garg** — Freelance Data Scientist
- GitHub: [@deepanshu0110](https://github.com/deepanshu0110)
- Hire: [freelancer.com/u/deepanshu0110](https://www.freelancer.com/u/deepanshu0110)

MIT License