# Titanic Survival Prediction
[![CI](https://github.com/deepanshu0110/titanic-survival-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/deepanshu0110/titanic-survival-prediction/actions)


![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange?style=flat-square)
![Accuracy](https://img.shields.io/badge/Test%20accuracy-80.4%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

End-to-end binary classification pipeline on the Kaggle Titanic dataset — EDA, preprocessing, model training, evaluation, and serialization.

---

## Business Problem

Given passenger attributes (age, sex, ticket class, fare), predict who survived. The same pipeline applies to real-world tasks: customer churn, loan default, fraud detection.

---

## Results

Kaggle `train.csv` (891 passengers), 80/20 split. The script selects the best model by cross-validation score.

| Model | Test accuracy | Test F1 | CV accuracy |
|---|---|---|---|
| Random Forest | 80.4% | 0.733 | 79.2% (± 6.8) |
| **Logistic Regression** | **80.4%** | **0.737** | **80.1% (± 5.7)** |

Both models tie on the test set. Logistic Regression wins on cross-validation, so it is the saved model.

**Key insights** (from the training data):
- Women survived at 74%, men at 19%, about 3.9 times the rate
- First class survived at 63%, third class at 24%, about 2.6 times the rate
- Children under 10 survived at 61%, above the 38% average, and above their class average in every class

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