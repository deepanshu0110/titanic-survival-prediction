"""
Quick ROC Curve Generator for Titanic Project
Run this after your main.py completes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("🎯 Creating ROC Curve for Titanic Model...")
print("=" * 50)

# Load and process data (simplified version)
df = pd.read_csv('data/train.csv')

# Basic preprocessing
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Encode categorical variables
le_sex = LabelEncoder()
le_embarked = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])
df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

# Prepare features
features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Embarked']
X = df[features]
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# Create ROC curve plot
plt.figure(figsize=(12, 8))

# Create subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Titanic Model Performance with ROC Curves', fontsize=16, fontweight='bold')

# Colors for different models
colors = ['darkorange', 'darkblue', 'darkgreen', 'red']
model_results = {}

for i, (name, model) in enumerate(models.items()):
    print(f"Training {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of survival
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    model_results[name] = {
        'accuracy': accuracy,
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc,
        'y_pred': y_pred
    }
    
    print(f"✅ {name}: Accuracy = {accuracy:.3f}, AUC = {roc_auc:.3f}")

# Plot 1: ROC Curves Comparison
ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.50)')
for i, (name, results) in enumerate(model_results.items()):
    ax1.plot(results['fpr'], results['tpr'], color=colors[i], lw=2, 
             label=f'{name} (AUC = {results["auc"]:.3f})')

ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curves - Model Comparison')
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

# Plot 2: AUC Comparison Bar Chart
model_names = list(model_results.keys())
auc_scores = [model_results[name]['auc'] for name in model_names]
accuracy_scores = [model_results[name]['accuracy'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = ax2.bar(x - width/2, accuracy_scores, width, label='Accuracy', alpha=0.8)
bars2 = ax2.bar(x + width/2, auc_scores, width, label='AUC Score', alpha=0.8)

ax2.set_xlabel('Models')
ax2.set_ylabel('Score')
ax2.set_title('Accuracy vs AUC Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(model_names)
ax2.legend()
ax2.set_ylim(0, 1)

# Add value labels on bars
def add_value_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

add_value_labels(ax2, bars1)
add_value_labels(ax2, bars2)

# Plot 3: Threshold Analysis for best model
best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['auc'])
best_results = model_results[best_model_name]

# Calculate metrics at different thresholds
thresholds = np.linspace(0, 1, 100)
precisions = []
recalls = []
f1_scores = []

for threshold in thresholds:
    y_pred_thresh = (model_results[best_model_name]['fpr'] >= threshold).astype(int)
    # This is simplified - in practice you'd recalculate properly
    
# For now, let's just show the ROC curve with threshold points
fpr, tpr, thresh = roc_curve(y_test, model.predict_proba(X_test)[:, 1])

# Mark some important threshold points
important_thresholds = [0.3, 0.5, 0.7]
for threshold in important_thresholds:
    # Find closest threshold
    idx = np.argmin(np.abs(thresh - threshold))
    ax3.plot(fpr[idx], tpr[idx], 'o', markersize=8, 
             label=f'Threshold = {threshold}')

ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'{best_model_name} ROC')
ax3.plot([0, 1], [0, 1], 'k--', lw=2)
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title(f'ROC Curve with Thresholds - {best_model_name}')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Feature Importance for best model
if hasattr(models[best_model_name], 'feature_importances_'):
    importances = models[best_model_name].feature_importances_
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    ax4.barh(feature_importance['feature'], feature_importance['importance'])
    ax4.set_xlabel('Importance')
    ax4.set_title(f'Feature Importance - {best_model_name}')
else:
    ax4.text(0.5, 0.5, 'Feature Importance\nNot Available for\nLogistic Regression', 
             ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Feature Importance')

plt.tight_layout()
plt.savefig('plots/roc_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ ROC analysis saved: plots/roc_analysis.png")
plt.show()

# Print summary
print("\n" + "=" * 50)
print("🎯 ROC CURVE ANALYSIS COMPLETE!")
print("=" * 50)
print("📊 Model Performance Summary:")
for name, results in model_results.items():
    print(f"   {name:20}: Accuracy = {results['accuracy']:.3f}, AUC = {results['auc']:.3f}")

print(f"\n🏆 Best Model: {best_model_name} (AUC = {model_results[best_model_name]['auc']:.3f})")

print("\n💡 ROC Curve Interpretation:")
print("   • AUC = 0.5: Random guessing")
print("   • AUC = 0.7-0.8: Good model")
print("   • AUC = 0.8-0.9: Very good model")
print("   • AUC > 0.9: Excellent model")
print("   • Your model AUC:", f"{model_results[best_model_name]['auc']:.3f}")

if model_results[best_model_name]['auc'] >= 0.8:
    print("   🎉 Your model is performing very well!")
elif model_results[best_model_name]['auc'] >= 0.7:
    print("   👍 Your model is performing well!")
else:
    print("   📈 Your model has room for improvement!")

print("=" * 50)