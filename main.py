"""
Titanic Survival Prediction - Main Script
Author: Deepanshu Garg
Date: 2025

This script performs complete analysis and prediction of Titanic passenger survival.
End-to-end Titanic survival prediction with EDA, feature engineering, and model evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class TitanicPredictor:
    def __init__(self):
        """Initialize the Titanic Predictor"""
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Create directories if they don't exist
        os.makedirs('plots', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
    
    def load_data(self):
        """Load the Titanic dataset"""
        print("📊 Loading Titanic dataset...")
        try:
            self.train_data = pd.read_csv('data/train.csv')
            print(f"✅ Training data loaded: {self.train_data.shape}")
            print(f"📋 Columns: {list(self.train_data.columns)}")
            return True
        except FileNotFoundError:
            print("❌ Error: Dataset not found!")
            print("Please download train.csv from Kaggle Titanic competition and place it in the 'data' folder")
            print("Link: https://www.kaggle.com/competitions/titanic/data")
            return False
    
    def explore_data(self):
        """Explore the dataset and create visualizations"""
        print("\n🔍 Exploring the dataset...")
        
        # Basic info
        print("\n📈 Dataset Overview:")
        print(self.train_data.info())
        print("\n📊 Statistical Summary:")
        print(self.train_data.describe())
        
        # Missing values
        print("\n❓ Missing Values:")
        missing_values = self.train_data.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # Survival rate
        survival_rate = self.train_data['Survived'].mean()
        print(f"\n⚡ Overall Survival Rate: {survival_rate:.2%}")
        
        # Create visualizations
        self.create_visualizations()
    
    def create_visualizations(self):
        """Create and save data visualization plots"""
        print("\n📊 Creating visualizations...")
        
        # Set up the plotting area
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Titanic Dataset Exploration', fontsize=16, fontweight='bold')
        
        # 1. Survival Count
        self.train_data['Survived'].value_counts().plot(kind='bar', ax=axes[0,0], color=['red', 'green'])
        axes[0,0].set_title('Survival Count')
        axes[0,0].set_xlabel('Survived (0=No, 1=Yes)')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=0)
        
        # 2. Survival by Gender
        survival_by_sex = pd.crosstab(self.train_data['Sex'], self.train_data['Survived'])
        survival_by_sex.plot(kind='bar', ax=axes[0,1], color=['red', 'green'])
        axes[0,1].set_title('Survival by Gender')
        axes[0,1].set_xlabel('Gender')
        axes[0,1].legend(['Died', 'Survived'])
        axes[0,1].tick_params(axis='x', rotation=0)
        
        # 3. Survival by Passenger Class
        survival_by_class = pd.crosstab(self.train_data['Pclass'], self.train_data['Survived'])
        survival_by_class.plot(kind='bar', ax=axes[0,2], color=['red', 'green'])
        axes[0,2].set_title('Survival by Passenger Class')
        axes[0,2].set_xlabel('Passenger Class')
        axes[0,2].legend(['Died', 'Survived'])
        axes[0,2].tick_params(axis='x', rotation=0)
        
        # 4. Age Distribution
        axes[1,0].hist([self.train_data[self.train_data['Survived']==0]['Age'].dropna(), 
                       self.train_data[self.train_data['Survived']==1]['Age'].dropna()], 
                      bins=30, alpha=0.7, label=['Died', 'Survived'], color=['red', 'green'])
        axes[1,0].set_title('Age Distribution by Survival')
        axes[1,0].set_xlabel('Age')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        
        # 5. Fare Distribution
        axes[1,1].hist([self.train_data[self.train_data['Survived']==0]['Fare'].dropna(), 
                       self.train_data[self.train_data['Survived']==1]['Fare'].dropna()], 
                      bins=30, alpha=0.7, label=['Died', 'Survived'], color=['red', 'green'])
        axes[1,1].set_title('Fare Distribution by Survival')
        axes[1,1].set_xlabel('Fare')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].legend()
        
        # 6. Survival by Embarked Port
        survival_by_embarked = pd.crosstab(self.train_data['Embarked'], self.train_data['Survived'])
        survival_by_embarked.plot(kind='bar', ax=axes[1,2], color=['red', 'green'])
        axes[1,2].set_title('Survival by Embarked Port')
        axes[1,2].set_xlabel('Embarked Port')
        axes[1,2].legend(['Died', 'Survived'])
        axes[1,2].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig('plots/data_exploration.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: plots/data_exploration.png")
        plt.show()
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        print("\n🧹 Cleaning and preprocessing data...")
        
        # Make a copy to avoid modifying original data
        df = self.train_data.copy()
        
        # 1. Handle missing values
        print("🔧 Handling missing values...")
        
        # Age: Fill with median age by passenger class and gender
        df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(
            lambda x: x.fillna(x.median())
        )
        
        # Embarked: Fill with most common port
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        
        # Fare: Fill with median
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        
        # 2. Feature Engineering
        print("⚙️ Creating new features...")
        
        # Family Size
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
        # Is Alone
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # Extract Title from Name
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 
                                          'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        
        # Age Groups
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                               labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        
        # Fare Groups
        df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
        
        # 3. Select features for modeling
        features_to_use = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 
                          'FamilySize', 'IsAlone', 'Title', 'AgeGroup', 'FareGroup']
        
        # Create feature matrix X and target vector y
        X = df[features_to_use].copy()
        y = df['Survived']
        
        # 4. Encode categorical variables
        print("🔤 Encoding categorical variables...")
        
        categorical_features = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup']
        
        for feature in categorical_features:
            le = LabelEncoder()
            X[feature] = le.fit_transform(X[feature].astype(str))
            self.label_encoders[feature] = le
        
        # 5. Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 6. Scale numerical features
        numerical_features = ['Age', 'Fare', 'FamilySize']
        self.X_train[numerical_features] = self.scaler.fit_transform(self.X_train[numerical_features])
        self.X_test[numerical_features] = self.scaler.transform(self.X_test[numerical_features])
        
        print(f"✅ Data preprocessing complete!")
        print(f"📊 Training set: {self.X_train.shape}")
        print(f"📊 Test set: {self.X_test.shape}")
        print(f"🎯 Features used: {list(self.X_train.columns)}")
    
    def train_models(self):
        """Train and evaluate different models"""
        print("\n🤖 Training machine learning models...")
        
        # Define models to try
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        best_model = None
        best_score = 0
        results = {}
        
        for name, model in models.items():
            print(f"\n🔄 Training {name}...")
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            cv_mean = cv_scores.mean()
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_score': cv_mean,
                'predictions': y_pred
            }
            
            print(f"📊 {name} Results:")
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   F1 Score:  {f1:.4f}")
            print(f"   CV Score:  {cv_mean:.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Track best model
            if cv_mean > best_score:
                best_score = cv_mean
                best_model = model
                self.model = model
        
        print(f"\n🏆 Best model: {[k for k, v in results.items() if v['model'] == best_model][0]} (CV Score: {best_score:.4f})")
        
        # Create results visualization
        self.plot_results(results)
        
        return results
    
    def plot_results(self, results):
        """Create visualizations for model results"""
        print("\n📊 Creating result visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Model Comparison
        model_names = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'cv_score']
        
        metric_values = {metric: [results[model][metric] for model in model_names] for metric in metrics}
        
        x = np.arange(len(model_names))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            axes[0,0].bar(x + i*width, metric_values[metric], width, label=metric.capitalize())
        
        axes[0,0].set_xlabel('Models')
        axes[0,0].set_ylabel('Score')
        axes[0,0].set_title('Model Performance Comparison')
        axes[0,0].set_xticks(x + width * 2)
        axes[0,0].set_xticklabels(model_names)
        axes[0,0].legend()
        axes[0,0].set_ylim(0, 1)
        
        # 2. Confusion Matrix for best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_score'])
        cm = confusion_matrix(self.y_test, results[best_model_name]['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1])
        axes[0,1].set_title(f'Confusion Matrix - {best_model_name}')
        axes[0,1].set_xlabel('Predicted')
        axes[0,1].set_ylabel('Actual')
        
        # 3. Feature Importance (for Random Forest)
        if 'Random Forest' in results:
            rf_model = results['Random Forest']['model']
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            axes[1,0].barh(feature_importance['feature'], feature_importance['importance'])
            axes[1,0].set_title('Feature Importance (Random Forest)')
            axes[1,0].set_xlabel('Importance')
        
        # 4. ROC Curve comparison
        from sklearn.metrics import roc_curve, roc_auc_score
        for name_r, res in results.items():
            model_r = res['model']
            if hasattr(model_r, 'predict_proba'):
                y_prob = model_r.predict_proba(self.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(self.y_test, y_prob)
                auc = roc_auc_score(self.y_test, y_prob)
                axes[1, 1].plot(fpr, tpr, label=f"{name_r} (AUC={auc:.3f})")
        axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1, 1].set_xlabel('False Positive Rate')
        axes[1, 1].set_ylabel('True Positive Rate')
        axes[1, 1].set_title('ROC Curve Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/model_results.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: plots/model_results.png")
        plt.show()
    
    def save_model(self):
        """Save the trained model and encoders"""
        print("\n💾 Saving model and preprocessors...")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': list(self.X_train.columns)
        }
        
        with open('models/titanic_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("✅ Model saved as: models/titanic_model.pkl")
    
    def create_readme(self):
        """Create a README file for the project"""
        readme_content = f"""# Titanic Survival Prediction

## Project Overview
This project predicts passenger survival on the Titanic using machine learning.

## Dataset
- **Source**: Kaggle Titanic Competition
- **Size**: {self.train_data.shape[0]} passengers
- **Features**: Age, Sex, Passenger Class, Fare, etc.

## Results Summary
- **Best Model**: Random Forest Classifier
- **Accuracy**: ~82-85%
- **Key Insights**: 
  - Women had higher survival rates than men
  - First-class passengers were more likely to survive
  - Age was a significant factor

## Files
- `main.py`: Main analysis script
- `models/titanic_model.pkl`: Trained model
- `plots/`: Generated visualizations
- `data/`: Dataset files

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Download dataset from Kaggle and place in `data/` folder
3. Run: `python main.py`

## Next Steps
- Try ensemble methods
- Add more feature engineering
- Create a web app for predictions

---
Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open('README.md', 'w') as f:
            f.write(readme_content)
        print("✅ Created: README.md")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚢 TITANIC SURVIVAL PREDICTION PROJECT")
        print("=" * 50)
        
        # Step 1: Load data
        if not self.load_data():
            return
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Preprocess data
        self.preprocess_data()
        
        # Step 4: Train models
        results = self.train_models()
        
        # Step 5: Save model
        self.save_model()
        
        # Step 6: Documentation lives in GitHub README — not regenerated here
        # self.create_readme()  # disabled: would overwrite the curated GitHub README
        
        print("\n" + "=" * 50)
        print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
        print("📁 Check the following outputs:")
        print("   - plots/data_exploration.png")
        print("   - plots/model_results.png") 
        print("   - models/titanic_model.pkl")
        # README is maintained on GitHub, not regenerated locally
        print("\n🚀 Ready to deploy on GitHub!")
        print("=" * 50)

def main():
    """Main function to run the project"""
    # Create and run the predictor
    predictor = TitanicPredictor()
    predictor.run_complete_analysis()

if __name__ == "__main__":
    main()