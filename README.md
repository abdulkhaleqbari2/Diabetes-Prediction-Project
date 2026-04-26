#  Diabetes Prediction — Machine Learning Classification

A supervised machine learning project that predicts the likelihood of diabetes in patients using clinical diagnostic measurements, with a full ML pipeline from data cleaning to multi-model evaluation.


📌 Project Overview
This project applies supervised classification techniques to the Pima Indians Diabetes Dataset to predict whether a patient has diabetes based on key health indicators. It walks through the complete data science workflow — from exploratory analysis and preprocessing to training and comparing multiple ML models.

🎯 Objective
Build and evaluate multiple classification models to predict diabetes onset (Outcome: 0 = No Diabetes, 1 = Diabetes), and identify the best-performing model based on accuracy and classification metrics.

📂 Dataset

Source: Pima Indians Diabetes Dataset
Features: 8 clinical attributes
Target: Outcome (Binary: 0 or 1)

FeatureDescriptionPregnanciesNumber of pregnanciesGlucosePlasma glucose concentrationBloodPressureDiastolic blood pressure (mm Hg)SkinThicknessTriceps skin fold thickness (mm)Insulin2-Hour serum insulin (mu U/ml)BMIBody mass indexDiabetesPedigreeFunctionDiabetes pedigree function (genetic influence)AgeAge in years

🔧 Tech Stack
Show Image
Show Image
Show Image
Show Image
Show Image

Language: Python 3
Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn


🔍 Project Workflow
1. Exploratory Data Analysis (EDA)

Distribution plots for Age, SkinThickness, and Pregnancies
Target class distribution (pie chart)
Group-wise aggregations by Outcome (mean glucose, pregnancies, etc.)
Correlation heatmap to identify feature relationships

2. Data Preprocessing

Zero-value Treatment: Replaced biologically invalid zeros (e.g., Glucose = 0, BMI = 0) with NaN
Missing Value Imputation: Filled nulls using outcome-stratified median imputation — separate medians computed for diabetic and non-diabetic groups to preserve class-specific patterns
Outlier Detection & Removal: Applied the IQR method across all features; visualized with boxplots before and after

3. Feature Engineering & Splitting

Feature matrix X: 8 clinical variables
Target vector y: Outcome
80/20 Train-Test Split with random_state=42
Feature Scaling: StandardScaler applied to normalize feature ranges

4. Model Training & Evaluation
Five classification algorithms were trained and compared:
ModelHighlightsLogistic RegressionBaseline linear classifierK-Nearest Neighbors (KNN)Distance-based non-parametric classifierSupport Vector Machine (SVM)Tuned via GridSearchCV (best: C=10, gamma=0.1)Decision TreeTuned with GridSearchCV over depth, split criteria, and featuresRandom ForestEnsemble model with tuned hyperparameters (n_estimators=130, max_depth=15)
Each model was evaluated using:

Accuracy Score
Confusion Matrix
Classification Report (Precision, Recall, F1-Score)

5. Model Comparison
All model accuracies compiled into a summary DataFrame, ranked from best to worst.

📊 Results Summary
ModelAccuracyRandom Forest Classifier⭐ BestSupport Vector Machine✅ StrongLogistic Regression✅ SolidK-Nearest Neighbors🔵 GoodDecision Tree🔵 Good (post-tuning)

Exact accuracy values depend on the cleaned dataset size after outlier removal.



📁 Project Structure
diabetes-prediction/
│
├── Project6.ipynb        # Main Jupyter Notebook
├── README.md             # Project documentation
└── data/
    └── diabetes.csv      # Dataset (add locally)

💡 Key Learnings

Handling medically invalid zero values requires domain-aware imputation, not simple mean/median replacement
Outcome-stratified imputation preserves class distribution signals during preprocessing
Hyperparameter tuning with GridSearchCV significantly improves model performance
Ensemble methods (Random Forest) consistently outperform single estimators on tabular medical data
