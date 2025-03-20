import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
data_path = '/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv'
data = pd.read_csv(data_path)

# Handle missing values (if any)
if data.isnull().sum().sum() > 0:
    data.fillna(data.median(), inplace=True)

# feature Enginneering
data['Chronic_Condition_Score'] = data[['anaemia', 'diabetes', 'high_blood_pressure']].sum(axis=1)

age_bins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
data['Age_Binned'] = age_bins.fit_transform(data[['age']])

data['EF_Creatinine_Interaction'] = data['ejection_fraction'] * data['serum_creatinine']

data.drop(columns=['age'], inplace=True)

X = data.drop(columns=['DEATH_EVENT'])
y = data['DEATH_EVENT']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Scale feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Generate predictions
y_pred = rf_model.predict(X_test_scaled)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance analysis
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importances, x='Importance', y='Feature', palette="viridis")
plt.title("Feature Importance")
plt.show()

# Compute probabilities for ROC-AUC
y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

# Display best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.2f}")

# Retrain model with the best parameters
optimized_model = grid_search.best_estimator_
optimized_model.fit(X_train_scaled, y_train)

# Evaluate optimized model
y_pred_optimized = optimized_model.predict(X_test_scaled)
optimized_accuracy = accuracy_score(y_test, y_pred_optimized)
print(f"Optimized Model Accuracy: {optimized_accuracy:.2f}")
print("\nOptimized Model Classification Report:")
print(classification_report(y_test, y_pred_optimized))

# Save the optimized model
joblib.dump(optimized_model, 'optimized_rf_model.pkl')

# Save predictions for submission
submission_df = pd.DataFrame({
    'Id': X_test.index,
    'DEATH_EVENT': y_pred_optimized
})
submission_df.to_csv('submission.csv', index=False)

