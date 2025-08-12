# diabetes_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)
from sklearn.metrics import f1_score, roc_auc_score


# ------------------------
# 1. Load dataset
# ------------------------
df = pd.read_csv('diabetes.csv')

# ------------------------
# 2. Treat zero values as missing and impute with median
# ------------------------
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace zeros with NaN
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

# Impute using median (safe assignment for pandas 3.0+)
for col in cols_with_zeros:
    df[col] = df[col].fillna(df[col].median())

# ------------------------
# 3. Standardize features (except Outcome)
# ------------------------
target_col = 'Outcome'
feature_cols = df.columns.drop(target_col)

scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Save scaler for later use
joblib.dump(scaler, 'standard_scaler.pkl')

# ------------------------
# 4. Train-test split (80/20)
# ------------------------
X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# ------------------------
# 5. Train models
# ------------------------
logistic_model = LogisticRegression(random_state=42)
rf_model = RandomForestClassifier(random_state=42)

logistic_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# ------------------------
# 6. Predictions
# ------------------------
y_pred_log = logistic_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# ------------------------
# 7. Evaluation
# ------------------------
# print("\n=== Logistic Regression ===")
# print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
# print(classification_report(y_test, y_pred_log))

# print("\n=== Random Forest ===")
# print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
# print(classification_report(y_test, y_pred_rf))

# ------------------------
# 8. Save models
# ------------------------
joblib.dump(logistic_model, 'logistic_regression_model.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')

print("\nModels and scaler saved successfully.")


def evaluate_classification_model(model, X_test, y_test, model_name):
    # Predict class labels
    y_pred = model.predict(X_test)
    
    # Predict probabilities for ROC curve and AUC
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n=== {model_name} Evaluation Metrics ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

# Example usage assuming:
# logistic_model, rf_model, X_test, y_test are already defined and trained

# evaluate_classification_model(logistic_model, X_test, y_test, "Logistic Regression")
# evaluate_classification_model(rf_model, X_test, y_test, "Random Forest")


# After evaluating models with the evaluate_classification_model function
# and having logistic_model, rf_model, X_test, y_test available:



# Predict on test set for both models
y_pred_log = logistic_model.predict(X_test)
y_proba_log = logistic_model.predict_proba(X_test)[:, 1]

y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Calculate F1-score and AUC for Logistic Regression
f1_log = f1_score(y_test, y_pred_log)
auc_log = roc_auc_score(y_test, y_proba_log)

# Calculate F1-score and AUC for Random Forest
f1_rf = f1_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_proba_rf)

print("\nModel Comparison based on F1-score and AUC score:")
print(f"Logistic Regression - F1 Score: {f1_log:.4f}, AUC Score: {auc_log:.4f}")
print(f"Random Forest       - F1 Score: {f1_rf:.4f}, AUC Score: {auc_rf:.4f}")

# Define a combined score or select based on whichever metric you prioritize.
# Here, we simply average both metrics for each model.
combined_log = (f1_log + auc_log) / 2
combined_rf = (f1_rf + auc_rf) / 2

if combined_rf > combined_log:
    best_model = rf_model
    best_model_name = 'Random Forest'
else:
    best_model = logistic_model
    best_model_name = 'Logistic Regression'

print(f"\n✅ Best performing model selected: {best_model_name}")

# Save the best model and the scaler
joblib.dump(best_model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("💾 Saved best model as 'diabetes_model.pkl' and scaler as 'scaler.pkl'")
