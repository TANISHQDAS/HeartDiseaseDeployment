"""
train_model.py
--------------
Task 1: Data Understanding and Preprocessing
Task 2: Model Development

Heart Disease Prediction - Model Training Script
Author: Abhi Pandey (23BAI10909)
"""

import os
import shutil
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import kagglehub

# ---------------------------------------------------------------
# Step 1: Download dataset directly from Kaggle using kagglehub
# ---------------------------------------------------------------
print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
print("Path to dataset files:", path)

# The dataset folder contains heart.csv - locate it automatically
csv_file = None
for file in os.listdir(path):
    if file.endswith(".csv"):
        csv_file = os.path.join(path, file)
        break

if csv_file is None:
    raise FileNotFoundError("No CSV file found in the downloaded Kaggle dataset folder.")

# Copy dataset next to this script so it also gets committed to GitHub
local_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heart.csv")
shutil.copy(csv_file, local_csv)
print(f"Dataset copied to: {local_csv}")

# ---------------------------------------------------------------
# Task 1: Data Understanding and Preprocessing
# ---------------------------------------------------------------

# 1. Load the dataset using Pandas
df = pd.read_csv(local_csv)

# 2. Display the first five records
print("\nFirst 5 records:")
print(df.head())

# 3. Identify numerical features and target variable
target_col = "target"
numerical_features = [col for col in df.columns if col != target_col]
print(f"\nNumerical features ({len(numerical_features)}):", numerical_features)
print(f"Target variable: '{target_col}'")

# 4. Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 5. Split the dataset into 80% training and 20% testing
X = df[numerical_features]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# ---------------------------------------------------------------
# Task 2: Model Development
# ---------------------------------------------------------------

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel: Random Forest Classifier")
print(f"Accuracy Score: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------------
# Save the trained model using Joblib
# ---------------------------------------------------------------
model_bundle = {
    "model": model,
    "feature_names": numerical_features,
    "accuracy": accuracy,
}
joblib.dump(model_bundle, "model.pkl")
print("\nTrained model saved as model.pkl")
