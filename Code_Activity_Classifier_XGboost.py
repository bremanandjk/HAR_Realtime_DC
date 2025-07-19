# === Exercise classifier for the handcrafted method ===

import sys
sys.coinit_flags = 0  # 0 means MTA

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier  # <-- XGBoost import
import joblib

# Load CSV
df = pd.read_csv("handcrafted_training_data.csv")

# Ensure last two columns are: 'Exercise_Type' and 'Label'
# Keep only rows labeled as actual reps (1)
df_rep = df[df['Label'] == 1]

# Extract and encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_rep['Exercise_Type'])  # Converts strings to ints

# Extract features
X = df_rep.drop(columns=['Timestamp', 'Exercise_Type', 'Label'])
X = X.select_dtypes(include=[np.number])

# Train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost classifier
clf = XGBClassifier(eval_metric='mlogloss', random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate 
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)

# Save model 
joblib.dump(clf, "XGB_classifier.pkl")

# Save to file
joblib.dump(label_encoder, 'label_encoder.pkl')

# Reporting
print(f"\nModel Accuracy on Test Set: {accuracy:.2f}\n")
print(f"Classification Report:\n{report}")