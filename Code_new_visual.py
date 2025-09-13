# === Exercise classifier with learning curves and error bands (fixed train/test split) ===

import sys
sys.coinit_flags = 0  # 0 means MTA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

# Load CSV
df = pd.read_csv("handcrafted_training_data.csv")

# Keep only rows labeled as actual reps (1)
df_rep = df[df['Label'] == 1]

# Extract labels (string classes)
y = df_rep['Exercise_Type']

# Extract numeric features (ignore Timestamp, Exercise_Type, Label)
X = df_rep.drop(columns=['Timestamp', 'Exercise_Type', 'Label'])
X = X.select_dtypes(include=[np.number])

# Define training fractions
train_sizes = np.linspace(0.1, 1.0, 10)

# Number of random splits for each training size
n_splits = 5  

# Containers
f1_train_means, f1_train_stds = [], []
f1_test_means, f1_test_stds = [], []
acc_train_means, acc_train_stds = [], []
acc_test_means, acc_test_stds = [], []

for size in train_sizes:
    f1_train_scores, f1_test_scores = [], []
    acc_train_scores, acc_test_scores = [], []

    if size < 1.0:
        sss = StratifiedShuffleSplit(n_splits=n_splits, train_size=size,
                                     test_size=None, random_state=42)
    else:
        # For full dataset, fix train/test to 80/20
        sss = StratifiedShuffleSplit(n_splits=n_splits, train_size=0.8,
                                     test_size=0.2, random_state=42)

    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        # Store metrics
        f1_train_scores.append(f1_score(y_train, y_train_pred, average="macro"))
        f1_test_scores.append(f1_score(y_test, y_test_pred, average="macro"))
        acc_train_scores.append(accuracy_score(y_train, y_train_pred))
        acc_test_scores.append(accuracy_score(y_test, y_test_pred))

    # Aggregate mean and std
    f1_train_means.append(np.mean(f1_train_scores))
    f1_train_stds.append(np.std(f1_train_scores))
    f1_test_means.append(np.mean(f1_test_scores))
    f1_test_stds.append(np.std(f1_test_scores))

    acc_train_means.append(np.mean(acc_train_scores))
    acc_train_stds.append(np.std(acc_train_scores))
    acc_test_means.append(np.mean(acc_test_scores))
    acc_test_stds.append(np.std(acc_test_scores))

# Train final model on full dataset
final_clf = RandomForestClassifier(n_estimators=100, random_state=42)
final_clf.fit(X, y)
joblib.dump(final_clf, "RF_classifier.pkl")

# === Plot Learning Curves with Error Bands ===
plt.figure(figsize=(10, 6))

# F1 Score
plt.plot(train_sizes, f1_train_means, marker='o', color="blue", label="Train F1")
plt.fill_between(train_sizes,
                 np.array(f1_train_means) - np.array(f1_train_stds),
                 np.array(f1_train_means) + np.array(f1_train_stds),
                 color="blue", alpha=0.2)

plt.plot(train_sizes, f1_test_means, marker='o', color="cyan", label="Test F1")
plt.fill_between(train_sizes,
                 np.array(f1_test_means) - np.array(f1_test_stds),
                 np.array(f1_test_means) + np.array(f1_test_stds),
                 color="cyan", alpha=0.2)

# Accuracy
plt.plot(train_sizes, acc_train_means, marker='s', color="red", label="Train Accuracy")
plt.fill_between(train_sizes,
                 np.array(acc_train_means) - np.array(acc_train_stds),
                 np.array(acc_train_means) + np.array(acc_train_stds),
                 color="red", alpha=0.2)

plt.plot(train_sizes, acc_test_means, marker='s', color="orange", label="Test Accuracy")
plt.fill_between(train_sizes,
                 np.array(acc_test_means) - np.array(acc_test_stds),
                 np.array(acc_test_means) + np.array(acc_test_stds),
                 color="orange", alpha=0.2)

plt.xlabel("Training Data Size (fraction)")
plt.ylabel("Score")
plt.title("Learning Curve with Error Bands (RandomForest)")
plt.legend()
plt.grid(True)
plt.show()

# Print last test report for inspection
print("\nFinal Cross-Validation (last split) Test Report:\n",
      classification_report(y_test, y_test_pred))
