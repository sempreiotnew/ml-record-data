# train.py
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import joblib

# --- Load features JSON ---
with open("features.json", "r") as f:
    nested_data = json.load(f)

# --- Flatten list of lists ---
flat_data = [item for sublist in nested_data for item in sublist]

# --- Convert to DataFrame ---
df = pd.DataFrame(flat_data)

# --- Prepare features and labels ---
if 'label' not in df.columns:
    raise ValueError("JSON data must include a 'label' field for supervised training.")

X = df.drop(columns=['label', 'sensor_id'], errors='ignore')  # drop non-feature columns
y = df['label']

# --- Split dataset ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train Random Forest ---
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# --- Predict ---
y_pred = clf.predict(X_test)

# --- Metrics ---
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Save trained model ---
joblib.dump(clf, "random_forest_model.pkl")
print("Model saved as random_forest_model.pkl")
