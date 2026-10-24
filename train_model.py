
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

DATA_FILE = "data/gesture_data.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "gesture_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE)

print("\nDataset shape:", df.shape)
print("\nSamples per gesture:")
print(df["label"].value_counts())

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train classifier
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluate on unseen test samples
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save confusion matrix image
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    xticks_rotation=45,
    cmap="Blues"
)
plt.title("Hand Gesture Recognition - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.show()

# Save trained model
joblib.dump(model, MODEL_FILE)

print(f"\nModel saved to: {MODEL_FILE}")
print("Confusion matrix saved to: confusion_matrix.png")
