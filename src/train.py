from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import joblib

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
  data.data, data.target, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("RandomForestClassfier Trained on Iris Data Successfully...")

# Evaluate
predictions = model.predict(X_test)
print("Evaluating Model Performance...")
acc = accuracy_score(y_test, predictions)
print(f"Accuracy: {acc}")

# Get the absolute path of parent directory
base_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
model_dir = os.path.join(base_dir, "model")
os.makedirs(model_dir, exist_ok=True)

# Save the model
model_path = os.path.join(model_dir, "random_forest_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved successfully at {model_path}!")
