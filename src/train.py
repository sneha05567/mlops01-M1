from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
