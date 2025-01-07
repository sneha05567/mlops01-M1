from src.train import accuracy_score, train_test_split
from src.train import RandomForestClassifier, load_iris
import unittest


class TestModel(unittest.TestCase):
    def test_accuracy(self):
        # Load data and split into train/test sets
        data = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )
        # Train Random Forest model and make predictions
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        # Assert accuracy is above 0.5
        self.assertGreater(accuracy_score(y_test, predictions), 0.5)


if __name__ == "__main__":
    unittest.main()
