from sklearn.tree import DecisionTreeClassifier
import joblib

class EvenOddModel:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def train(self):
        X = [[i] for i in range(100)]
        y = [i % 2 for i in range(100)]
        self.model.fit(X, y)
        return True

    def predict(self, number):
        return self.model.predict([[number]])[0]

    def save_model(self, path="model.pkl"):
        joblib.dump(self.model, path)
        return path
