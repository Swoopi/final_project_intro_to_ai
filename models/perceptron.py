class Perceptron:
    def __init__(self):
        self.weights = None

    def train(self, features, labels):
        # Placeholder for training logic
        print("Training Perceptron...")

    def predict(self, features):
        # Placeholder for prediction logic
        return [0] * len(features)

    def evaluate(self, features, labels):
        # Placeholder for evaluation logic
        predictions = self.predict(features)
        accuracy = sum(1 for true, pred in zip(labels, predictions) if true == pred) / len(labels)
        print(f"Accuracy: {accuracy}")
