import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None

    def train(self, features, labels):
        # Assuming features is a 2D numpy array and labels is a 1D numpy array
        # Adding a column of ones to the features to account for the bias term in the weights
        features = np.hstack([np.ones((features.shape[0], 1)), features])
        
        # Initialize weights if it has not been initialized yet
        if self.weights is None:
            self.weights = np.zeros(features.shape[1])
        
        # Training the perceptron using the Perceptron Learning Algorithm
        for _ in range(self.n_iterations):
            for x, label in zip(features, labels):
                # Compute the prediction using the current weights
                prediction = 1 if np.dot(x, self.weights) >= 0 else 0
                # Update the weights if the prediction is wrong
                if label != prediction:
                    self.weights += self.learning_rate * (label - prediction) * x

        print("Training Perceptron...")

    def predict(self, features):
        # Adding a column of ones to the features to account for the bias term in the weights
        features = np.hstack([np.ones((features.shape[0], 1)), features])
        # Compute the predictions
        predictions = np.where(np.dot(features, self.weights) >= 0, 1, 0)
        return predictions

    def evaluate(self, features, labels):
        predictions = self.predict(features)
        accuracy = sum(1 for true, pred in zip(labels, predictions) if true == pred) / len(labels)
        print(f"Accuracy: {accuracy:.2f}")