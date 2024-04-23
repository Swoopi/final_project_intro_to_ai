import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None

    def train(self, features, labels, val_features, val_labels):
        features = np.hstack([np.ones((features.shape[0], 1)), features])
        if self.weights is None:
            self.weights = np.zeros(features.shape[1])

        best_accuracy = 0  # Track the best validation accuracy
        best_weights = None
        
        for _ in range(self.n_iterations):
            for x, label in zip(features, labels):
                prediction = 1 if np.dot(x, self.weights) >= 0 else 0
                if label != prediction:
                    self.weights += self.learning_rate * (label - prediction) * x
            val_predictions = self.predict(val_features)
            val_accuracy = np.mean(val_predictions == val_labels)
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_weights = self.weights.copy()
        if best_weights is not None:
            self.weights = best_weights

    def predict(self, features):
        # Add a column of ones to the features for the bias term
        features = np.hstack([np.ones((features.shape[0], 1)), features])
        # Compute the predictions for all features at once
        return np.where(np.dot(features, self.weights) >= 0, 1, 0)

class OneVsAllClassifier:
    def __init__(self, n_classes, learning_rate=0.1, n_iterations=1000):
        self.n_classes = n_classes
        self.perceptrons = [Perceptron(learning_rate, n_iterations) for _ in range(n_classes)]

    def train(self, features, labels, val_features, val_labels):
        for i in range(self.n_classes):
            print(f"Training perceptron for class {i}")
            # Create binary labels for the current class vs all other classes
            binary_labels = (labels == i).astype(int)
            binary_val_labels = (val_labels == i).astype(int)

            self.perceptrons[i].train(features, binary_labels, val_features, binary_val_labels)

    def predict(self, features):
        # Predictions need to be aggregated across all classifiers
        predictions = np.array([perceptron.predict(features) for perceptron in self.perceptrons]).T
        # Choose the class with the highest confidence (the one with the maximum output)
        return np.argmax(predictions, axis=1)

    def evaluate(self, features, labels):
        predictions = self.predict(features)
        accuracy = np.mean(predictions == labels)
        print(f"Accuracy: {accuracy:.2f}")
