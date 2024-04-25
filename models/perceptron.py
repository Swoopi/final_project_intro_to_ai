import numpy as np
import time

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000, n_classes=2):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.n_classes = n_classes
        self.weights = None

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        # Extend training data with a bias term
        trainingData = np.hstack([np.ones((trainingData.shape[0], 1)), trainingData])
        num_features = trainingData.shape[1]
        if self.weights is None:
            if self.n_classes > 2:
                # Initialize a weight matrix for multiclass classification
                self.weights = np.zeros((self.n_classes, num_features))
            else:
                # Initialize a weight vector for binary classification
                self.weights = np.zeros(num_features)

        best_accuracy = 0
        best_weights = None
        start_time = time.time()

        for _ in range(self.n_iterations):
            for x, label in zip(trainingData, trainingLabels):
                if self.n_classes > 2:
                    # Multiclass case: compute outputs and apply softmax
                    scores = np.dot(self.weights, x)
                    predicted_label = np.argmax(scores)
                else:
                    # Binary case: make a prediction
                    predicted_label = 1 if np.dot(self.weights, x) >= 0 else 0

                if label != predicted_label:
                    # Update rule depends on binary or multiclass case
                    if self.n_classes > 2:
                        # Apply updates for the correct class and the predicted class
                        self.weights[label, :] += self.learning_rate * x
                        self.weights[predicted_label, :] -= self.learning_rate * x
                    else:
                        # Binary classification weight update
                        self.weights += self.learning_rate * (label - predicted_label) * x

            # Validation at the end of each iteration
            val_predictions = self.predict(validationData)
            val_accuracy = np.mean(val_predictions == validationLabels)
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_weights = self.weights.copy()

        if best_weights is not None:
            self.weights = best_weights
        if self.n_classes == 10:
                print("Perceptron Digit Time: --- %s seconds ---" % (time.time() - start_time))
        if self.n_classes == 2:
                print("Perceptron Face Time--- %s seconds ---" % (time.time() - start_time))

            

    def predict(self, features):
        # Add a bias term to the features
        features = np.hstack([np.ones((features.shape[0], 1)), features])
        if self.n_classes > 2:
            scores = np.dot(features, self.weights.T)
            return np.argmax(scores, axis=1)
        else:
            return np.where(np.dot(features, self.weights) >= 0, 1, 0)