import numpy as np

def sigmoid(x):
    """Sigmoid activation function with numerical stability enhancement."""
    # Clip x to avoid overflow in exp
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize network parameters
        self.input_size = input_size + 1  # Adding 1 for bias node
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

    def feedforward(self, X):
        # Forward propagation through our network
        self.z1 = np.dot(X, self.W1)  # Dot product of X (input) and first set of weights
        self.a1 = sigmoid(self.z1)  # Activation function
        self.z2 = np.dot(self.a1, self.W2)  # Dot product of hidden layer (a1) and second set of weights
        output = sigmoid(self.z2)  # Final activation function
        return output

    def backprop(self, X, y, output):
        # Backward propagate through the network
        self.error = y - output  # Error in output
        self.d_output = self.error * sigmoid_derivative(output)  # Derivative of sigmoid to error
        
        self.error_hidden_layer = self.d_output.dot(self.W2.T)
        self.d_hidden_layer = self.error_hidden_layer * sigmoid_derivative(self.a1)
        
        # Update the weights
        self.W1 += X.T.dot(self.d_hidden_layer)
        self.W2 += self.a1.T.dot(self.d_output)

    def train(self, X, y, epochs=10000):
        X = np.hstack([np.ones((X.shape[0], 1)), X])  # Adding bias unit to the input layer
        for epoch in range(epochs):
            output = self.feedforward(X)
            self.backprop(X, y, output)
            if epoch % 1000 == 0:
                print(f'Epoch {epoch} Loss: {np.mean(np.square(y - output))}')  # Mean squared error

    def predict(self, features):
        features = np.hstack([np.ones((features.shape[0], 1)), features])  # Adding bias unit to the input layer
        predictions = self.feedforward(features)
        return predictions

    def evaluate(self, features, labels):
        predictions = self.predict(features)
        predictions = [1 if p > 0.5 else 0 for p in predictions]  # Threshold predictions
        accuracy = np.mean(np.array(predictions) == labels)
        print(f"Accuracy: {accuracy}")
