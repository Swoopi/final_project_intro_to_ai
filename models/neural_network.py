import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))

    def relu(self, z):
        return np.maximum(0, z)

    def derivative_relu(self, z):
        return (z > 0).astype(float)
    
    def cross_entropy_loss(self, output, y):
        m = y.shape[0]
        epsilon = 1e-10
        log_likelihood = -np.log(output[np.arange(m), y] + epsilon)
        loss = np.sum(log_likelihood) / m
        return loss

    
    
    
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    


    def one_hot(self, y):
        one_hot = np.zeros((y.size, self.output_size))
        one_hot[np.arange(y.size), y] = 1
        return one_hot
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y, output):
        m = y.shape[0]
        
        dZ2 = output - self.one_hot(y)
        self.dW2 = np.dot(self.a1.T, dZ2) / m
        self.db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dZ1 = np.dot(dZ2, self.W2.T) * self.derivative_relu(self.z1)
        self.dW1 = np.dot(X.T, dZ1) / m
        self.db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
        
    

        
    
    


    def update(self, learning_rate):
        self.W1 -= learning_rate * self.dW1
        self.W2 -= learning_rate * self.dW2
        self.b1 -= learning_rate * self.db1
        self.b2 -= learning_rate * self.db2

    def train(self, X, y, learning_rate=0.01, n_iterations=1000):
        print(f"Training data types X: {X.dtype}, y: {y.dtype}")
        print(f"Training data shapes X: {X.shape}, y: {y.shape}")
        for i in range(n_iterations):
            output = self.forward(X)
            self.backward(X, y, output)
            self.update(learning_rate)
            if i % 100 == 0:
               loss = self.cross_entropy_loss(output, y)
               print(f"Iteration: {i}, Loss: {loss:.4f}")
        loss = self.cross_entropy_loss(output, y)
        print(f"Final Loss: {loss:.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
    
    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def evaluate(self, X, y):
        acc = self.accuracy(X, y)
        print(f"Accuracy: {acc:.2f}")
        return acc
    

    
    
