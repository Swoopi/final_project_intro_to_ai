# main.py
from utilities.data_loader import load_images
from utilities.preprocessing import preprocess_data
from models.perceptron import Perceptron, OneVsAllClassifier
from models.neural_network import NeuralNetwork
from utilities.util import plot_images, accuracy_score, confusion_matrix, plot_confusion_matrix
import numpy as np

def main():
    print("Starting")
    # Load data for digits and faces
    digits_data_images, digits_data_labels = load_images('./data/digitdata/trainingimages', './data/digitdata/traininglabels', 28, 28)
    faces_data_images, faces_data_labels = load_images('./data/facedata/facedatatrain', './data/facedata/facedatatrainlabels', 70, 70)
    digits_test_images, digits_test_labels = load_images('./data/digitdata/testimages', './data/digitdata/testlabels', 28, 28)
    faces_test_images, faces_test_labels = load_images('./data/facedata/facedatatest', './data/facedata/facedatatestlabels', 70, 70)

    # Preprocess data for digits and faces
    digits_data = preprocess_data(digits_data_images, digits_data_labels)
    faces_data = preprocess_data(faces_data_images, faces_data_labels)
    digits_test_data = preprocess_data(digits_test_images, digits_test_labels)
    faces_test_data = preprocess_data(faces_test_images, faces_test_labels)

    # Initialize models
    perceptron_digits = OneVsAllClassifier(n_classes=10)  # For digit classification
    perceptron_faces = Perceptron(learning_rate=0.01, n_iterations=1000)  # For face detection (binary classification)

    # Train models on training data
    perceptron_digits.train(digits_data['features'], digits_data['labels'])
    perceptron_faces.train(faces_data['features'], faces_data['labels'].astype(int))  # Ensure labels are integers

    # Evaluate models
    print("Evaluation for Perceptron on Digit Data:")
    digits_predictions = perceptron_digits.predict(digits_test_data['features'])
    digits_accuracy = accuracy_score(digits_test_data['labels'], digits_predictions)
    print(f"Digits Accuracy: {digits_accuracy:.2f}")

    print("Evaluation for Perceptron on Face Data:")
    faces_predictions = perceptron_faces.predict(faces_test_data['features'])
    faces_accuracy = accuracy_score(faces_test_data['labels'], faces_predictions)
    print(f"Faces Accuracy: {faces_accuracy:.2f}")

    # Visualize results
    print("Visualizing Digits:")
    plot_images(digits_data_images[:5], digits_data_labels[:5])

    print("Visualizing Faces:")
    plot_images(faces_data_images[:5], faces_data_labels[:5])

    # Display confusion matrices
    print("Confusion Matrix for Digits:")
    cm_digits = confusion_matrix(digits_test_data['labels'], digits_predictions, 10)
    plot_confusion_matrix(cm_digits)

    print("Confusion Matrix for Faces:")
    cm_faces = confusion_matrix(faces_test_data['labels'], faces_predictions, 2)
    plot_confusion_matrix(cm_faces)

if __name__ == "__main__":
    main()
