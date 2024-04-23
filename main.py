# main.py
from utilities.data_loader import load_images
from utilities.preprocessing import preprocess_data
from models.perceptron import Perceptron, OneVsAllClassifier
from models.neural_network import NeuralNetwork
from utilities.util import plot_images, accuracy_score, confusion_matrix, plot_confusion_matrix, plot_training_history, visualize_predictions, visualize_face_predictions
import numpy as np


def one_hot_encode(labels, num_classes):
    """ Convert labels to one-hot encoded format. """
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

def main():
    print("Starting")
    # Load training, validation, and test data for digits and faces
    digits_data_images, digits_data_labels = load_images('./data/digitdata/trainingimages', './data/digitdata/traininglabels', 28, 28)
    faces_data_images, faces_data_labels = load_images('./data/facedata/facedatatrain', './data/facedata/facedatatrainlabels', 70, 60)
    digits_val_images, digits_val_labels = load_images('./data/digitdata/validationimages', './data/digitdata/validationlabels', 28, 28)
    faces_val_images, faces_val_labels = load_images('./data/facedata/facedatavalidation', './data/facedata/facedatavalidationlabels', 70, 60)
    digits_test_images, digits_test_labels = load_images('./data/digitdata/testimages', './data/digitdata/testlabels', 28, 28)
    faces_test_images, faces_test_labels = load_images('./data/facedata/facedatatest', './data/facedata/facedatatestlabels', 70, 60)

    # Flatten and normalize data
    digits_data = preprocess_data(digits_data_images, digits_data_labels)
    faces_data = preprocess_data(faces_data_images, faces_data_labels)
    digits_test_data = preprocess_data(digits_test_images, digits_test_labels)
    faces_test_data = preprocess_data(faces_test_images, faces_test_labels)
    digits_val_data = preprocess_data(digits_val_images, digits_val_labels)
    faces_val_data = preprocess_data(faces_val_images, faces_val_labels)


    # Initialize models
    perceptron_digits = OneVsAllClassifier(n_classes=10)
    perceptron_faces = Perceptron(learning_rate=0.01, n_iterations=1000)
    nn_digits = NeuralNetwork(input_size=28*28, hidden_size=128, output_size=10)
    nn_faces = NeuralNetwork(input_size=70*60, hidden_size=128, output_size=2)

    # Train models on training data with validation
    perceptron_digits.train(digits_data['features'], digits_data['labels'], digits_val_data['features'], digits_val_data['labels'])
    perceptron_faces.train(faces_data['features'], faces_data['labels'].astype(int), faces_val_data['features'], faces_val_data['labels'].astype(int))
    history_digits = nn_digits.train(digits_data['features'], digits_data['labels'], digits_val_data['features'], digits_val_data['labels'], learning_rate=0.01, n_iterations=1000)
    history_faces = nn_faces.train(faces_data['features'], faces_data['labels'], faces_val_data['features'], faces_val_data['labels'], learning_rate=0.01, n_iterations=1000)

    # Evaluate models
    print("Evaluation for Perceptron on Digit Data:")
    digits_predictions = perceptron_digits.predict(digits_test_data['features'])
    digits_accuracy = accuracy_score(digits_test_data['labels'], digits_predictions)
    print(f"Digits Accuracy: {digits_accuracy:.2f}")

    print("Evaluation for Perceptron on Face Data:")
    faces_predictions = perceptron_faces.predict(faces_test_data['features'])
    faces_accuracy = accuracy_score(faces_test_data['labels'], faces_predictions)
    print(f"Faces Accuracy: {faces_accuracy:.2f}")

    print("Evaluation for Neural Network on Digit Data:")
    print(nn_digits.evaluate(digits_test_data['features'], digits_test_data['labels']))

    print("Evaluation for Neural Network on Face Data:")
    print(nn_faces.evaluate(faces_test_data['features'], faces_test_data['labels']))

    # Visualizations
    print("Visualizing Digits:")
    plot_images(digits_data_images[:5], digits_data_labels[:5])
    print("Visualizing Faces:")
    plot_images(faces_data_images[:5], faces_data_labels[:5])

    print("Confusion Matrix for Digits:")
    cm_digits = confusion_matrix(digits_test_data['labels'], digits_predictions, 10)
    plot_confusion_matrix(cm_digits)

    print("Confusion Matrix for Faces:")
    cm_faces = confusion_matrix(faces_test_data['labels'], faces_predictions, 2)
    plot_confusion_matrix(cm_faces)

    plot_training_history(history_digits['loss'], history_digits['accuracy'])
    plot_training_history(history_faces['loss'], history_faces['accuracy'])

    digits_predictions = nn_digits.predict(digits_test_data['features'])
    visualize_predictions(digits_test_data['features'], digits_test_data['labels'], digits_predictions, "Digit Prediction Visualization")

    faces_predictions = nn_faces.predict(faces_test_data['features'])
    visualize_face_predictions(faces_test_data['features'], faces_test_data['labels'], faces_predictions)

if __name__ == "__main__":
    main()
