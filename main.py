# main.py
from utilities.data_loader import load_images
from utilities.preprocessing import preprocess_data
from models.perceptron import Perceptron
from models.neural_network import NeuralNetwork
from utilities.util import  accuracy_score, confusion_matrix, plot_confusion_matrix, plot_training_history, plot_training_history
import numpy as np


def main():
    print("Starting")
    
    # Load training, validation, and test data for digits and faces
    print("Loading Data")
    digits_data_images, digits_data_labels = load_images('./data/digitdata/trainingimages', './data/digitdata/traininglabels', 28, 28)
    faces_data_images, faces_data_labels = load_images('./data/facedata/facedatatrain', './data/facedata/facedatatrainlabels', 70, 60)
    digits_val_images, digits_val_labels = load_images('./data/digitdata/validationimages', './data/digitdata/validationlabels', 28, 28)
    faces_val_images, faces_val_labels = load_images('./data/facedata/facedatavalidation', './data/facedata/facedatavalidationlabels', 70, 60)
    digits_test_images, digits_test_labels = load_images('./data/digitdata/testimages', './data/digitdata/testlabels', 28, 28)
    faces_test_images, faces_test_labels = load_images('./data/facedata/facedatatest', './data/facedata/facedatatestlabels', 70, 60)
    
    
    # Flatten and normalize data
    print("Data Loaded, Processing...")
    digits_data = preprocess_data(digits_data_images, digits_data_labels)
    faces_data = preprocess_data(faces_data_images, faces_data_labels, extract_features_flag=False)
    digits_test_data = preprocess_data(digits_test_images, digits_test_labels)
    faces_test_data = preprocess_data(faces_test_images, faces_test_labels, extract_features_flag=False)
    digits_val_data = preprocess_data(digits_val_images, digits_val_labels)
    faces_val_data = preprocess_data(faces_val_images, faces_val_labels, extract_features_flag=False)
    
    
    # Initialize models
    print("Data Processed, Initializing Models...")
    perceptron_digits = Perceptron(learning_rate=0.01, n_iterations=1000, n_classes=10)
    perceptron_faces = Perceptron(learning_rate=0.01, n_iterations=1000)
    nn_digits = NeuralNetwork(input_size=28*28 + 4, hidden_size=128, output_size=10, lambda_reg=0.001)
    nn_faces = NeuralNetwork(input_size=70*60, hidden_size=128, output_size=2, lambda_reg=0.01)

    percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for percent in percentages:
        print(f"Testing with {percent}% of data")
        train_size_digits = int(len(digits_data['features']) * percent / 100)
        train_size_faces = int(len(faces_data['features']) * percent / 100)

        digits_features = digits_data['features'][:train_size_digits]
        digits_labels = digits_data['labels'][:train_size_digits]
        faces_features = faces_data['features'][:train_size_faces]
        faces_labels = faces_data['labels'][:train_size_faces]

        # Train perceptron models
        perceptron_digits.train(digits_features, digits_labels, digits_val_data['features'], digits_val_data['labels'])
        perceptron_faces.train(faces_features, faces_labels.astype(int), faces_val_data['features'], faces_val_data['labels'].astype(int))

        # Train neural network models
        history_digits = nn_digits.train(digits_features, digits_labels, digits_val_data['features'], digits_val_data['labels'], learning_rate=0.01, n_iterations=1000)
        history_faces = nn_faces.train(faces_features, faces_labels, faces_val_data['features'], faces_val_data['labels'], learning_rate=0.01, n_iterations=1000)

        # Evaluate models
        digits_predictions = perceptron_digits.predict(digits_test_data['features'])
        digits_accuracy = accuracy_score(digits_test_data['labels'], digits_predictions)
        print(f"Digits Accuracy: {digits_accuracy:.3f}")

        faces_predictions = perceptron_faces.predict(faces_test_data['features'])
        faces_accuracy = accuracy_score(faces_test_data['labels'], faces_predictions)
        print(f"Faces Accuracy: {faces_accuracy:.3f}")

        print("Evaluation for Neural Network on Digit Data:")
        print(nn_digits.evaluate(digits_test_data['features'], digits_test_data['labels']))
        
        print("Evaluation for Neural Network on Face Data:")
        print(nn_faces.evaluate(faces_test_data['features'], faces_test_data['labels']))

    # Train models on training data with validation
    print("Models Initialized, Training Perceptron...")
    perceptron_digits.train(digits_data['features'], digits_data['labels'], digits_val_data['features'], digits_val_data['labels'])
    perceptron_faces.train(faces_data['features'], faces_data['labels'].astype(int), faces_val_data['features'], faces_val_data['labels'].astype(int))

    print("Perceptron Trained, Training Neural Network...")

    history_digits = nn_digits.train(digits_data['features'], digits_data['labels'], digits_val_data['features'], digits_val_data['labels'], learning_rate=0.01, n_iterations=1000)
    history_faces = nn_faces.train(faces_data['features'], faces_data['labels'], faces_val_data['features'], faces_val_data['labels'], learning_rate=0.01, n_iterations=1000)

    

    # Evaluate models
    print("Evaluation for Perceptron on Digit Data:")
    digits_predictions = perceptron_digits.predict(digits_test_data['features'])
    digits_accuracy = accuracy_score(digits_test_data['labels'], digits_predictions)
    print(f"Digits Accuracy: {digits_accuracy:.3f}")

    print("Evaluation for Perceptron on Face Data:")
    faces_predictions = perceptron_faces.predict(faces_test_data['features'])
    faces_accuracy = accuracy_score(faces_test_data['labels'], faces_predictions)
    print(f"Faces Accuracy: {faces_accuracy:.3f}")

    print("Evaluation for Neural Network on Digit Data:")
    print(nn_digits.evaluate(digits_test_data['features'], digits_test_data['labels']))


    print("Evaluation for Neural Network on Face Data:")
    print(nn_faces.evaluate(faces_test_data['features'], faces_test_data['labels']))

    

    print("Confusion Matrix for Digits:")
    cm_digits = confusion_matrix(digits_test_data['labels'], digits_predictions, 10)
    plot_confusion_matrix(cm_digits)

    print("Confusion Matrix for Faces:")
    cm_faces = confusion_matrix(faces_test_data['labels'], faces_predictions, 2)
    plot_confusion_matrix(cm_faces)

    print("Neural Network Data")
    plot_training_history(history_digits['loss'], history_digits['accuracy'])
    plot_training_history(history_faces['loss'], history_faces['accuracy'])

    digits_predictions = nn_digits.predict(digits_test_data['features'])

    faces_predictions = nn_faces.predict(faces_test_data['features'])


if __name__ == "__main__":
    main()
