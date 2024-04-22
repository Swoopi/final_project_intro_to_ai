from utilities.data_loader import load_images
from utilities.preprocessing import preprocess_data
from models.perceptron import Perceptron, OneVsAllClassifier
from models.neural_network import NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np

def plot_images(images, labels, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f"Label: {labels[i]} ")
        ax.axis('off')
    plt.show()

def accuracy_score(true_labels, predicted_labels):
    correct_count = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    return correct_count / len(true_labels)

def confusion_matrix(true_labels, predicted_labels, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(true_labels, predicted_labels):
        cm[true][pred] += 1
    return cm


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])  # Direct use of cm.shape since cm is now a numpy array
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Label each cell with the counts
    for i, j in np.ndindex(cm.shape):  # Works correctly as cm is a numpy array
        plt.text(j, i, str(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > np.max(cm)/2 else "black")

    plt.show()


def main():
    print("Starting")
    # Load data
    digits_data_images, digits_data_labels = load_images('./data/digitdata/trainingimages', './data/digitdata/traininglabels', 28, 28)
    faces_data_images, faces_data_labels = load_images('./data/facedata/facedatatrain', './data/facedata/facedatatrainlabels', 70, 70)

    digits_test_images, digits_test_labels = load_images('./data/digitdata/testimages', './data/digitdata/testlabels', 28, 28)
    faces_test_images, faces_test_labels = load_images('./data/facedata/facedatatest', './data/facedata/facedatatestlabels', 70, 70)

    # Preprocess data
    digits_data = preprocess_data(digits_data_images, digits_data_labels)
    faces_data = preprocess_data(faces_data_images, faces_data_labels)
    digits_test_data = preprocess_data(digits_test_images, digits_test_labels)
    faces_test_data = preprocess_data(faces_test_images, faces_test_labels)

    # Initialize models
    perceptron = OneVsAllClassifier(n_classes=10)  # Assuming 10 classes for digits 0-9
    neural_network = NeuralNetwork()

    # Train models
    perceptron.train(digits_data['features'], digits_data['labels'])
    neural_network.train(faces_data['features'], faces_data['labels'])

    # Evaluate models
    print("Evaluation for Perceptron on Digit Data:")
    digits_predictions = perceptron.predict(digits_test_data['features'])
    digits_accuracy = accuracy_score(digits_test_data['labels'], digits_predictions)
    print(f"Digits Accuracy: {digits_accuracy:.2f}")

    print("Evaluation for Neural Network on Face Data:")
    faces_predictions = neural_network.predict(faces_test_data['features'])
    faces_accuracy = accuracy_score(faces_test_data['labels'], faces_predictions)
    print(f"Faces Accuracy: {faces_accuracy:.2f}")

    # Visualizations
    print("Visualizing Digits:")
    plot_images(digits_data_images[:5], digits_data_labels[:5])

    print("Visualizing Faces:")
    plot_images(faces_data_images[:5], faces_data_labels[:5], num_images=5)

    # Confusion matrices
    print("Confusion Matrix for Digits:")
    num_classes_digits = 10  # Fixed number for digit classification
    cm_digits = confusion_matrix(digits_test_data['labels'], digits_predictions, num_classes_digits)
    plot_confusion_matrix(cm_digits)

    print("Confusion Matrix for Faces:")
    num_classes_faces = 2  # Assuming binary classification for faces
    cm_faces = confusion_matrix(faces_test_data['labels'], faces_predictions, num_classes_faces)
    plot_confusion_matrix(cm_faces)

if __name__ == "__main__":
    main()
