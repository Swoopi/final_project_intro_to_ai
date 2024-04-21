from utilities.data_loader import load_images
from utilities.preprocessing import preprocess_data
from models.perceptron import Perceptron
from models.neural_network import NeuralNetwork

import matplotlib.pyplot as plt

def plot_images(images, labels, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f"Label: {labels[i]}")
        ax.axis('off')
    plt.show()


def main():
    print("Starting")
    # Load data
    digits_data_images, digits_data_labels = load_images('./data/digitdata/trainingimages', './data/digitdata/traininglabels', 28, 28)
    faces_data_images, faces_data_labels = load_images('./data/facedata/facedatatrain', './data/facedata/facedatatrainlabels', 70, 70)

    # Preprocess data (assuming preprocess returns a dict with 'features' and 'labels')
    digits_data = preprocess_data(digits_data_images, digits_data_labels)
    faces_data = preprocess_data(faces_data_images, faces_data_labels)

    # Initialize models
    perceptron = Perceptron()
    neural_network = NeuralNetwork()

    # Train models
    perceptron.train(digits_data['features'], digits_data['labels'])
    neural_network.train(faces_data['features'], faces_data['labels'])

    # Example evaluation (add actual evaluation code)
    print("Evaluation for Perceptron on Digit Data:")
    # perceptron.evaluate(test_features, test_labels)

    print("Evaluation for Neural Network on Face Data:")
    print("Digits data images shape:", digits_data_images.shape)
    print("Digits data labels shape:", digits_data_labels.shape)
    print("Faces data images shape:", faces_data_images.shape)
    print("Faces data labels shape:", faces_data_labels.shape)

    # Check data type
    print("Digits data images type:", digits_data_images.dtype)
    print("Faces data images type:", faces_data_images.dtype)

    # neural_network.evaluate(test_features, test_labels)
    print("Visualizing Digits:")
    plot_images(digits_data_images[:5], digits_data_labels[:5])

    # Visualize some faces
    print("Visualizing Faces:")
    plot_images(faces_data_images[:5], faces_data_labels[:5], num_images=5)

    # Add additional code to present results, such as accuracy, confusion matrix, etc.

if __name__ == "__main__":
    main()
