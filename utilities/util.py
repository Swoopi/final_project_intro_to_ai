import numpy as np
import matplotlib.pyplot as plt
#! Util.py, a utility file containing helper functions for data loading, preprocessing, and evaluation
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
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, str(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > np.max(cm)/2 else "black")
    plt.show()


def plot_training_history(losses, accuracies):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def visualize_predictions(images, labels, predictions, title="Prediction Visualization With Neural Network and Test data"):
    plt.figure(figsize=(10, 2))
    for i in range(min(5, len(images))):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i].reshape((28, 28)), cmap='gray')
        plt.title(f"Pred: {predictions[i]}\nTrue: {labels[i]}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

def visualize_face_predictions(images, labels, predictions, title="Prediction Visualization"):
    plt.figure(figsize=(10, 2))
    for i in range(min(5, len(images))):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i].reshape((70, 60)), cmap='gray')
        plt.title(f"Pred: {predictions[i]}\nTrue: {labels[i]}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()



