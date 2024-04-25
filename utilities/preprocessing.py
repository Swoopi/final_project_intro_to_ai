import numpy as np

def extract_features(images):
    # Assuming images is a numpy array of shape (num_samples, height, width)
    num_samples, height, width = images.shape
    features = []

    for image in images:
        # Calculate percentage of active pixels in four quadrants
        center_h, center_v = height // 2, width // 2
        quadrants = [
            image[:center_h, :center_v],  # Top-left
            image[:center_h, center_v:],  # Top-right
            image[center_h:, :center_v],  # Bottom-left
            image[center_h:, center_v:],  # Bottom-right
        ]
        quadrant_features = [np.mean(quadrant) for quadrant in quadrants]
        features.append(quadrant_features)

    return np.array(features)


def preprocess_data(images, labels, extract_features_flag=True):
    num_samples, height, width = images.shape
    flattened_images = images.reshape(num_samples, height * width)

    if extract_features_flag:
        # Calculate the quadrant features
        quadrant_features = extract_features(images)
        # Augment the flattened data with the new quadrant features
        augmented_data = np.concatenate((flattened_images, quadrant_features), axis=1)
    else:
        augmented_data = flattened_images  # Use only flattened images for face data

    # Return a dictionary containing the features and labels
    return {'features': augmented_data, 'labels': labels}
