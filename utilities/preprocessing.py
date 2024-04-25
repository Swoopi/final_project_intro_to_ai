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




def preprocess_data(images, labels):
    # Flatten the images from (num_samples, height, width) to (num_samples, height*width)
    num_samples, height, width = images.shape
    flattened_images = images.reshape(num_samples, height * width)

    # Calculate the quadrant features
    quadrant_features = extract_features(images)  # This function computes the quadrant features as shown before

    # Augment the flattened data with the new quadrant features
    augmented_data = np.concatenate((flattened_images, quadrant_features), axis=1)

    # Return a dictionary containing the features and labels
    return {'features': augmented_data, 'labels': labels}


