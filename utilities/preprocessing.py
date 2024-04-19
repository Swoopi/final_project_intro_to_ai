import numpy as np

def preprocess_data(images, labels):
    # Assuming images is a numpy array of shape (num_samples, height, width)
    # and labels is a numpy array of shape (num_samples,)

    # Flatten the images from (num_samples, height, width) to (num_samples, height*width)
    num_samples, height, width = images.shape
    flattened_images = images.reshape(num_samples, height * width)

    # Normalize the pixel values from 0-1
    # Assuming the maximum pixel value is 1 if using binary images from your load_images
    normalized_images = flattened_images.astype(np.float32) / 1.0

    # Return a dictionary containing the features and labels
    return {'features': normalized_images, 'labels': labels}
