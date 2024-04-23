import numpy as np

def load_images(file_path, label_path, image_height, image_width):
    with open(file_path, 'r') as file:
        lines = [line.rstrip('\n') for line in file]  # Only remove the newline character at the end

    # Read the label data
    with open(label_path, 'r') as file:
        labels = [int(label.strip()) for label in file]

    # Parse the images
    num_images = len(lines) // image_height
    images = np.zeros((num_images, image_height, image_width), dtype=int)
    
    for i in range(num_images):
        base_index = i * image_height
        for j in range(image_height):
            line = lines[base_index + j]
            # Right-pad the line with spaces if it is shorter than expected
            line = line.ljust(image_width, ' ')
            images[i, j] = [common_mapping(char) for char in line]
    
    return images, np.array(labels)


def common_mapping(char):
    return 1 if char in ('#', '+') else 0