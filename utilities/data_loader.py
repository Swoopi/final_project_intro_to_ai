import numpy as np

def common_mapping(char):
    return 1 if char == '#' else 0

def load_images(file_path, label_path, image_height, image_width):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Read the label data
    with open(label_path, 'r') as file:
        labels = file.readlines()
        labels = [int(label.strip()) for label in labels]
    
    # Parse the images
    num_images = len(lines) // image_height
    images = np.zeros((num_images, image_height, image_width), dtype=int)
    
    for i in range(num_images):
        base_index = i * image_height
        for j in range(image_height):
            line = lines[base_index + j].strip()
            if len(line) < image_width:
                # Pad the line if it is shorter than expected
                line += ' ' * (image_width - len(line))
            elif len(line) > image_width:
                # Trim the line if it is longer than expected
                line = line[:image_width]
            images[i, j] = [1 if char == '+' or char == '#' else 0 for char in line]
    
    return images, np.array(labels)


