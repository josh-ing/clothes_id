import cv2
import json
import numpy as np
import jax.numpy as jnp

def load_dataset_annotation(json_path, num_classes):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    image_to_labels = {}
    for annotation in data['annotations']:
        image_id = annotation['imageId']
        label_ids = [int(label_id) for label_id in annotation['labelId']]

        multi_hot_label = np.zeros(num_classes, dtype=np.float32)
        multi_hot_label[label_ids] = 1.0
        image_to_labels[image_id] = multi_hot_label
    return image_to_labels



def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
    img = cv2.resize(img, target_size)
    img = img.astype(jnp.float32) / 255.0
    return img