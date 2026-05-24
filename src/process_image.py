import cv2
import jax.numpy as jnp

def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)

    img = cv2.resize(img, target_size)

    img = img.astype(jnp.float32) / 255.0
    return img