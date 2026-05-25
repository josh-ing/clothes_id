import numpy as np
import tensorflow as tf
import cv2
from datasets import load_dataset

def preprocess_hf_image(pil_image, target_size=(224, 224)):
    """Converts a Hugging Face PIL Image to a JAX-ready NumPy array."""
    img = np.array(pil_image)
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
    img = cv2.resize(img, target_size)
    return img.astype(np.float32) / 255.0

def create_hf_dataloader(dataset_path, split="train", batch_size=32, num_classes=228):
    """Downloads (or loads cached) HF data and builds a tf.data pipeline."""
    
    print(f"Loading Hugging Face dataset: {dataset_path} ({split} split)...")
    hf_dataset = load_dataset(dataset_path, split=split)
    
    def generator():
        for item in hf_dataset:
            try:
                # Process the image
                img = preprocess_hf_image(item['image'])
                
                labels = np.zeros(num_classes, dtype=np.float32)

                label_ids = [int(x) for x in item['labelId']] 
                labels[label_ids] = 1.0
                
                yield {'image': img, 'labels': labels}
            except Exception:
                continue

    output_signature = {
        'image': tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
        'labels': tf.TensorSpec(shape=(num_classes,), dtype=tf.float32)
    }
    
    tf_dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    tf_dataset = tf_dataset.shuffle(buffer_size=1000)
    tf_dataset = tf_dataset.batch(batch_size, drop_remainder=True)
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    
    return tf_dataset.as_numpy_iterator()