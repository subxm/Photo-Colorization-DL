import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 128

def preprocess_image(image_path):
    """
    Reads an image and prepares it for the model.
    """

    image = Image.open(image_path).convert("RGB")

    image = image.resize((IMG_SIZE, IMG_SIZE))

    rgb = np.array(image).astype("float32") / 255.0

    gray = tf.image.rgb_to_grayscale(rgb)

    gray = tf.expand_dims(gray, axis=0)

    return gray, rgb