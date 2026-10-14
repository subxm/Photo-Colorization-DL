import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# ==========================
# Configuration
# ==========================
MODEL_PATH = "results/colorizer.keras"
IMAGE_SIZE = (128, 128)

# ==========================
# Load Model
# ==========================
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ==========================
# Image Preprocessing
# ==========================
def preprocess_image(image_path):
    """
    Loads and preprocesses an image for the model.
    Returns:
        gray_input -> (1,128,128,1)
        rgb_image  -> (128,128,3)
    """

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    rgb = np.array(image).astype(np.float32) / 255.0

    gray = tf.image.rgb_to_grayscale(rgb)
    gray = tf.expand_dims(gray, axis=0)

    return gray, rgb

# ==========================
# Prediction Function
# ==========================
def colorize_image(image_path):
    gray, original = preprocess_image(image_path)

    prediction = model.predict(gray, verbose=0)[0]

    prediction = np.clip(prediction, 0, 1)

    return original, gray.numpy()[0].squeeze(), prediction

# ==========================
# Test
# ==========================
if __name__ == "__main__":

    IMAGE_PATH = "uploads/test.jpg"

    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Image not found: {IMAGE_PATH}")
        exit()

    original, grayscale, prediction = colorize_image(IMAGE_PATH)

    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(grayscale, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(prediction)
    plt.title("Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.show()