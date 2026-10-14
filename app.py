import streamlit as st
import tensorflow as tf
import time
from PIL import Image
import numpy as np
from preprocess import preprocess_image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Image Colorizer",
    page_icon=None,
    layout="wide"
)

# -----------------------------
# Load CSS
# -----------------------------
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("results/colorizer.keras")

model = load_model()

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">

<h1>Deep Learning Image Colorization</h1>

<p>
Automatically restore grayscale photographs using a deep learning model
trained to predict realistic colors.
</p>

<span class="tech">TensorFlow</span>

<span class="tech">Keras</span>

<span class="tech">CNN</span>

</div>
""", unsafe_allow_html=True)

# -----------------------------
# Upload
# -----------------------------
uploaded = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded is not None:

    image_path = "uploads/temp.jpg"

    with open(image_path, "wb") as f:
        f.write(uploaded.read())

        # ----------------------------------
    # AI Processing Pipeline
    # ----------------------------------

    progress = st.progress(0)

    status = st.empty()

    start = time.time()

    status.markdown("**Preparing image...**")
    time.sleep(0.3)
    progress.progress(15)

    gray, original = preprocess_image(image_path)

    status.markdown("**Converting to grayscale...**")
    time.sleep(0.3)
    progress.progress(35)

    status.markdown("**Loading trained model...**")
    time.sleep(0.25)
    progress.progress(55)

    status.markdown("**Running neural network...**")

    prediction = model.predict(gray, verbose=0)[0]

    progress.progress(85)

    status.markdown("**Generating color image...**")

    prediction = np.clip(prediction, 0, 1)

    time.sleep(0.25)

    progress.progress(100)

    elapsed = time.time() - start

    status.success(f"Completed in {elapsed:.2f} seconds")

    time.sleep(0.8)

    progress.empty()
    status.empty()

col1, col2 = st.columns(2, gap="large")

with col1:

    st.markdown(
        """
        <div class="image-card">
            <div class="image-title">Original Image</div>
        """,
        unsafe_allow_html=True,
    )

    st.image(original, use_container_width=True)

    st.markdown(
        """
            <div class="image-caption">
                Uploaded grayscale photograph
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:

    st.markdown(
        """
        <div class="image-card">
            <div class="image-title">Colorized Image</div>
        """,
        unsafe_allow_html=True,
    )

    st.image(prediction, use_container_width=True)

    st.markdown(
        """
            <div class="image-caption">
                Generated using the trained deep learning model
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------
    # Advanced View
    # -----------------------------

    with st.expander("Advanced View"):

        c1, c2, c3 = st.columns(3)

        with c1:
            st.write("Original")
            st.image(original)

        with c2:
            st.write("Grayscale")
            st.image(gray[0].numpy().squeeze(), clamp=True)

        with c3:
            st.write("Prediction")
            st.image(prediction)

    # -----------------------------
    # Information
    # -----------------------------

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="info">

        <b>Model</b> : CNN Image Colorizer

        <br><br>

        <b>Input Size</b> : 128 × 128

        <br><br>

        <b>Processing Time</b> : {elapsed:.2f} seconds

        <br><br>

        <b>Status</b> : Complete

        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------
    # Download
    # -----------------------------

    prediction_uint8 = (prediction * 255).astype(np.uint8)

    prediction_image = Image.fromarray(prediction_uint8)

    prediction_image.save("outputs/colorized.png")

    with open("outputs/colorized.png", "rb") as file:

        st.download_button(
            label="Download Colorized Image",
            data=file,
            file_name="colorized.png",
            mime="image/png",
            use_container_width=True,
        )