import os
from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd

# ---------- Utilities ----------
@st.cache_resource
def load_model(model_path: Path):
    """Safely load a bundled Keras model (.h5)."""
    return tf.keras.models.load_model(model_path)

def preprocess_image(img: Image.Image, size=(256, 256)):
    img = img.convert("RGB").resize(size)
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)

def probs_to_topk_names(probs, class_names, k=3):
    idxs = np.argsort(-probs)[:k]
    return [(class_names[i], float(probs[i])) for i in idxs]

# ---------- Main ----------
def main():
    st.set_page_config(page_title="🌾 Paddy Doctor CNN", layout="centered")
    st.title("🌾 Paddy Rice Disease Classifier")

    # Paths relative to repo root
    model_path = Path("models/best_model.h5")
    demo_dir = Path("demo_images")

    if not model_path.exists():
        st.error("Model file not found. Please make sure 'models/best_model.h5' is in your repo.")
        st.stop()

    # Load model
    model = load_model(model_path)

    # Infer number of classes
    try:
        n_classes = model.output_shape[-1]
    except Exception:
        n_classes = 13
    class_names = [f"Class_{i}" for i in range(n_classes)]

    # Sidebar options
    st.sidebar.header("Options")
    top_k = st.sidebar.slider("Top-K Predictions", 1, 5, 3)

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Demo selector
    demo_images = [p for p in demo_dir.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    selected_demo = st.sidebar.selectbox(
        "Choose demo image", ["None"] + [str(p.relative_to(demo_dir)) for p in demo_images]
    )

    # Load chosen image
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    elif selected_demo != "None":
        image = Image.open(demo_dir / selected_demo)
        st.image(image, caption=f"Demo Image: {selected_demo}", use_container_width=True)
    else:
        st.info("Upload an image or pick a demo image from the sidebar.")
        st.stop()

    # Predict
    x = preprocess_image(image)
    probs = model.predict(x)[0]
    topk = probs_to_topk_names(probs, class_names, k=top_k)

    # Display results
    st.subheader("🔍 Predictions")
    for name, p in topk:
        st.write(f"**{name}** — {p:.4f}")

    df = pd.DataFrame({"class": class_names, "probability": probs}).sort_values(
        "probability", ascending=False
    )
    st.bar_chart(df.set_index("class"))

    st.caption(f"Model: {model_path}  |  Classes: {len(class_names)}")


if __name__ == "__main__":
    main()
