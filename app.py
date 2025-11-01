import os
from typing import List

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf


@st.cache_resource
def load_model(model_path: str):
    """Load and return a Keras model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path)


def load_class_names_from_dir(data_dir: str) -> List[str]:
    if not os.path.isdir(data_dir):
        return []
    # List directories only (classes)
    items = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    items.sort()
    return items


def preprocess_image(img: Image.Image, target_size=(256, 256)) -> np.ndarray:
    img = img.convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img).astype('float32') / 255.0
    return np.expand_dims(arr, axis=0)


def probs_to_topk_names(probs: np.ndarray, class_names: List[str], k: int = 3):
    """Return list of (name, prob) tuples for top-k predictions for a single sample."""
    idxs = np.argsort(-probs)[:k]
    results = []
    for i in idxs:
        name = class_names[i] if i < len(class_names) else f"Class_{i}"
        results.append((name, float(probs[i])))
    return results


def main():
    st.set_page_config(page_title="Mini Paddy-Doctor Classifier", layout="centered")
    st.title("Mini image classifier — show predicted class names & probabilities")

    st.sidebar.header("Settings")
    default_model = "best_model.h5"
    model_path = st.sidebar.text_input("Model path", default_model)

    # optional: data directory to infer class names automatically
    default_data_dir = r"C:\Users\Hp\Documents\MCE411 Assignment\paddy-doctor-diseases-small-augmented-26k"
    data_dir = st.sidebar.text_input("Dataset directory (optional, to infer class names)", default_data_dir)

    st.sidebar.markdown("Provide class names (one per line) to override inferred names:")
    classes_text = st.sidebar.text_area("Class names (optional)")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    top_k = st.sidebar.slider("Top K", min_value=1, max_value=5, value=3)

    if uploaded_file is None:
        st.info("Upload an image (jpg/png) to get predictions. You can also set model/data paths in the sidebar.")
        st.stop()

    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Couldn't load model: {e}")
        st.stop()

    # Determine class names
    if classes_text.strip():
        class_names = [line.strip() for line in classes_text.replace(',','\n').splitlines() if line.strip()]
    else:
        class_names = load_class_names_from_dir(data_dir)

    # fallback if no class names found
    if not class_names:
        # try to infer number of outputs from model output shape
        try:
            out_shape = model.output_shape
            n_classes = int(out_shape[-1])
        except Exception:
            n_classes = 13
        class_names = [f"Class_{i}" for i in range(n_classes)]

    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)

    # Preprocess and predict
    x = preprocess_image(image)
    probs = model.predict(x)[0]

    topk = probs_to_topk_names(probs, class_names, k=top_k)

    st.subheader("Predictions")
    st.write(f"Top-{top_k} predictions:")
    for name, p in topk:
        st.write(f"{name}: {p:.4f}")

    # Show a bar chart of probabilities for all classes
    try:
        import pandas as pd
        df = pd.DataFrame({'class': class_names, 'probability': probs})
        df = df.sort_values('probability', ascending=False)
        st.bar_chart(df.set_index('class'))
    except Exception:
        st.write("(Could not render chart — pandas may be missing.)")

    st.markdown("---")
    st.caption(f"Model: {model_path} | Number of classes: {len(class_names)}")


if __name__ == "__main__":
    main()
