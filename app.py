import os
from typing import List

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import io

# model_path = r'C:\Users\Hp\Documents\MCE411 Assignment\best_model.h5'
# data_dir = r'C:\Users\Hp\Documents\MCE411 Assignment\paddy-doctor-diseases-small-augmented-26k'
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
    st.title("Mini image classifier - Paddy Rice Diseases")

    st.sidebar.header("Settings")
    default_model = r"C:\Users\Hp\Documents\MCE411 Assignment\best_model.h5"
    model_path = st.sidebar.text_input("Model path", default_model)

    # optional: data directory to infer class names automatically
    default_data_dir = r"C:\Users\Hp\Documents\MCE411 Assignment\paddy-doctor-diseases-small-augmented-26k"
    data_dir = st.sidebar.text_input("Dataset directory (optional, to infer class names)", default_data_dir)

    st.sidebar.markdown("Provide class names (one per line) to override inferred names:")
    classes_text = st.sidebar.text_area("Class names (optional)")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    top_k = st.sidebar.slider("Top K", min_value=1, max_value=5, value=3)

    # Demo image selector (optional) — allows choosing an image from the dataset folder
    use_demo = st.sidebar.checkbox("Use demo image from dataset (instead of upload)")
    demo_choice = None
    demo_file_path = None

    def list_demo_images(root_dir: str, exts=('.jpg', '.jpeg', '.png', '.bmp'), max_items=500):
        results = []
        if not root_dir or not os.path.isdir(root_dir):
            return results
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.lower().endswith(exts):
                    results.append(os.path.join(dirpath, fname))
                    if len(results) >= max_items:
                        return results
        return results

    if use_demo:
        demo_images = list_demo_images(data_dir)
        if not demo_images:
            st.sidebar.warning("No demo images found in dataset directory. Check Dataset directory setting.")
            use_demo = False
        else:
            # show a shorter label for readability
            demo_labels = [os.path.relpath(p, data_dir) for p in demo_images]
            demo_choice = st.sidebar.selectbox("Choose demo image", demo_labels)
            if demo_choice:
                # compute full path
                demo_file_path = os.path.join(data_dir, demo_choice)
                # normalize path
                demo_file_path = os.path.normpath(demo_file_path)

    # If neither uploaded file nor demo chosen, prompt the user
    if (not uploaded_file) and (not demo_file_path):
        st.info("Upload an image (jpg/png) or select a demo image from the sidebar to get predictions.")
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

    # Load image bytes either from uploaded file or from chosen demo path
    try:
        file_bytes = None
        if demo_file_path:
            # Read from disk
            try:
                with open(demo_file_path, 'rb') as f:
                    file_bytes = f.read()
            except Exception as e:
                st.error(f"Could not read demo image from disk: {e}")
                st.stop()
        else:
            # uploaded_file may be a BytesIO-like object or a Streamlit UploadedFile
            if hasattr(uploaded_file, "read"):
                try:
                    file_bytes = uploaded_file.read()
                except Exception:
                    file_bytes = None

            # Fallback: try getvalue (some file-like objects expose this)
            if not file_bytes and hasattr(uploaded_file, "getvalue"):
                try:
                    file_bytes = uploaded_file.getvalue()
                except Exception:
                    file_bytes = None

            # Final fallback: if uploaded_file is bytes, use it directly
            if not file_bytes and isinstance(uploaded_file, bytes):
                file_bytes = uploaded_file

        if not file_bytes:
            st.error("Image could not be read (empty bytes). Try a different image or choose a demo image.")
            st.stop()

        image = Image.open(io.BytesIO(file_bytes))
        st.image(image, caption=(f"Demo: {os.path.relpath(demo_file_path, data_dir)}" if demo_file_path else "Uploaded image"), use_container_width=True)
    except Exception as e:
        st.error(f"Could not open image: {e}")
        st.stop()

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
