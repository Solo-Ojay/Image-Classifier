import os
from typing import List
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import io
from pathlib import Path

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
    st.title("🌾 Mini Image Classifier - Paddy Rice Diseases")

    st.sidebar.header("Settings")

    # Relative paths (works in Streamlit Cloud)
    default_model = str(Path(__file__).parent / "best_model.h5")
    default_data_dir = str(Path(__file__).parent / "paddy-doctor-diseases-small-augmented-26k")

    model_path = st.sidebar.text_input("Model path", default_model)
    data_dir = st.sidebar.text_input("Dataset directory (optional, to infer class names)", default_data_dir)

    st.sidebar.markdown("Provide class names (one per line) to override inferred names:")
    classes_text = st.sidebar.text_area("Class names (optional)")

    # Prepare local folders
    workspace_root = Path.cwd()
    models_dir = workspace_root / "models"
    demo_dir = Path(data_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load the default or custom model
    if os.path.exists(model_path):
        st.sidebar.success(f"Using model: {model_path}")
    else:
        st.sidebar.error(f"Model not found: {model_path}")
        st.stop()

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    top_k = st.sidebar.slider("Top K", min_value=1, max_value=5, value=3)

    # Demo image selector — allows choosing an image from dataset folder
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
        # Validate dataset directory first
        if not data_dir or not os.path.isdir(data_dir):
            st.sidebar.error("Dataset directory not found. Please update the Dataset directory setting to a valid path.")
            use_demo = False
            demo_images = []
        else:
            # Allow narrowing to a class subfolder
            class_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            class_folders.sort()
            chosen_class = st.sidebar.selectbox("(Optional) Limit to class folder", ["All classes"] + class_folders)

            # Compute root for images listing
            root_for_images = data_dir if chosen_class == "All classes" else os.path.join(data_dir, chosen_class)
            demo_images = list_demo_images(root_for_images)

            if not demo_images:
                st.sidebar.warning("No demo images found in selected dataset path.")
                use_demo = False
            else:
                demo_labels = [os.path.relpath(p, data_dir) for p in demo_images]
                demo_choice = st.sidebar.selectbox("Choose demo image", demo_labels)
                if demo_choice:
                    demo_file_path = os.path.join(data_dir, demo_choice)
                    demo_file_path = os.path.normpath(demo_file_path)

    # If neither uploaded file nor demo chosen, prompt user
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
        class_names = [line.strip() for line in classes_text.replace(',', '\n').splitlines() if line.strip()]
    else:
        class_names = load_class_names_from_dir(data_dir)

    # Fallback if no class names found
    if not class_names:
        try:
            out_shape = model.output_shape
            n_classes = int(out_shape[-1])
        except Exception:
            n_classes = 13
        class_names = [f"Class_{i}" for i in range(n_classes)]

    # Load image bytes
    try:
        file_bytes = None
        if demo_file_path:
            with open(demo_file_path, 'rb') as f:
                file_bytes = f.read()
        else:
            if hasattr(uploaded_file, "read"):
                file_bytes = uploaded_file.read()
            if not file_bytes and hasattr(uploaded_file, "getvalue"):
                file_bytes = uploaded_file.getvalue()
            if not file_bytes and isinstance(uploaded_file, bytes):
                file_bytes = uploaded_file

        if not file_bytes:
            st.error("Image could not be read (empty bytes). Try a different image or choose a demo image.")
            st.stop()

        image = Image.open(io.BytesIO(file_bytes))
        caption = f"Demo: {os.path.relpath(demo_file_path, data_dir)}" if demo_file_path else "Uploaded image"
        st.image(image, caption=caption, use_container_width=True)
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

    # Show a bar chart of probabilities
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
