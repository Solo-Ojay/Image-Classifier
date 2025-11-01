import os
from typing import List

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import urllib.request
import zipfile
from pathlib import Path
import shutil

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

    # --- Optional remote resources (download model or demo images) ---
    st.sidebar.markdown("### Remote resources (optional)")
    demo_zip_url = st.sidebar.text_input("Demo images zip URL (optional)", value="")

    # prepare local folders
    workspace_root = Path.cwd()
    models_dir = workspace_root / "models"
    demo_dir = workspace_root / "demo_images"
    models_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)

    def download_file(url: str, dest: Path) -> bool:
        """Download URL to dest path. Returns True on success."""
        try:
            with urllib.request.urlopen(url) as resp, open(dest, 'wb') as out_file:
                shutil.copyfileobj(resp, out_file)
            return True
        except Exception as e:
            st.sidebar.error(f"Download failed: {e}")
            return False

    def download_and_extract_zip(url: str, dest_dir: Path) -> bool:
        tmp_zip = dest_dir / "_tmp_download.zip"
        ok = download_file(url, tmp_zip)
        if not ok:
            return False
        try:
            with zipfile.ZipFile(tmp_zip, 'r') as z:
                z.extractall(dest_dir)
            tmp_zip.unlink()
            return True
        except Exception as e:
            st.sidebar.error(f"Failed to extract zip: {e}")
            return False

    # If a default model exists in models/, prefer it and make it the only model
    default_model_file = models_dir / "best_model.h5"
    use_default_model = False
    if default_model_file.exists():
        # force using the bundled default model
        use_default_model = st.sidebar.checkbox("Use repository default model (models/best_model.h5)", value=True)
        if use_default_model:
            model_path = str(default_model_file)
            st.sidebar.info(f"Using default model: {default_model_file}")
        else:
            # if user unchecks, allow other model inputs below
            pass

    # When default model is not used, show model download/upload controls
    if not use_default_model:
        model_url = st.sidebar.text_input("Model download URL (.h5)", value="")

        # Buttons to download resources
        if model_url:
            if st.sidebar.button("Download model from URL"):
                dest_model = models_dir / os.path.basename(model_url.split('?')[0])
                st.sidebar.info(f"Downloading model to {dest_model}")
                if download_file(model_url, dest_model):
                    st.sidebar.success("Model downloaded")
                    # set the model_path to downloaded file for this run
                    model_path = str(dest_model)

        if demo_zip_url:
            if st.sidebar.button("Download demo images zip"):
                st.sidebar.info(f"Downloading demo images to {demo_dir}")
                if download_and_extract_zip(demo_zip_url, demo_dir):
                    st.sidebar.success("Demo images downloaded and extracted")
                    # set the data_dir to demo_dir for this run
                    data_dir = str(demo_dir)

        # Allow uploading a model file directly via the sidebar (useful for deployed apps)
        model_upload = st.sidebar.file_uploader("Upload model file (.h5)", type=["h5", "keras", "hdf5"]) 
        if model_upload is not None:
            try:
                upload_name = getattr(model_upload, 'name', 'uploaded_model.h5')
                dest_model = models_dir / upload_name
                # write bytes to disk
                with open(dest_model, 'wb') as f:
                    f.write(model_upload.read())
                st.sidebar.success(f"Model uploaded and saved to {dest_model}")
                model_path = str(dest_model)
            except Exception as e:
                st.sidebar.error(f"Failed to save uploaded model: {e}")

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
        # Validate dataset directory first
        if not data_dir or not os.path.isdir(data_dir):
            st.sidebar.error("Dataset directory not found. Please update the Dataset directory setting to a valid path.")
            use_demo = False
            demo_images = []
        else:
            # Allow narrowing to a class subfolder for convenience
            class_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            class_folders.sort()
            chosen_class = st.sidebar.selectbox("(Optional) Limit to class folder", ["All classes"] + class_folders)

            # compute root for images listing
            root_for_images = data_dir if chosen_class == "All classes" else os.path.join(data_dir, chosen_class)
            demo_images = list_demo_images(root_for_images)

            if not demo_images:
                st.sidebar.warning("No demo images found in the selected dataset path. Try a different Dataset directory or class folder.")
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
