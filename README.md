# Mini Streamlit classifier

This is a small Streamlit app to run your trained Keras model (`best_model.h5`) on uploaded images and display the predicted class names and probabilities.

How to run (Windows PowerShell):

1. Create and activate a Python environment (optional but recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app:

```powershell
streamlit run "c:\Users\Hp\Documents\MCE411 Assignment\app.py"
```

Notes:
- By default the app expects `best_model.h5` in the project root. You can change the model path in the sidebar.
- To provide human-readable class names, either paste them (one per line) in the sidebar or set the dataset directory in the sidebar (the app will list folders inside it).
- The app rescales images to 256x256 and divides by 255. Adjust `app.py` if your model expects a different preprocessing.
