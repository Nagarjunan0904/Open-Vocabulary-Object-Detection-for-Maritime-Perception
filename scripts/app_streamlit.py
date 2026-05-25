"""
app_streamlit.py
----------------
Streamlit UI for the maritime open-vocabulary detection project.

This file owns ONLY the UI logic.
All inference is imported from scripts/evaluate_metrics.py.
All drawing is imported from scripts/visualize_results.py.
No model code is defined here.
"""

import streamlit as st
from pathlib import Path
from PIL import Image
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── CHANGE 1: Import paths from config instead of redefining them ─────────────
from scripts.config import DEVICE

# ── CHANGE 2: Import loaders and inference functions from evaluate_metrics ─────
# Previously app_streamlit.py defined infer_yolo(), infer_owlvit(), infer_dino()
# locally — identical copies of functions already in evaluate_metrics.py.
# Now we import the canonical versions directly.
from scripts.evaluate_metrics import (
    load_yolo, load_owl, load_dino,
    eval_yolo, eval_owl, eval_dino,
)

# ── CHANGE 3: Import drawing from visualize_results instead of redefining it ──
# draw_predictions() was defined locally here and was a duplicate of
# visualize_and_save() in visualize_results.py. Now we import it.
from scripts.visualize_results import draw_predictions


# ─────────────────────────────────────────────────────────────────────────────
# LAZY MODEL LOADERS with Streamlit caching
#
# CHANGE 4: @st.cache_resource wraps the load_* functions from evaluate_metrics.
# This is the correct place for caching — the UI layer decides caching policy,
# the inference module stays framework-agnostic.
#
# Previously these were defined here as separate load_yolo/load_owlvit/load_dino
# functions that duplicated the loading logic from evaluate_metrics.py.
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def cached_load_yolo():
    return load_yolo()

@st.cache_resource
def cached_load_owl():
    return load_owl()

@st.cache_resource
def cached_load_dino():
    return load_dino()


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("🚤 Open-Vocabulary Maritime Detection")
st.write("Choose a model and upload an image to perform detection.")

model_choice = st.selectbox(
    "Choose model",
    ["YOLOv8 (Closed Set)", "OWL-ViT (Zero-shot)", "Grounding DINO (Phrase Grounding)"],
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

prompt = None
if model_choice != "YOLOv8 (Closed Set)":
    prompt = st.text_input("Enter detection text prompt", "boat")

run_btn = st.button("Run Detection")

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

if run_btn and uploaded_file is not None:

    # CHANGE 5: Save upload to a temp path inside the project's results dir
    # instead of BASE_DIR root to avoid polluting the project root.
    from scripts.config import RESULTS_DIR
    tmp_path = RESULTS_DIR / "temp_uploaded.jpg"
    tmp_path.write_bytes(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded image", width=500)

    if model_choice == "YOLOv8 (Closed Set)":
        st.subheader("YOLOv8 detections")
        model = cached_load_yolo()
        boxes, scores, labels = eval_yolo(tmp_path, model=model)
        draw_predictions(tmp_path, boxes, scores, labels)

    elif model_choice == "OWL-ViT (Zero-shot)":
        st.subheader(f'OWL-ViT detections → "{prompt}"')
        processor, model = cached_load_owl()
        # CHANGE 6: OWL-ViT needs a list. Split on comma if user typed
        # "boat, buoy" so they get two queries instead of one long string.
        queries = [q.strip() for q in prompt.split(",") if q.strip()]
        boxes, scores, labels = eval_owl(
            tmp_path, processor=processor, model=model, queries=queries
        )
        draw_predictions(tmp_path, boxes, scores, labels)

    elif model_choice == "Grounding DINO (Phrase Grounding)":
        st.subheader(f'GroundingDINO detections → "{prompt}"')
        processor, model = cached_load_dino()
        boxes, scores, labels = eval_dino(
            tmp_path, processor=processor, model=model, prompt=prompt
        )
        draw_predictions(tmp_path, boxes, scores, labels)