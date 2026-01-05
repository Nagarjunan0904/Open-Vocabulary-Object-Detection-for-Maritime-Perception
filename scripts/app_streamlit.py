import streamlit as st
from pathlib import Path
from PIL import Image
import torch
from ultralytics import YOLO
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]
VAL_DIR = BASE_DIR / "data/images/val"

DEVICE_GPU = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_CPU = "cpu"     # safer for heavy models


# ============================================================
# LAZY LOADERS  (Only loads when user selects)
# ============================================================

@st.cache_resource
def load_yolo():
    weights = BASE_DIR / "notebooks/runs/detect/yolov8m_stable2/weights/best.pt"
    return YOLO(str(weights))


@st.cache_resource
def load_owlvit():
    model_name = "google/owlvit-base-patch32"
    processor = OwlViTProcessor.from_pretrained(model_name)
    model = OwlViTForObjectDetection.from_pretrained(
        model_name,
        torch_dtype=torch.float32      # avoid mixed precision crash
    ).to(DEVICE_CPU)                  # ← run OWL-ViT on CPU for stability
    return processor, model


@st.cache_resource
def load_dino():
    model_name = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    ).to(DEVICE_CPU)                  # ← run DINO on CPU for stability
    return processor, model


# ============================================================
# INFERENCE HELPERS
# ============================================================

def infer_yolo(model, image_path):
    results = model(str(image_path))[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    labels = ["obstacle"] * len(scores)
    return boxes, scores, labels


def infer_owlvit(processor, model, image_path, query):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[query], images=image, return_tensors="pt").to(DEVICE_CPU)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        target_sizes=[image.size[::-1]],   # height, width
        threshold=0.15
    )[0]

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = [query] * len(scores)

    return boxes, scores, labels


def infer_dino(processor, model, image_path, query):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=[query], return_tensors="pt").to(DEVICE_CPU)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        target_sizes=[image.size[::-1]]
    )[0]

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]
    return boxes, scores, labels


# ============================================================
# DRAWING
# ============================================================

import matplotlib.pyplot as plt

def draw_predictions(image_path, boxes, scores, labels):
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             fill=False, color="lime", linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1, f"{label} {score:.2f}",
                fontsize=8, color="yellow",
                bbox=dict(facecolor="black", alpha=0.5))

    ax.axis("off")
    st.pyplot(fig)


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🚤 Open-Vocabulary Maritime Detection — Stable Mode")
st.write("Choose a model and upload an image to perform detection.")

model_choice = st.selectbox(
    "Choose Model (1 model at a time)", 
    ["YOLOv8 (Closed Set)", "OWL-ViT (Zero-shot)", "Grounding DINO (Phrase Grounding)"]
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

prompt = None
if model_choice != "YOLOv8 (Closed Set)":
    prompt = st.text_input("Enter detection text prompt", "boat")

run_btn = st.button("Run Detection")

# ============================================================
# RUN ON BUTTON CLICK
# ============================================================

if run_btn and uploaded_file is not None:
    img_path = BASE_DIR / "temp_uploaded.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Image", width=500)

    # -------------------------- YOLO --------------------------
    if model_choice == "YOLOv8 (Closed Set)":
        st.subheader("YOLOv8 Detections")
        model = load_yolo()
        boxes, scores, labels = infer_yolo(model, img_path)
        draw_predictions(img_path, boxes, scores, labels)

    # ------------------------ OWL-ViT -------------------------
    elif model_choice == "OWL-ViT (Zero-shot)":
        st.subheader(f"OWL-ViT Detections → \"{prompt}\"")
        processor, model = load_owlvit()
        boxes, scores, labels = infer_owlvit(processor, model, img_path, prompt)
        draw_predictions(img_path, boxes, scores, labels)

    # ---------------------- Grounding DINO --------------------
    elif model_choice == "Grounding DINO (Phrase Grounding)":
        st.subheader(f"GroundingDINO Detections → \"{prompt}\"")
        processor, model = load_dino()
        boxes, scores, labels = infer_dino(processor, model, img_path, prompt)
        draw_predictions(img_path, boxes, scores, labels)