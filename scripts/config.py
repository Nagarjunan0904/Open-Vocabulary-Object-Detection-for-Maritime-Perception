"""
config.py
---------
Single source of truth for all paths, model names, and device config.

Every other file in this project should import from here:
    from scripts.config import BASE_DIR, VAL_DIR, YOLO_WEIGHTS, ...

DO NOT define BASE_DIR or model paths anywhere else.
"""

from pathlib import Path
import torch

# ── Project root ──────────────────────────────────────────────────────────────
# config.py lives at Implementation/scripts/config.py
# parents[0] = scripts/   parents[1] = Implementation/
BASE_DIR = Path(__file__).resolve().parents[1]

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR    = BASE_DIR / "data"
VAL_DIR     = DATA_DIR / "images" / "val"
TRAIN_DIR   = DATA_DIR / "images" / "train"
ANN_DIR     = DATA_DIR / "annotations"
LABEL_DIR   = DATA_DIR / "labels"

# ── Output paths ──────────────────────────────────────────────────────────────
REPORT_DIR  = BASE_DIR / "reports"
RESULTS_DIR = BASE_DIR / "results"
RUNS_DIR    = BASE_DIR / "runs"
MODELS_DIR  = BASE_DIR / "models"

# ── Weights ───────────────────────────────────────────────────────────────────
# This is the ONE canonical path to your fine-tuned model.
# If you rename the run folder, change it here only.
YOLO_WEIGHTS    = RUNS_DIR / "yolov8m_stable" / "weights" / "best.pt"
YOLO_PRETRAINED = MODELS_DIR / "weights" / "yolov8m.pt"   # original pretrained

# ── HuggingFace model names ───────────────────────────────────────────────────
OWL_MODEL  = "google/owlvit-base-patch32"
DINO_MODEL = "IDEA-Research/grounding-dino-tiny"

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Create output dirs if missing ─────────────────────────────────────────────
for _dir in (REPORT_DIR, RESULTS_DIR, RUNS_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)