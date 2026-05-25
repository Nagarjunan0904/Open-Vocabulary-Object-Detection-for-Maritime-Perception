#!/usr/bin/env python3
"""
evaluate_metrics.py
--------------------
CANONICAL inference + metrics module for the maritime detection project.

All other files (notebooks, app_streamlit.py, visualize_results.py) should
import from here instead of re-defining their own inference functions.

Usage (CLI):
    python scripts/evaluate_metrics.py
    python scripts/evaluate_metrics.py --num-images 50
    python scripts/evaluate_metrics.py --num-images -1 --prompt "boat . buoy . obstacle"

Usage (import):
    from scripts.evaluate_metrics import load_yolo, eval_yolo, eval_owl, eval_dino, compute_metrics
"""

# ─────────────────────────────────────────────────────────────
# CHANGE 1: Removed all top-level path definitions.
#           They now live in scripts/config.py — single source
#           of truth. Every file imports from there.
# ─────────────────────────────────────────────────────────────
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

# ─────────────────────────────────────────────────────────────
# CHANGE 2: Import all paths and constants from config.py.
#           Create scripts/config.py with these contents:
#
#   from pathlib import Path
#   import torch
#
#   BASE_DIR     = Path(__file__).resolve().parents[1]
#   DATA_DIR     = BASE_DIR / "data"
#   VAL_DIR      = DATA_DIR / "images" / "val"
#   REPORT_DIR   = BASE_DIR / "reports"
#   RESULTS_DIR  = BASE_DIR / "results"
#   YOLO_WEIGHTS = BASE_DIR / "runs" / "yolov8m_stable" / "weights" / "best.pt"
#   OWL_MODEL    = "google/owlvit-base-patch32"
#   DINO_MODEL   = "IDEA-Research/grounding-dino-tiny"
#   DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
#
# ─────────────────────────────────────────────────────────────
try:
    from scripts.config import (
        VAL_DIR, REPORT_DIR,
        YOLO_WEIGHTS, OWL_MODEL, DINO_MODEL, DEVICE,
    )
except ModuleNotFoundError:
    # Fallback when running directly as a script (python scripts/evaluate_metrics.py)
    # so you don't need to set PYTHONPATH manually.
    from pathlib import Path as _P
    _BASE = _P(__file__).resolve().parents[1]
    VAL_DIR      = _BASE / "data" / "images" / "val"
    REPORT_DIR   = _BASE / "reports"
    YOLO_WEIGHTS = _BASE / "runs" / "yolov8m_stable" / "weights" / "best.pt"
    OWL_MODEL    = "google/owlvit-base-patch32"
    DINO_MODEL   = "IDEA-Research/grounding-dino-tiny"
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

REPORT_DIR.mkdir(exist_ok=True, parents=True)


# ═════════════════════════════════════════════════════════════
# CHANGE 3: Model loading is now LAZY — wrapped in functions
#           instead of running at import time.
#
#           OLD problem: importing this module in app_streamlit.py
#           or any notebook would immediately load all 3 models
#           into GPU memory, even if you only needed YOLO.
#
#           NEW behaviour: models only load when you call
#           load_yolo() / load_owl() / load_dino() explicitly.
#           app_streamlit.py wraps these with @st.cache_resource.
#           Notebooks call them once at the top of the relevant cell.
# ═════════════════════════════════════════════════════════════

def load_yolo():
    """Load and return the fine-tuned YOLOv8 model."""
    from ultralytics import YOLO  # local import keeps top-level fast
    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(
            f"YOLOv8 weights not found at {YOLO_WEIGHTS}\n"
            "Check YOLO_WEIGHTS in scripts/config.py"
        )
    return YOLO(str(YOLO_WEIGHTS))


def load_owl():
    """Load and return (processor, model) for OWL-ViT."""
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    processor = OwlViTProcessor.from_pretrained(OWL_MODEL)
    model = OwlViTForObjectDetection.from_pretrained(
        OWL_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)
    return processor, model


def load_dino():
    """Load and return (processor, model) for GroundingDINO."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    processor = AutoProcessor.from_pretrained(DINO_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        DINO_MODEL,
        torch_dtype=torch.float32,
    ).to(DEVICE)
    return processor, model


# ═════════════════════════════════════════════════════════════
# INFERENCE FUNCTIONS
#
# CHANGE 4: Each function now accepts the model as a parameter
#           instead of referencing a global variable.
#
#           OLD:  def eval_yolo(img):  uses global `yolo`
#           NEW:  def eval_yolo(img, model): caller passes model in
#
#           This makes each function independently testable,
#           importable without side effects, and usable in both
#           the Streamlit app and notebooks without modification.
# ═════════════════════════════════════════════════════════════

def eval_yolo(img_path, model):
    """
    Run YOLOv8 inference on a single image.

    Args:
        img_path: str or Path to image file
        model:    loaded YOLO model (from load_yolo())

    Returns:
        boxes:  np.ndarray shape (N, 4) in xyxy format
        scores: np.ndarray shape (N,)
        labels: list[str] of length N
    """
    with torch.no_grad():
        res = model(str(img_path))[0]
    boxes  = res.boxes.xyxy.cpu().numpy()
    scores = res.boxes.conf.cpu().numpy()
    labels = ["obstacle"] * len(scores)
    return boxes, scores, labels


def eval_owl(img_path, processor, model, queries):
    """
    Run OWL-ViT zero-shot inference on a single image.

    Args:
        img_path:  str or Path to image file
        processor: loaded OwlViTProcessor (from load_owl())
        model:     loaded OwlViTForObjectDetection (from load_owl())
        queries:   list[str] of category names, e.g. ["boat", "buoy"]

    Returns:
        boxes:  np.ndarray shape (N, 4) in xyxy format
        scores: np.ndarray shape (N,)
        labels: list[str] of length N
    """
    image = Image.open(img_path).convert("RGB")

    # ─────────────────────────────────────────────────────────
    # CHANGE 5: Guard kept from original — OWL-ViT crashes if
    #           given fewer than 2 queries. Fallback ensures
    #           the function never silently fails.
    # ─────────────────────────────────────────────────────────
    if len(queries) < 2:
        queries = ["boat", "buoy", "dock"]

    inputs = processor(
        text=queries, images=image, return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)
    processed = processor.post_process_grounded_object_detection(
        outputs, target_sizes=target_sizes
    )[0]

    boxes  = processed["boxes"].cpu().numpy()
    scores = processed["scores"].cpu().numpy()
    labels = processed["labels"]
    return boxes, scores, labels


def eval_dino(img_path, processor, model, prompt):
    """
    Run GroundingDINO phrase-grounding inference on a single image.

    Args:
        img_path:  str or Path to image file
        processor: loaded AutoProcessor (from load_dino())
        model:     loaded AutoModelForZeroShotObjectDetection (from load_dino())
        prompt:    str, e.g. "boat . buoy . obstacle"

    Returns:
        boxes:  np.ndarray shape (N, 4) in xyxy format
        scores: np.ndarray shape (N,)
        labels: list[str] of length N
    """
    image = Image.open(img_path).convert("RGB")
    inputs = processor(
        images=image, text=[prompt], return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs, target_sizes=[image.size[::-1]]
    )[0]

    boxes  = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]
    return boxes, scores, labels


# ═════════════════════════════════════════════════════════════
# METRICS AGGREGATOR
#
# CHANGE 6: compute_metrics() signature updated to match the new
#           model-as-parameter convention. eval_fn now receives
#           the model via **kwargs just like before, but callers
#           pass the model object in kwargs explicitly.
#
#           Example:
#               model = load_yolo()
#               compute_metrics(eval_yolo, images, model=model)
#
#               proc, mdl = load_owl()
#               compute_metrics(eval_owl, images,
#                               processor=proc, model=mdl,
#                               queries=["boat", "buoy"])
# ═════════════════════════════════════════════════════════════

def compute_metrics(eval_fn, images, **kwargs):
    """
    Run eval_fn over a list of images and aggregate detection statistics.

    Args:
        eval_fn: one of eval_yolo, eval_owl, eval_dino
        images:  list of Path objects pointing to images
        **kwargs: passed directly to eval_fn (model, processor, queries, prompt…)

    Returns:
        dict with keys: num_images, num_detections, avg_confidence,
                        max_confidence, avg_inference_time_ms

    Timing note:
        On CUDA, uses torch.cuda.Event for GPU-accurate millisecond timing
        (same approach as NB04). Falls back to time.time() on CPU so the
        function works identically without a GPU.
    """
    total_det  = 0
    confs      = []
    times      = []
    use_cuda   = torch.cuda.is_available()

    for img in images:
        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)
            start_event.record()
            boxes, scores, _ = eval_fn(img, **kwargs)
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)   # ms, GPU-accurate
        else:
            t0 = time.time()
            boxes, scores, _ = eval_fn(img, **kwargs)
            elapsed = (time.time() - t0) * 1000             # ms, wall clock

        total_det += len(scores)
        confs.extend(scores.tolist() if hasattr(scores, "tolist") else list(scores))
        times.append(elapsed)

    return {
        "num_images":            len(images),
        "num_detections":        total_det,
        "avg_confidence":        float(np.mean(confs)) if confs else 0.0,
        "max_confidence":        float(np.max(confs))  if confs else 0.0,
        "avg_inference_time_ms": float(np.mean(times)) if times else 0.0,
    }


# ═════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate all three models on the maritime validation set."
    )
    parser.add_argument(
        "--num-images", type=int, default=200,
        help="Number of val images to sample. Use -1 for all."
    )
    parser.add_argument(
        "--prompt", type=str, default="boat . buoy . obstacle",
        help="Dot-separated phrase prompt for GroundingDINO."
    )
    parser.add_argument(
        "--queries", type=str, nargs="+",
        default=["boat", "buoy", "dock", "obstacle"],
        help="List of category names for OWL-ViT."
    )
    args = parser.parse_args()

    # Collect images
    images = sorted(VAL_DIR.glob("*.jpg"))
    if args.num_images > 0:
        images = images[: args.num_images]
    print(f"Running evaluation on {len(images)} images from {VAL_DIR}")

    # ─────────────────────────────────────────────────────────
    # CHANGE 7: Models are now loaded explicitly here in __main__
    #           instead of at module import. Importing this file
    #           from a notebook or app no longer triggers any
    #           model downloads or GPU allocations.
    # ─────────────────────────────────────────────────────────
    print("Loading YOLOv8...")
    yolo_model = load_yolo()

    print("Loading OWL-ViT...")
    owl_proc, owl_model = load_owl()

    print("Loading GroundingDINO...")
    dino_proc, dino_model = load_dino()

    print("Evaluating YOLOv8...")
    y = compute_metrics(eval_yolo, images, model=yolo_model)

    print("Evaluating OWL-ViT...")
    o = compute_metrics(eval_owl, images,
                        processor=owl_proc, model=owl_model,
                        queries=args.queries)

    print("Evaluating GroundingDINO...")
    d = compute_metrics(eval_dino, images,
                        processor=dino_proc, model=dino_model,
                        prompt=args.prompt)

    pd.DataFrame([y]).to_csv(REPORT_DIR / "metrics_yolo.csv",    index=False)
    pd.DataFrame([o]).to_csv(REPORT_DIR / "metrics_owlvit.csv",  index=False)
    pd.DataFrame([d]).to_csv(REPORT_DIR / "metrics_dino.csv",    index=False)

    print(f"\n✔ Metrics saved to {REPORT_DIR}")
    print(f"  YOLOv8  → {y}")
    print(f"  OWL-ViT → {o}")
    print(f"  DINO    → {d}")