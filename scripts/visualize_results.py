#!/usr/bin/env python3
"""
visualize_results.py
--------------------
CANONICAL drawing module for the maritime detection project.

This file owns ONLY visualization logic — no model loading, no inference.
All inference is imported from scripts/evaluate_metrics.py.

Usage (CLI):
    python scripts/visualize_results.py --model yolo
    python scripts/visualize_results.py --model owlvit --prompt "boat, buoy"
    python scripts/visualize_results.py --model dino   --prompt "boat . buoy . obstacle"

Usage (import):
    from scripts.visualize_results import draw_predictions, visualize_and_save
"""

import argparse
import matplotlib.pyplot as plt
from PIL import Image

# ── CHANGE 1: Remove all path definitions, import from config ─────────────────
# OLD: BASE_DIR = Path(__file__).resolve().parents[1]
#      DATA_DIR = BASE_DIR / "data" / "images" / "val"
#      SAVE_ROOT = BASE_DIR / "models"
# These were a third independent definition of paths that already exist in config.
try:
    from scripts.config import VAL_DIR, MODELS_DIR
except ModuleNotFoundError:
    from pathlib import Path as _P
    _BASE    = _P(__file__).resolve().parents[1]
    VAL_DIR  = _BASE / "data" / "images" / "val"
    MODELS_DIR = _BASE / "models"

MODELS_DIR.mkdir(exist_ok=True, parents=True)

# ── CHANGE 2: Remove init_models() entirely ───────────────────────────────────
# OLD: init_models() re-implemented full model loading with its own hardcoded
#      weight paths — a third copy of logic already in evaluate_metrics.py.
# NEW: run_visualization() calls load_yolo/load_owl/load_dino from
#      evaluate_metrics.py and passes the loaded models into eval_* functions.
try:
    from scripts.evaluate_metrics import (
        load_yolo, load_owl, load_dino,
        eval_yolo, eval_owl, eval_dino,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from scripts.evaluate_metrics import (
        load_yolo, load_owl, load_dino,
        eval_yolo, eval_owl, eval_dino,
    )


# ═════════════════════════════════════════════════════════════
# DRAWING FUNCTIONS
#
# CHANGE 3: Added draw_predictions() — the function that
#           app_streamlit.py imports and calls for live UI display.
#           This is the Streamlit-facing version: renders to screen
#           via st.pyplot() when called from the app, but can also
#           be used standalone in notebooks.
#
# visualize_and_save() is kept for CLI/batch use — it writes to disk.
# Both share the same core drawing logic to avoid any divergence.
# ═════════════════════════════════════════════════════════════

def _draw_boxes_on_ax(ax, boxes, scores, labels):
    """
    Core box-drawing logic shared by both draw_predictions()
    and visualize_and_save(). Kept private — callers use the
    public functions below.
    """
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, color="lime", linewidth=2
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1, f"{label} {score:.2f}",
            fontsize=8, color="yellow",
            bbox=dict(facecolor="black", alpha=0.5)
        )
    ax.axis("off")


def draw_predictions(image_path, boxes, scores, labels):
    """
    Display annotated detections inline — used by app_streamlit.py.
    Calls st.pyplot() so it renders inside the Streamlit UI.

    Args:
        image_path: str or Path to image
        boxes:      np.ndarray (N, 4) xyxy
        scores:     np.ndarray (N,)
        labels:     list[str] length N
    """
    # Import here so this file stays importable even without Streamlit installed
    import streamlit as st

    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    _draw_boxes_on_ax(ax, boxes, scores, labels)
    st.pyplot(fig)
    plt.close(fig)


def visualize_and_save(model_name, image_path, boxes, scores, labels, save_dir):
    """
    Save annotated detections to disk — used by CLI batch runs and notebooks.

    Args:
        model_name: str, used in the output filename (e.g. "yolo")
        image_path: str or Path to source image
        boxes:      np.ndarray (N, 4) xyxy
        scores:     np.ndarray (N,)
        labels:     list[str] length N
        save_dir:   Path, where to write the output image
    """
    from pathlib import Path
    image_path = Path(image_path)

    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    _draw_boxes_on_ax(ax, boxes, scores, labels)

    fname     = f"{image_path.stem}_{model_name}.jpg"
    save_path = save_dir / fname
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Saved {save_path}")


# ═════════════════════════════════════════════════════════════
# BATCH VISUALIZATION
#
# CHANGE 4: run_visualization() no longer does any inference
#           itself — it calls eval_* from evaluate_metrics.py.
#           The owlvit inline inference block (40+ lines) and
#           the dino inline inference block are both deleted.
#           All three models now follow the same clean pattern:
#               load → eval → visualize_and_save
# ═════════════════════════════════════════════════════════════

def run_visualization(model_name, prompt="boat", n_images=20):
    """
    Run batch visualization for one model over the first n_images val images.

    Args:
        model_name: one of "yolo", "owlvit", "dino"
        prompt:     text prompt for owlvit/dino
                    for owlvit: comma-separated, e.g. "boat, buoy"
                    for dino:   dot-separated,   e.g. "boat . buoy . obstacle"
        n_images:   how many val images to process (default 20)
    """
    save_dir = MODELS_DIR / f"{model_name}_results"
    save_dir.mkdir(exist_ok=True, parents=True)

    image_list = sorted(VAL_DIR.glob("*.jpg"))[:n_images]
    if not image_list:
        print(f"❌ No images found in {VAL_DIR}")
        return
    print(f"Processing {len(image_list)} images → {save_dir}")

    if model_name == "yolo":
        model = load_yolo()
        for img_path in image_list:
            boxes, scores, labels = eval_yolo(img_path, model=model)
            visualize_and_save("yolo", img_path, boxes, scores, labels, save_dir)

    elif model_name == "owlvit":
        processor, model = load_owl()
        # Split comma-separated prompt into list of queries for OWL-ViT
        queries = [q.strip() for q in prompt.split(",") if q.strip()]
        for img_path in image_list:
            boxes, scores, labels = eval_owl(
                img_path, processor=processor, model=model, queries=queries
            )
            visualize_and_save("owlvit", img_path, boxes, scores, labels, save_dir)

    elif model_name == "dino":
        processor, model = load_dino()
        for img_path in image_list:
            boxes, scores, labels = eval_dino(
                img_path, processor=processor, model=model, prompt=prompt
            )
            visualize_and_save("dino", img_path, boxes, scores, labels, save_dir)

    print(f"✔ All results saved to {save_dir}")


# ═════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-visualize detections for one model over the val set."
    )
    parser.add_argument(
        "--model", choices=["yolo", "owlvit", "dino"], required=True
    )
    parser.add_argument(
        "--prompt", type=str, default="boat",
        help="For owlvit: comma-separated nouns. For dino: dot-separated phrase."
    )
    parser.add_argument(
        "--n-images", type=int, default=20,
        help="Number of val images to process."
    )
    args = parser.parse_args()
    run_visualization(args.model, prompt=args.prompt, n_images=args.n_images)